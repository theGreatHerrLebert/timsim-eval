"""HYE quant / fold-change eval (P0.2).

Scores whether a search engine recovers the KNOWN per-organism log2 fold-changes of a two-condition
Human/Yeast/E.coli mixture (design.toml), from a JOINT DiaNN run over both conditions' `.d` (so DiaNN does
cross-run normalization / MaxLFQ as one experiment — a post-hoc merge of two independent runs cannot).

Primary metric: peptide-level ``log2(B/A)`` on **organism-unique** sequences (exposes the co-isolation
interference that protein roll-up hides), complete-case (quantified in both), vs the design's expected
log2FC. Per organism we report the **median residual** (``median(log2FC − expected)``, robust to outliers)
and its **MAD**, plus ``%correct`` within a tolerance band, an **eligibility** table (how many sequences are
even measurable per organism) and a **detection** table (A-only / B-only / both) so complete-case
survivorship bias is visible rather than hidden.

Three normalization VIEWS are emitted (never human-anchor the primary — that would force HUMAN's residual to
zero and launder a global error into yeast/ecoli):
  - ``raw``            — DiaNN ``Precursor.Quantity`` (non-normalized);
  - ``normalised``     — DiaNN ``Precursor.Normalised`` (**the primary engine-performance result**);
  - ``human_anchored`` — normalised, then recentred so median HUMAN log2FC = 0 (a DIAGNOSTIC for global
                         scaling only, not the score).

Ratio compression (yeast/ecoli pulled toward 0) is reported as *cross-species-associated ratio compression*,
not proven interference — a true-leakage metric needs per-peak organism provenance (a documented upgrade).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .parsing import replace_I_with_L

ORGANISMS = ["HUMAN", "YEAST", "ECOLI"]
QUANT_COLS = {"raw": "Precursor.Quantity", "normalised": "Precursor.Normalised"}


def build_seq2org(peptides_path: str, occurrences_path: str, proteome_path: str) -> dict:
    """Normalized (I→L) stripped sequence → organism, or ``None`` when the sequence maps to >1 organism
    (ambiguous, excluded from the primary metric — v1's "Unknown" drop; no razor, which would turn
    attribution ambiguity into apparent accuracy). Ground-truth mapping via the digest, not DiaNN's
    algorithm-dependent protein grouping."""
    pep = pq.read_table(peptides_path, columns=["peptide_id", "sequence"]).to_pandas()
    occ = pq.read_table(occurrences_path, columns=["peptide_id", "protein_id"]).to_pandas()
    prot = pq.read_table(proteome_path, columns=["protein_id", "organism"]).to_pandas()

    occ = occ.merge(prot, on="protein_id", how="left")
    pep_org = occ.groupby("peptide_id")["organism"].apply(lambda s: set(x for x in s if pd.notna(x)))
    pep = pep.merge(pep_org.rename("organisms"), on="peptide_id", how="left")
    pep["seqL"] = pep["sequence"].astype(str).apply(replace_I_with_L)

    seq2org: dict = {}
    for seqL, grp in pep.groupby("seqL"):
        orgs: set = set()
        for s in grp["organisms"]:
            if isinstance(s, set):
                orgs |= s
        seq2org[seqL] = next(iter(orgs)) if len(orgs) == 1 else None
    return seq2org


def expected_log2fc(design_path: str) -> dict:
    """Per-organism expected log2(B/A) from the design's two conditions (``mix`` fractions; ``"rest"`` =
    ``1 − Σ others``)."""
    import tomllib

    d = tomllib.load(open(design_path, "rb"))
    conds = {c["name"]: dict(c["mix"]) for c in d["condition"]}
    if len(conds) != 2:
        raise ValueError(f"quant expects exactly 2 conditions, design has {sorted(conds)}")

    def resolve(mix):
        rest = [k for k, v in mix.items() if isinstance(v, str) and v == "rest"]
        if len(rest) > 1:
            raise ValueError(f"a condition has multiple 'rest' entries: {rest}")
        known = {k: float(v) for k, v in mix.items() if not (isinstance(v, str) and v == "rest")}
        if rest:
            known[rest[0]] = 1.0 - sum(known.values())
        return known

    ref = d.get("design", {}).get("reference", "A")
    other = next(n for n in conds if n != ref)
    a, b = resolve(conds[ref]), resolve(conds[other])
    return {org: math.log2(b[org] / a[org]) for org in a if a.get(org, 0) > 0 and b.get(org, 0) > 0}


def _peptide_quant(df: pd.DataFrame, run: str, col: str, run_col: str = "Run") -> pd.Series:
    """Per-sequence quantity for one run: sum the (normalized) quantity across charge/mod forms. Zero or
    missing → absent (dropped)."""
    sub = df[df[run_col].astype(str) == str(run)]
    q = sub.groupby("seqL")[col].sum()
    return q[np.isfinite(q) & (q > 0)]  # finite positive only (drop 0/NaN/±inf/negative)


def _robust(residuals: np.ndarray, delta: float) -> dict:
    if len(residuals) == 0:
        return {"n": 0, "median_residual": None, "mad": None, "pct_correct": None}
    med = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - med)))
    return {
        "n": int(len(residuals)),
        "median_residual": med,
        "mad": mad,
        "pct_correct": float(np.mean(np.abs(residuals) <= delta)),
    }


def score_quant(report_df, seq2org, expected, run_a, run_b, qvalue=0.01, delta=0.5, run_col="Run") -> dict:
    """Core scorer. ``report_df`` is a joint DiaNN report (rows per run×precursor); ``run_a``/``run_b`` select
    conditions A (reference) and B by matching ``run_col`` (``Run`` name, or ``Run.Index`` when a joint
    search's ``.d`` folders share a name). Returns a metrics dict (JSON-serialisable)."""
    df = report_df.copy()
    if "Q.Value" in df.columns:
        df = df[df["Q.Value"] <= qvalue]
    df["seqL"] = df["Stripped.Sequence"].astype(str).apply(replace_I_with_L)
    df["org"] = df["seqL"].map(seq2org)  # None = ambiguous/unmapped

    # Eligibility: unique sequences per organism seen at all in the (q-filtered) report, + exclusions.
    seen = df.drop_duplicates("seqL")
    eligibility = {org: int((seen["org"] == org).sum()) for org in ORGANISMS}
    eligibility["ambiguous_or_unmapped"] = int(seen["org"].isna().sum())

    views: dict = {}
    detection: dict = {}
    for view, col in QUANT_COLS.items():
        qa = _peptide_quant(df, run_a, col, run_col)
        qb = _peptide_quant(df, run_b, col, run_col)
        both = qa.index.intersection(qb.index)
        fc = pd.Series(np.log2(qb[both].values / qa[both].values), index=both)
        org_of = pd.Series({s: seq2org.get(s) for s in both})

        # Detection table (only need to compute once; use the normalised pass which is the primary).
        if view == "normalised":
            for org in ORGANISMS:
                a_seqs = {s for s in qa.index if seq2org.get(s) == org}
                b_seqs = {s for s in qb.index if seq2org.get(s) == org}
                detection[org] = {
                    "a_only": len(a_seqs - b_seqs),
                    "b_only": len(b_seqs - a_seqs),
                    "both": len(a_seqs & b_seqs),
                }

        # Human-anchor recentring is derived from this view's HUMAN median (diagnostic).
        human_fc = fc[org_of == "HUMAN"]
        anchor = float(np.median(human_fc)) if len(human_fc) else 0.0

        def organism_block(shift: float) -> dict:
            out = {}
            for org in ORGANISMS:
                if org not in expected:
                    continue
                vals = fc[org_of == org].values - shift
                res = vals - expected[org]
                blk = _robust(res, delta)
                blk["median_log2fc"] = float(np.median(vals)) if len(vals) else None
                blk["expected_log2fc"] = expected[org]
                out[org] = blk
            return out

        views[view] = organism_block(0.0)
        if view == "normalised":
            views["human_anchored"] = organism_block(anchor)
            views["human_anchored_shift"] = anchor

    return {
        "expected": expected,
        "views": views,
        "detection": detection,
        "eligibility": eligibility,
        "params": {"qvalue": qvalue, "delta": delta, "run_a": run_a, "run_b": run_b},
    }


def summary_text(m: dict) -> str:
    lines = ["timsim v2 HYE quant — fold-change recovery (primary view: Precursor.Normalised)"]
    prim = m["views"]["normalised"]
    lines.append(f"  {'organism':8} {'n':>5} {'median log2FC':>14} {'expected':>9} {'residual':>9} {'MAD':>6} {'%correct':>8}")
    for org in ORGANISMS:
        b = prim.get(org)
        if not b:
            continue
        def f(x, n=3):
            return "  —  " if x is None else f"{x:.{n}f}"
        lines.append(f"  {org:8} {b['n']:>5} {f(b['median_log2fc']):>14} {f(b['expected_log2fc']):>9} "
                     f"{f(b['median_residual']):>9} {f(b['mad'],2):>6} "
                     f"{('—' if b['pct_correct'] is None else f'{b['pct_correct']*100:.0f}%'):>8}")
    d = m["detection"]
    lines.append("  detection (A-only / B-only / both): " +
                 "  ".join(f"{o} {d[o]['a_only']}/{d[o]['b_only']}/{d[o]['both']}" for o in ORGANISMS if o in d))
    e = m["eligibility"]
    lines.append(f"  eligible unique seqs: " + " ".join(f"{o}={e[o]}" for o in ORGANISMS) +
                 f"  ambiguous/unmapped={e['ambiguous_or_unmapped']}")
    lines.append(f"  human-anchor diagnostic shift: {m['views'].get('human_anchored_shift', 0.0):.3f} "
                 f"(applied only in the human_anchored view, not the score)")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="timsim-v2-quant-eval",
                                 description="HYE per-organism fold-change recovery from a JOINT DiaNN run")
    ap.add_argument("--report", required=True, type=Path, help="JOINT DiaNN report.parquet (both conditions)")
    ap.add_argument("--peptides", required=True, type=Path)
    ap.add_argument("--occurrences", required=True, type=Path)
    ap.add_argument("--proteome", required=True, type=Path)
    ap.add_argument("--design", required=True, type=Path, help="design.toml (expected ratios)")
    ap.add_argument("--run-col", default="Run.Index",
                    help="report column selecting the run (default Run.Index — robust when a joint search's "
                         ".d folders share a name; use 'Run' to select by run name)")
    ap.add_argument("--run-a", required=True, help="condition A (reference) value in --run-col (e.g. 0)")
    ap.add_argument("--run-b", required=True, help="condition B value in --run-col (e.g. 1)")
    ap.add_argument("--fdr", type=float, default=0.01)
    ap.add_argument("--delta", type=float, default=0.5, help="log2FC tolerance for %%correct")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    report_df = pq.read_table(str(a.report)).to_pandas()
    if a.run_col not in report_df.columns:
        ap.error(f"--run-col {a.run_col!r} not in report (columns include Run, Run.Index)")
    runs = set(report_df[a.run_col].astype(str).unique())
    for r in (a.run_a, a.run_b):
        if str(r) not in runs:
            ap.error(f"run {r!r} not in report {a.run_col} values: {sorted(runs)}")
    seq2org = build_seq2org(str(a.peptides), str(a.occurrences), str(a.proteome))
    expected = expected_log2fc(str(a.design))
    m = score_quant(report_df, seq2org, expected, a.run_a, a.run_b, qvalue=a.fdr, delta=a.delta,
                    run_col=a.run_col)
    print(summary_text(m))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
