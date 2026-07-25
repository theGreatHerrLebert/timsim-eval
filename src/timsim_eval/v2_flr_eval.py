"""Phospho site-localization / FLR eval (P1.3).

Scores whether a search engine puts the phosphate on the RIGHT residue — the false-localization rate (FLR),
empirical against simulator truth (not the engine's self-estimated FLR). See PHOSPHO_FLR.md.

PRIMARY metric — isolated single-isomer FLR(τ): restrict to eligible peptidoforms (≥2 candidate S/T/Y sites)
where exactly ONE phospho isomer is present in the run, then for calls accepted at localization confidence
≥ τ (DiaNN `PTM.Site.Confidence`), ``FLR(τ) = #wrong-site / #accepted``. Always paired with
``correct-localization recall(τ) = #correct&accepted / #eligible`` so an engine can't score FLR 0 by
declining to localize. Reported as an FLR-vs-τ / recall-vs-τ CURVE plus one operating point.

Co-eluting-isomer *component recovery* (peptides with ≥2 present isomers) is a SEPARATE mixture task, not
folded into FLR — reported here only as a coverage count; the full one-to-one component matcher is a
follow-on (PHOSPHO_FLR.md §secondary).

Localization vs identification are separate: a call whose backbone/charge isn't a true rendered phospho
precursor is an ID error (belongs to FDP), NOT a localization error — excluded from the FLR denominator.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .parsing import replace_I_with_L

PHOSPHO = "Phospho"
STY = set("STY")


def localized_sites(modified_sequence: str, tag: str = "UniMod:21") -> frozenset:
    """1-based residue positions carrying `tag` (phospho) in a DiaNN `Modified.Sequence`
    (e.g. ``ACGARIVS(UniMod:21)RPEELR`` → {8}). A ``(...)`` group modifies the residue it FOLLOWS."""
    sites, pos, i, s = set(), 0, 0, str(modified_sequence)
    while i < len(s):
        if s[i] == "(":
            j = s.index(")", i)
            if s[i + 1 : j] == tag and pos > 0:  # residue mod; ignore a spurious n-term-position tag
                sites.add(pos)  # the residue just consumed (1-based)
            i = j + 1
        else:
            pos += 1
            i += 1
    return frozenset(sites)


def build_truth_isoforms(truth_path, precursors_path, modforms_path, peptides_path) -> dict:
    """`(backbone_seqL, charge) -> {"sites": [frozenset(1-based phospho sites) per PRESENT isomer],
    "n_sty": int}`. A present isomer = a rendered phospho precursor (truth) with abundance > 0. Truth carries
    no site, so join precursor→modform→mod_positions (Phospho only). modform `mod_positions` are 0-based
    residue indices (annotate inserts after index p) → +1 to match the 1-based DiaNN parse."""
    truth = pq.read_table(truth_path, columns=["precursor_id", "charge", "abundance"]).to_pandas()
    truth = truth[truth["abundance"] > 0]
    prec = pq.read_table(precursors_path, columns=["precursor_id", "modform_id"]).to_pandas()
    mf = pq.read_table(modforms_path, columns=["modform_id", "peptide_id", "mod_positions", "mod_names"]).to_pandas()
    pep = pq.read_table(peptides_path, columns=["peptide_id", "sequence"]).to_pandas()

    mf = mf.merge(pep, on="peptide_id", how="left")
    mf["seqL"] = mf["sequence"].astype(str).apply(replace_I_with_L)

    def _sites(row):
        # 0-based render mod_positions → 1-based to match localized_sites(). FAIL-CLOSED: a phospho must
        # land on an S/T/Y residue — if it doesn't, the 0-vs-1-based convention is wrong (or the data is),
        # and every FLR would be silently corrupted. Assert it self-checks against the real modforms.
        seq = str(row["sequence"])
        out = set()
        for p, n in zip(row["mod_positions"], row["mod_names"]):
            if n != PHOSPHO:
                continue
            p = int(p)
            if not (0 <= p < len(seq) and seq[p] in STY):
                raise ValueError(
                    f"phospho position convention mismatch: modform {row['modform_id']} phospho at 0-based "
                    f"{p} of {seq!r} is not S/T/Y — check mod_positions base (expected 0-based residue index)"
                )
            out.add(p + 1)
        return frozenset(out)

    mf["phos_sites"] = mf.apply(_sites, axis=1)
    mf["n_sty"] = mf["sequence"].astype(str).apply(lambda s: sum(1 for c in s if c in STY))

    j = truth.merge(prec, on="precursor_id", how="left").merge(
        mf[["modform_id", "seqL", "phos_sites", "n_sty"]], on="modform_id", how="left"
    )
    j = j[j["phos_sites"].apply(lambda s: isinstance(s, frozenset) and len(s) > 0)]  # phospho precursors only

    out: dict = {}
    for (seqL, charge), grp in j.groupby(["seqL", "charge"]):
        # per distinct phospho isomer (site-set) → summed rendered abundance. Positional isomers of the same
        # peptidoform (e.g. P@S vs P@T) co-elute, so which one(s) are "present" for a localization call is an
        # abundance question, resolved by the threshold in score_flr.
        isomers = grp.groupby("phos_sites")["abundance"].sum().to_dict()  # frozenset(sites) -> abundance
        out[(seqL, int(charge))] = {"isomers": isomers, "n_sty": int(grp["n_sty"].iloc[0])}
    return out


def _flr_curve(eligible, calls, taus, target_flr):
    """FLR(τ) / recall(τ) over an ``{key: true_site frozenset}`` map, given ``{key: (loc, conf)}`` calls.
    FLR = wrong/accepted (accepted = conf≥τ); recall = correct/|eligible| (so declining can't fake FLR 0)."""
    curve = []
    for tau in taus:
        accepted = correct = 0
        for key, true_site in eligible.items():
            call = calls.get(key)
            if call is None:
                continue
            loc, conf = call
            if conf < tau:
                continue
            accepted += 1
            if loc == true_site:
                correct += 1
        curve.append({
            "tau": tau, "n_accepted": accepted,
            "flr": ((accepted - correct) / accepted) if accepted else None,
            "recall": (correct / len(eligible)) if eligible else None,
        })
    op = next((r for r in curve if r["flr"] is not None and r["flr"] <= target_flr), None)
    return curve, op


def score_flr(report_df, truth_iso, taus=None, target_flr=0.01, qvalue=0.01,
              present_min_frac=0.1, dominance_min=2.0) -> dict:
    """Two abundance-aware localization metrics over ≥2-candidate-site peptidoforms. A single-phospho isomer
    counts as PRESENT if its rendered abundance ≥ ``present_min_frac`` × the top isomer's (trace isomers
    don't count). Then:
    - **isolated** (exactly ONE present isomer) → the clean FLR(τ) primary vs that site;
    - **dominant** (≥2 present, co-eluting) → dominant-isomer classification: score DiaNN's call against the
      MOST-ABUNDANT present isomer, stratified by dominance ratio (top/second) — a near-equal mixture
      (ratio < ``dominance_min``) is genuinely ambiguous, so its FLR floor is the mixing, not the engine."""
    taus = taus if taus is not None else [round(x, 2) for x in np.arange(0.0, 1.001, 0.05)]
    df = report_df.copy()
    # Single-run only: the "most-confident row per (seqL,charge)" dedup below would cherry-pick the best
    # localization across runs/features and make FLR optimistic. The per-sample phospho search is one run.
    if "Run" in df.columns and df["Run"].nunique() > 1:
        raise ValueError(f"FLR expects a single-run report; got runs {sorted(df['Run'].unique())}")
    if "Q.Value" in df.columns:
        df = df[df["Q.Value"] <= qvalue]
    df["seqL"] = df["Stripped.Sequence"].astype(str).apply(replace_I_with_L)
    df["loc_sites"] = df["Modified.Sequence"].astype(str).apply(localized_sites)
    df["nphos"] = df["loc_sites"].apply(len)
    conf_col, charge_col = "PTM.Site.Confidence", "Precursor.Charge"

    # Classify each ≥2-STY peptidoform by its PRESENT single-phospho isomers (abundance-thresholded).
    isolated, dominant, dom_clear, dom_ambig = {}, {}, {}, {}
    for key, v in truth_iso.items():
        if v["n_sty"] < 2:
            continue
        singles = {s: a for s, a in v["isomers"].items() if len(s) == 1}
        if not singles:
            continue
        top = max(singles.values())
        present = {s: a for s, a in singles.items() if a >= present_min_frac * top}
        if len(present) == 1:
            isolated[key] = next(iter(present))
        elif len(present) >= 2:
            ranked = sorted(present.values(), reverse=True)
            ratio = ranked[0] / ranked[1] if ranked[1] > 0 else float("inf")
            top_site = max(present, key=present.get)
            dominant[key] = top_site
            (dom_clear if ratio >= dominance_min else dom_ambig)[key] = top_site

    # One DiaNN phospho call per (seqL, charge): the most confident single-phospho row. Drop non-finite
    # localization confidences first (a NaN would otherwise pass every `conf >= τ` test spuriously).
    ph = df[df["nphos"] == 1].copy()
    ph = ph[ph[conf_col].apply(lambda x: pd.notna(x) and np.isfinite(x))]
    ph["key"] = list(zip(ph["seqL"], ph[charge_col].astype(int)))
    ph = ph.sort_values(conf_col, ascending=False, kind="mergesort").drop_duplicates("key")
    calls = {k: (loc, conf) for k, loc, conf in zip(ph["key"], ph["loc_sites"], ph[conf_col])}

    iso_curve, iso_op = _flr_curve(isolated, calls, taus, target_flr)
    dom_curve, dom_op = _flr_curve(dominant, calls, taus, target_flr)
    clear_curve, _ = _flr_curve(dom_clear, calls, taus, target_flr)
    ambig_curve, _ = _flr_curve(dom_ambig, calls, taus, target_flr)

    return {
        "isolated": {"eligible": len(isolated), "flr_curve": iso_curve, "operating_point": iso_op},
        "dominant": {
            "eligible": len(dominant), "flr_curve": dom_curve, "operating_point": dom_op,
            "clear_dominance": {"eligible": len(dom_clear), "flr_curve": clear_curve},
            "ambiguous": {"eligible": len(dom_ambig), "flr_curve": ambig_curve},
        },
        "target_flr": target_flr,
        "params": {"qvalue": qvalue, "present_min_frac": present_min_frac, "dominance_min": dominance_min},
    }


def _curve_lines(curve, target):
    op = next((r for r in curve if r["flr"] is not None and r["flr"] <= target), None)
    out = []
    if op:
        out.append(f"    operating point (FLR<={target*100:.0f}%): tau={op['tau']:.2f}  "
                   f"accepted={op['n_accepted']:,}  FLR={op['flr']*100:.2f}%  recall={op['recall']*100:.1f}%")
    for r in curve:
        if r["tau"] in (0.0, 0.5, 0.9, 0.99) and r["flr"] is not None:
            out.append(f"    tau={r['tau']:.2f}  accepted={r['n_accepted']:>5}  "
                       f"FLR={r['flr']*100:5.2f}%  recall={r['recall']*100:5.1f}%")
    return out


def summary_text(m: dict) -> str:
    t = m["target_flr"]
    lines = ["timsim v2 phospho FLR — site localization (abundance-aware)"]
    iso = m["isolated"]
    lines.append(f"  ISOLATED single-isomer (one present isomer, >=2 S/T/Y): {iso['eligible']:,} eligible")
    lines += _curve_lines(iso["flr_curve"], t) if iso["eligible"] else ["    (none — co-elution dominates; see dominant)"]
    dom = m["dominant"]
    lines.append(f"  DOMINANT-isomer classification (co-eluting, score vs top isomer): {dom['eligible']:,} eligible")
    lines += _curve_lines(dom["flr_curve"], t)
    lines.append(f"    clear dominance (top/second >= {m['params']['dominance_min']}): "
                 f"{dom['clear_dominance']['eligible']:,}")
    lines += ["  " + x for x in _curve_lines(dom["clear_dominance"]["flr_curve"], t)]
    lines.append(f"    ambiguous (near-equal isomers, FLR floor is the mixing): {dom['ambiguous']['eligible']:,}")
    lines += ["  " + x for x in _curve_lines(dom["ambiguous"]["flr_curve"], t)]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="timsim-v2-flr-eval",
                                 description="phospho site-localization FLR from a DiaNN --monitor-mod report")
    ap.add_argument("--report", required=True, type=Path, help="DiaNN report.parquet (--monitor-mod UniMod:21)")
    ap.add_argument("--truth", required=True, type=Path, help="render answer key (precursor_id, charge, abundance)")
    ap.add_argument("--precursors", required=True, type=Path)
    ap.add_argument("--modforms", required=True, type=Path)
    ap.add_argument("--peptides", required=True, type=Path)
    ap.add_argument("--fdr", type=float, default=0.01)
    ap.add_argument("--target-flr", type=float, default=0.01)
    ap.add_argument("--present-min-frac", type=float, default=0.1,
                    help="an isomer is 'present' if its abundance >= this fraction of the top isomer's")
    ap.add_argument("--dominance-min", type=float, default=2.0,
                    help="top/second isomer abundance ratio above which a co-eluting call is 'clear dominance'")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    report_df = pq.read_table(str(a.report)).to_pandas()
    truth_iso = build_truth_isoforms(str(a.truth), str(a.precursors), str(a.modforms), str(a.peptides))
    m = score_flr(report_df, truth_iso, target_flr=a.target_flr, qvalue=a.fdr,
                  present_min_frac=a.present_min_frac, dominance_min=a.dominance_min)
    print(summary_text(m))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
