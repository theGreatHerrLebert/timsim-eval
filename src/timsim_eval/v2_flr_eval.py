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
            if s[i + 1 : j] == tag:
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
    mf["phos_sites"] = mf.apply(
        lambda r: frozenset(int(p) + 1 for p, n in zip(r["mod_positions"], r["mod_names"]) if n == PHOSPHO),
        axis=1,
    )
    mf["n_sty"] = mf["sequence"].astype(str).apply(lambda s: sum(1 for c in s if c in STY))

    j = truth.merge(prec, on="precursor_id", how="left").merge(
        mf[["modform_id", "seqL", "phos_sites", "n_sty"]], on="modform_id", how="left"
    )
    j = j[j["phos_sites"].apply(lambda s: isinstance(s, frozenset) and len(s) > 0)]  # phospho precursors only

    out: dict = {}
    for (seqL, charge), grp in j.groupby(["seqL", "charge"]):
        present = list({s for s in grp["phos_sites"]})  # distinct isomers present
        out[(seqL, int(charge))] = {"sites": present, "n_sty": int(grp["n_sty"].iloc[0])}
    return out


def score_flr(report_df, truth_iso, taus=None, target_flr=0.01, qvalue=0.01) -> dict:
    """FLR(τ) curve over the isolated single-isomer, ≥2-candidate-site eligible set."""
    taus = taus if taus is not None else [round(x, 2) for x in np.arange(0.0, 1.001, 0.05)]
    df = report_df.copy()
    if "Q.Value" in df.columns:
        df = df[df["Q.Value"] <= qvalue]
    df["seqL"] = df["Stripped.Sequence"].astype(str).apply(replace_I_with_L)
    df["loc_sites"] = df["Modified.Sequence"].astype(str).apply(localized_sites)
    df["nphos"] = df["loc_sites"].apply(len)
    conf_col = "PTM.Site.Confidence"
    charge_col = "Precursor.Charge"

    # Eligible truth: single present isomer, single phospho, ≥2 candidate S/T/Y sites.
    eligible = {
        k: v["sites"][0]
        for k, v in truth_iso.items()
        if len(v["sites"]) == 1 and len(v["sites"][0]) == 1 and v["n_sty"] >= 2
    }
    n_multi_isomer = sum(1 for v in truth_iso.values() if len(v["sites"]) >= 2)

    # One DiaNN phospho call per (seqL, charge): the most confident single-phospho row.
    ph = df[df["nphos"] == 1].copy()
    ph["key"] = list(zip(ph["seqL"], ph[charge_col].astype(int)))
    ph = ph.sort_values(conf_col, ascending=False, kind="mergesort").drop_duplicates("key")
    calls = {k: (r_loc, r_conf) for k, r_loc, r_conf in zip(ph["key"], ph["loc_sites"], ph[conf_col])}

    curve = []
    for tau in taus:
        accepted = correct = 0
        for key, true_site in eligible.items():
            call = calls.get(key)
            if call is None:
                continue
            loc, conf = call
            if conf is None or conf < tau:
                continue
            accepted += 1
            if loc == true_site:
                correct += 1
        wrong = accepted - correct
        curve.append({
            "tau": tau,
            "n_accepted": accepted,
            "flr": (wrong / accepted) if accepted else None,
            "recall": (correct / len(eligible)) if eligible else None,
        })

    # Operating point: the smallest τ whose FLR ≤ target (max recall at that FLR).
    op = None
    for row in curve:
        if row["flr"] is not None and row["flr"] <= target_flr:
            op = row
            break

    return {
        "eligible_single_isomer": len(eligible),
        "n_multi_isomer_coeluting": n_multi_isomer,  # SECONDARY task coverage (not scored here)
        "flr_curve": curve,
        "operating_point": op,
        "target_flr": target_flr,
        "params": {"qvalue": qvalue},
    }


def summary_text(m: dict) -> str:
    lines = ["timsim v2 phospho FLR — site localization (PRIMARY: isolated single-isomer)"]
    lines.append(f"  eligible peptidoforms (single isomer, >=2 S/T/Y sites): {m['eligible_single_isomer']:,}")
    lines.append(f"  co-eluting multi-isomer (secondary task, not in FLR): {m['n_multi_isomer_coeluting']:,}")
    op = m["operating_point"]
    if op:
        lines.append(f"  operating point (FLR<={m['target_flr']*100:.0f}%): tau={op['tau']:.2f}  "
                     f"accepted={op['n_accepted']:,}  FLR={op['flr']*100:.2f}%  recall={op['recall']*100:.1f}%")
    else:
        lines.append(f"  no tau reaches FLR<={m['target_flr']*100:.0f}%")
    lines.append("  FLR / recall vs tau:")
    for r in m["flr_curve"]:
        if r["tau"] in (0.0, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0) and r["flr"] is not None:
            lines.append(f"    tau={r['tau']:.2f}  accepted={r['n_accepted']:>5}  "
                         f"FLR={r['flr']*100:5.2f}%  recall={r['recall']*100:5.1f}%")
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
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    report_df = pq.read_table(str(a.report)).to_pandas()
    truth_iso = build_truth_isoforms(str(a.truth), str(a.precursors), str(a.modforms), str(a.peptides))
    m = score_flr(report_df, truth_iso, target_flr=a.target_flr, qvalue=a.fdr)
    print(summary_text(m))
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(m, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
