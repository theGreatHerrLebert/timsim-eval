"""Phospho FLR scorer — parser + FLR(τ) curve on synthetic truth/localization."""
import numpy as np
import pandas as pd

from timsim_eval.v2_flr_eval import localized_sites, score_flr


def test_localized_sites_parser():
    assert localized_sites("ACGARIVS(UniMod:21)RPEELR") == frozenset({8})
    assert localized_sites("ACKQS(UniMod:21)LGLTIQK") == frozenset({5})
    # ignores non-phospho tags (n-term UniMod:1, fixed UniMod:4); only UniMod:21 counts.
    # residues A(1)C(2)K(3)S(4)T(5)R(6); phospho follows T -> {5}
    assert localized_sites("(UniMod:1)AC(UniMod:4)KST(UniMod:21)R") == frozenset({5})
    assert localized_sites("PEPTIDEK") == frozenset()


def _case(n=100, n_wrong=15, seed=0):
    """n eligible single-isomer peptidoforms (2 S/T/Y sites, true phospho on site 3). DiaNN gets
    n-n_wrong right (site 3, high confidence) and n_wrong wrong (site 7, low confidence)."""
    rng = np.random.default_rng(seed)
    truth_iso, rows = {}, []
    for i in range(n):
        seqL = f"AASAAASK{i}"  # keeps keys unique; 2 S sites at positions 3 and 7 in the AASAAASK prefix
        truth_iso[(seqL, 2)] = {"sites": [frozenset({3})], "n_sty": 2}
        wrong = i >= (n - n_wrong)
        loc = 7 if wrong else 3
        conf = rng.uniform(0.5, 0.8) if wrong else rng.uniform(0.9, 1.0)
        rows.append({"Run": "d", "Stripped.Sequence": seqL, "Precursor.Charge": 2,
                     "Modified.Sequence": seqL[:loc] + "(UniMod:21)" + seqL[loc:],
                     "PTM.Site.Confidence": conf, "Q.Value": 0.001})
    return pd.DataFrame(rows), truth_iso


def test_flr_curve_and_operating_point():
    rep, truth_iso = _case(n=100, n_wrong=15)
    m = score_flr(rep, truth_iso, taus=[0.0, 0.85, 0.9, 0.95], target_flr=0.05)
    assert m["eligible_single_isomer"] == 100
    c = {r["tau"]: r for r in m["flr_curve"]}
    # tau=0: all 100 accepted, 15 wrong -> FLR 15%, recall 85 correct/100 eligible
    assert c[0.0]["n_accepted"] == 100 and abs(c[0.0]["flr"] - 0.15) < 1e-9
    assert abs(c[0.0]["recall"] - 0.85) < 1e-9
    # tau=0.9: the 15 wrong calls (conf<0.8) are excluded -> only 85 correct accepted, FLR 0
    assert c[0.9]["n_accepted"] == 85 and c[0.9]["flr"] == 0.0
    assert abs(c[0.9]["recall"] - 0.85) < 1e-9
    # operating point at FLR<=5% exists and is the lowest such tau
    assert m["operating_point"] is not None and m["operating_point"]["flr"] <= 0.05


def test_multi_isomer_excluded_from_primary():
    rep, truth_iso = _case(n=50, n_wrong=0)
    # add a co-eluting 2-isomer peptidoform: must NOT count as eligible single-isomer
    truth_iso[("COELUTES", 2)] = {"sites": [frozenset({3}), frozenset({7})], "n_sty": 2}
    m = score_flr(rep, truth_iso, taus=[0.0])
    assert m["eligible_single_isomer"] == 50
    assert m["n_multi_isomer_coeluting"] == 1


def test_single_site_peptide_not_eligible():
    # a peptide with only ONE candidate S/T/Y site -> localization trivial -> excluded
    rep, truth_iso = _case(n=10, n_wrong=0)
    truth_iso[("ONESITE", 2)] = {"sites": [frozenset({3})], "n_sty": 1}
    m = score_flr(rep, truth_iso, taus=[0.0])
    assert m["eligible_single_isomer"] == 10  # the n_sty==1 one is excluded
