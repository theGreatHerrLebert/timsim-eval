"""Phospho FLR scorer — parser + abundance-aware isolated / dominant metrics."""
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
    assert localized_sites("(UniMod:21)PEPS(UniMod:21)T") == frozenset({4})  # n-term tag ignored


def _report(keys_sites_conf):
    """keys_sites_conf: list of (seqL, charge, localized_site, confidence)."""
    rows = []
    for seqL, z, site, conf in keys_sites_conf:
        # Modified.Sequence with UniMod:21 after residue `site`
        pad = "A" * (max(site, 1) + 2)
        rows.append({"Run": "d", "Stripped.Sequence": seqL, "Precursor.Charge": z,
                     "Modified.Sequence": pad[:site] + "(UniMod:21)" + pad[site:],
                     "PTM.Site.Confidence": conf, "Q.Value": 0.001})
    return pd.DataFrame(rows)


def test_isolated_flr_curve():
    # 100 peptidoforms, one DOMINANT isomer (site 3, abundance 100) + a trace isomer (site 7, 1.0 < 10% -> not
    # present) -> isolated. DiaNN gets 85 right (site 3, high conf), 15 wrong (site 7, low conf).
    truth, calls = {}, []
    rng = np.random.default_rng(0)
    for i in range(100):
        k = (f"PEP{i}", 2)
        truth[k] = {"isomers": {frozenset({3}): 100.0, frozenset({7}): 1.0}, "n_sty": 2}
        wrong = i >= 85
        calls.append((f"PEP{i}", 2, 7 if wrong else 3, rng.uniform(0.5, 0.8) if wrong else rng.uniform(0.9, 1.0)))
    m = score_flr(_report(calls), truth, taus=[0.0, 0.9], present_min_frac=0.1)
    iso = m["isolated"]
    assert iso["eligible"] == 100 and m["dominant"]["eligible"] == 0
    c = {r["tau"]: r for r in iso["flr_curve"]}
    assert c[0.0]["n_accepted"] == 100 and abs(c[0.0]["flr"] - 0.15) < 1e-9
    assert c[0.9]["n_accepted"] == 85 and c[0.9]["flr"] == 0.0  # low-conf wrong calls excluded


def test_dominant_and_ambiguous_split():
    # co-eluting pairs: 60 clear-dominance (site3=100, site7=20 -> ratio 5), 40 ambiguous (100 vs 100).
    truth, calls = {}, []
    for i in range(60):
        k = (f"CLR{i}", 2)
        truth[k] = {"isomers": {frozenset({3}): 100.0, frozenset({7}): 20.0}, "n_sty": 2}
        calls.append((f"CLR{i}", 2, 3, 0.95))  # localize to the dominant (site 3) -> correct
    for i in range(40):
        k = (f"AMB{i}", 2)
        truth[k] = {"isomers": {frozenset({3}): 100.0, frozenset({7}): 100.0}, "n_sty": 2}
        calls.append((f"AMB{i}", 2, 7, 0.95))  # localize to site 7; dominant is 3 (tie->max key) -> counts wrong
    m = score_flr(_report(calls), truth, taus=[0.0], present_min_frac=0.1, dominance_min=2.0)
    d = m["dominant"]
    assert d["eligible"] == 100
    assert d["clear_dominance"]["eligible"] == 60 and d["ambiguous"]["eligible"] == 40
    # clear-dominance calls all hit the dominant site -> FLR 0
    assert d["clear_dominance"]["flr_curve"][0]["flr"] == 0.0


def test_single_site_peptide_not_eligible():
    truth = {("ONESITE", 2): {"isomers": {frozenset({3}): 100.0}, "n_sty": 1}}  # only 1 candidate STY
    m = score_flr(_report([("ONESITE", 2, 3, 0.99)]), truth, taus=[0.0])
    assert m["isolated"]["eligible"] == 0 and m["dominant"]["eligible"] == 0
