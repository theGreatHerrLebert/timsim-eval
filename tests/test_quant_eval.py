"""HYE quant scorer — recovers known per-organism fold-changes from a synthetic joint report."""
import numpy as np
import pandas as pd

from timsim_eval.v2_quant_eval import score_quant


def _synthetic_report(seed=0, n_per=100):
    rng = np.random.default_rng(seed)
    ratios = {"HUMAN": 1.0, "YEAST": 0.20 / 0.30, "ECOLI": 0.15 / 0.05}
    rows, seq2org = [], {}
    for org, r in ratios.items():
        for i in range(n_per):
            s = f"{org[:1]}PEP{i}K"
            seq2org[s] = org
            a = rng.lognormal(12, 1.0)
            b = a * r * rng.lognormal(0, 0.15)
            for run, q in [("A_R1", a), ("B_R1", b)]:
                rows.append({"Run": run, "Stripped.Sequence": s, "Precursor.Charge": 2,
                             "Precursor.Quantity": q, "Precursor.Normalised": q, "Q.Value": 0.001})
    return pd.DataFrame(rows), seq2org


def test_quant_recovers_known_ratios():
    rep, seq2org = _synthetic_report()
    expected = {"HUMAN": 0.0, "YEAST": np.log2(0.20 / 0.30), "ECOLI": np.log2(0.15 / 0.05)}
    m = score_quant(rep, seq2org, expected, "A_R1", "B_R1", qvalue=0.01, delta=0.5)
    prim = m["views"]["normalised"]
    for org in ["HUMAN", "YEAST", "ECOLI"]:
        b = prim[org]
        assert b["n"] == 100
        assert abs(b["median_residual"]) < 0.1, f"{org} residual {b['median_residual']}"
        assert b["pct_correct"] >= 0.9, f"{org} correct {b['pct_correct']}"
    # detection: all quantified in both conditions
    assert all(m["detection"][o]["both"] == 100 for o in ["HUMAN", "YEAST", "ECOLI"])


def test_ambiguous_and_missing_excluded():
    rep, seq2org = _synthetic_report()
    # make one sequence ambiguous (organism None) and one B-only (drop from A)
    seq2org["HPEP0K"] = None
    rep = rep[~((rep["Run"] == "A_R1") & (rep["Stripped.Sequence"] == "HPEP1K"))]
    expected = {"HUMAN": 0.0, "YEAST": np.log2(0.20 / 0.30), "ECOLI": np.log2(0.15 / 0.05)}
    m = score_quant(rep, seq2org, expected, "A_R1", "B_R1")
    assert m["eligibility"]["ambiguous_or_unmapped"] >= 1
    # HPEP1K present in B only → not complete-case → HUMAN n drops below 100 (ambiguous+missing removed)
    assert m["views"]["normalised"]["HUMAN"]["n"] <= 98
    assert m["detection"]["HUMAN"]["b_only"] >= 1


def test_human_anchor_is_diagnostic_only():
    # a global 2x scaling on B should shift raw/normalised HUMAN off 0, but human_anchored recentres it.
    rep, seq2org = _synthetic_report()
    rep.loc[rep["Run"] == "B_R1", ["Precursor.Quantity", "Precursor.Normalised"]] *= 2.0
    expected = {"HUMAN": 0.0, "YEAST": np.log2(0.20 / 0.30), "ECOLI": np.log2(0.15 / 0.05)}
    m = score_quant(rep, seq2org, expected, "A_R1", "B_R1")
    assert m["views"]["normalised"]["HUMAN"]["median_log2fc"] > 0.8      # ~ +1 (2x)
    assert abs(m["views"]["human_anchored"]["HUMAN"]["median_residual"]) < 0.05  # recentred
