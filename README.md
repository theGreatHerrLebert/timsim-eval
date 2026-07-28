# timsim-eval

A **benchmark harness built on a known answer key.** [timsim](https://github.com/theGreatHerrLebert/timsim)
renders a synthetic proteomics run whose every precursor is known ground truth; `timsim-eval` takes a
search engine's output over that run (DiaNN / Sage / FragPipe), scores it against the truth, and reports
hierarchical recall, false-discovery proportion, recall-by-abundance, RT / ion-mobility / intensity
agreement, HYE per-organism fold-change recovery, and phospho site-localization FLR — as metrics, plots,
and a self-contained HTML report.

Because the truth is a **fixed oracle**, the same measurement answers three different questions depending
on what you hold still and what you vary:

| you're asking | fix | vary | the number tells you |
|---|---|---|---|
| **Is the simulation realistic?** | the search tool | the render / predictors | did the simulator get more real |
| **Which tool is better?** | the rendered dataset | the engine (DiaNN vs Sage vs FragPipe) | a head-to-head on identical ground truth |
| **Did my software regress?** | the rendered dataset | *your* tool's version / config | whether a change helped or broke recall/FDP |

The first is simulator development; the second and third are why the harness ships `diann_executor`,
`sage_executor`, `fragpipe_executor`, and `tool_comparison` — a synthetic run with a real answer key is a
benchmark and a regression test for **any** DIA tool, not only for the simulator that made it.

Lifted out of the `imspy-simulation` monorepo package so consumers ingest only what they need — e.g. the
[`timsim-necro`](https://github.com/theGreatHerrLebert/timsim-necro) DAG's score nodes and its
`golden/` regression gate. The core (DiaNN-based) path is **imspy-free**.

## What it does

- **Parse** any engine's output into a common schema — `parsing.py` holds *all* of it: DiaNN
  (`parse_diann_report`, `format_diann_sequence`), Sage (`parse_sage_results`, `format_sage_sequence`)
  and FragPipe (`parse_fragpipe_psm`, `parse_fragpipe_combined`). It is the adapter layer that lets one
  scorer compare tools that report results differently. (`diann_executor.py` / `sage_executor.py` /
  `fragpipe_executor.py` *run* the engines; they do not parse.)
- **Build truth** — `v2_truth.py` joins the v2 Parquet answer keys (`precursors`, `peptides`,
  `modforms`, `peptide_rt`, `precursor_ccs`, `peptide_quantities`) into the frame the scorers consume,
  reproducing the render's RT and CCS→1/K0 conversions exactly.
- **Compare** IDs vs. ground truth — hierarchical precursor/peptide/protein recall, FDR calibration,
  RT / ion-mobility / intensity agreement (`comparison.py`, `metrics.py`).
- **Compare tools** — `tool_comparison.py` puts multiple engines' scores side by side on one dataset.
- **Report** — JSON + text summaries (`report.py`), plots (`plots.py`), and the self-contained HTML
  report (`html_report.py`, driven from `tool_comparison.py`).
- **Drive** — `runner.py` orchestrates multi-run / multi-tool sweeps; `cli.py` is the config-driven
  end-to-end validate CLI (see below).

## The scorers the DAG calls

The [`timsim-necro`](https://github.com/theGreatHerrLebert/timsim-necro) DAG invokes **four** scorer
entry points, one per evaluation axis. Each is a `python -m` module with its own `--help`:

| module | axis | headline numbers |
|---|---|---|
| `v2_thermo_eval` | ID recall / FDP | hierarchical recall over ever-stricter denominators (all → present → in-window → has-fragments → above an abundance floor), FDP, recall-vs-abundance-decile curve |
| `v2_quant_eval` | HYE fold-change recovery | per-organism **median residual** `median(log2FC − expected)` + **MAD** + **%correct** in a tolerance band, on organism-unique peptides, in three normalization views |
| `v2_flr_eval` | phospho site localization | **FLR(τ)** = wrong-site ÷ accepted and **correct-localization recall(τ)** = correct&accepted ÷ eligible, as curves over the confidence threshold τ plus one operating point |
| `v2_dda_eval` | DDA-PASEF (Sage) | conditional ID recall (correct ÷ selected) and FDP against the render's per-selection-event answer key |

`v2_eval.py` is the original Bruker/DiaNN v2 scorer (backbone-level matching, RT/IM/quant correlation,
pass/fail thresholds); `v2_thermo_eval` is its Thermo-and-hierarchy successor.

```bash
python -m timsim_eval.v2_thermo_eval --help
python -m timsim_eval.v2_quant_eval  --help
python -m timsim_eval.v2_flr_eval    --help
python -m timsim_eval.v2_dda_eval    --help
```

### `v2_quant_eval`'s three normalization views

`raw` (DiaNN `Precursor.Quantity`), `normalised` (DiaNN `Precursor.Normalised` — **the primary
engine-performance result**), and `human_anchored` (normalised, recentred so median HUMAN log2FC = 0 —
a *diagnostic* for global scaling only, never the score, since anchoring on human forces its residual
to zero and launders a global error into yeast/E. coli).

### `--background-report`: what the FDP number means on a noised run

`v2_thermo_eval` and `v2_eval` accept `--background-report FILE`. On a run rendered with **A2
real-data noise** (background peaks sampled from a real reference `.d`), some of the engine's IDs are
*real* peptides carried in from that blank — they are not simulator false positives, but a naive FDP
counts them as such.

Pass the DiaNN report of a **noise-only control** (`timsim-render --noise-only`, A2 on, synthetic
signal off, same seed) and its IDs are subtracted before FDP is computed. This **changes the meaning of
the headline FDP**: without it, FDP on a noised run is an upper bound inflated by the reference blank's
own peptides; with it, FDP measures what the synthetic signal actually caused. Report which one you
quoted.

## Install

```bash
pip install "timsim-eval @ git+https://github.com/theGreatHerrLebert/timsim-eval"
```

Pure-Python deps only: pandas, numpy, scipy, matplotlib, pyarrow, toml. **No imspy, no torch, no Rust.**

Installing also puts a `timsim-eval` console script on your PATH — the config-driven end-to-end
validate CLI (`cli.py`: TOML config + CLI overrides → simulate/reuse a run, execute DiaNN or FragPipe,
score, plot, report). Without installing, invoke it the same way as the scorers:

```bash
timsim-eval --help              # console script
python -m timsim_eval.cli --help   # equivalent, no install needed
```

## Tests

```bash
pip install -e ".[dev]"     # adds pytest
pytest tests -q             # or: PYTHONPATH=src python -m pytest tests -q
```

`tests/test_quant_eval.py` drives `score_quant` on a synthetic joint HYE report (known ratios in →
near-zero per-organism residuals out); `tests/test_flr_eval.py` covers the `localized_sites` UniMod
parser and `score_flr`'s isolated / dominant-isomer metrics.

## imspy is optional (three legacy-v1 modes only)

The v2 DiaNN/Thermo path needs nothing from imspy. Three optional modes still reach into the legacy v1
stack, and each raises a clear message telling you what to `pip install` if you invoke it:

| mode | needs | why |
|---|---|---|
| read a v1 `synthetic_data.db` (`comparison.py`) | `imspy-simulation` | v1 Bruker synthetic-experiment DB reader |
| simulate-from-eval (`runner.py`) | `imspy-simulation` | runs the v1 simulator inline; timsim-necro renders separately |
| Bruker `.d` peak-distribution stats (`peak_distribution.py`) | `imspy-core` | reads a `.d` via the v1 `TimsDatasetDIA` |

The single hard imspy dependency the core path had — `remove_unimod_annotation` — is vendored inline
(a two-line regex) in `parsing.py`.
