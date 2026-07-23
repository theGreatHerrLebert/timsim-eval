# timsim-eval

A **benchmark harness built on a known answer key.** [timsim](https://github.com/theGreatHerrLebert/timsim)
renders a synthetic proteomics run whose every precursor is known ground truth; `timsim-eval` takes a
search engine's output over that run (DiaNN / Sage / FragPipe), scores it against the truth, and reports
hierarchical recall, false-discovery proportion, recall-by-abundance, RT / ion-mobility / intensity
agreement — as metrics, plots, and a self-contained HTML report.

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
[`timsim-necro`](https://github.com/theGreatHerrLebert/timsim-necro) DAG's score node and its
`golden/` regression gate. The core (DiaNN-based) path is **imspy-free**.

## What it does

- **Parse** any engine's output into a common schema (`parsing.py`, `sage_parsing.py`, `diann_*`) — the
  adapter layer that lets one scorer compare tools that report results differently.
- **Compare** IDs vs. ground truth — hierarchical precursor/peptide/protein recall, FDR calibration,
  RT / ion-mobility / intensity agreement (`comparison.py`, `metrics.py`).
- **Compare tools** — `tool_comparison.py` puts multiple engines' scores side by side on one dataset.
- **Report** — plots + a self-contained HTML report (`report.py`, `plots.py`).
- **Drive** — `v2_thermo_eval` (score one report vs one truth) is the entry point the DAG calls;
  `runner.py` orchestrates multi-run / multi-tool sweeps.

```bash
python -m timsim_eval.v2_thermo_eval --help
```

## Install

```bash
pip install "timsim-eval @ git+https://github.com/theGreatHerrLebert/timsim-eval"
```

Pure-Python deps only: pandas, numpy, scipy, matplotlib, pyarrow. **No imspy, no torch, no Rust.**

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
