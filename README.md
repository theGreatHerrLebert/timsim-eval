# timsim-eval

The **evaluation / validation harness** for the [timsim](https://github.com/theGreatHerrLebert/timsim)
v2 simulator: it takes a search engine's output (DiaNN / Sage / FragPipe) run over a *simulated* dataset,
compares the identifications against the render's **ground-truth manifest**, and produces metrics, plots,
and an HTML report — so you can answer "how realistic is the synthetic run?" quantitatively.

Lifted out of the `imspy-simulation` monorepo package so the [`timsim-necro`](https://github.com/theGreatHerrLebert/timsim-necro)
DAG's benchmarking step ingests only what it needs. The core (DiaNN-based) eval path is **imspy-free**.

## What it does

- **Parse** search output into a common schema (`parsing.py`, `sage_parsing.py`, `diann_*`).
- **Compare** IDs vs. the simulator's ground truth — precursor/peptide/protein recall, FDR calibration,
  RT / ion-mobility / intensity agreement (`comparison.py`, `metrics.py`).
- **Report** — plots + a self-contained HTML report (`report.py`, `plots.py`).
- **Drive** — `v2_thermo_eval` (Thermo `.raw` → DiaNN → eval) is the entry point the DAG calls;
  `runner.py` orchestrates multi-run sweeps.

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
