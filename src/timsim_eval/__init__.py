"""timsim-eval — the timsim v2 evaluation harness: parse search output (DiaNN/Sage/FragPipe), compare to the
render's ground-truth manifest, compute metrics + plots + HTML report. Lifted out of imspy-simulation.

The DiaNN-based v2_thermo_eval path is imspy-free (one vendored regex). A few OPTIONAL modes (simulate-from-
eval, reading a Bruker .d) keep lazy imspy imports — install imspy only if you use those."""
