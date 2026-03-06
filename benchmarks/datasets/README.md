# Dataset Pointers and Real LP Integration

This repository includes generator-based workloads and real LP suite integration:

- LP: Netlib LP (MPS/SIF) via [Netlib repository](https://netlib.org/lp/data/)
- LP: STOCHLP via [Mészáros STOCHLP index](http://old.sztaki.hu/~meszaros/public_ftp/lptestset/stochlp/)
- LP: MISC via [Mészáros MISC index](http://old.sztaki.hu/~meszaros/public_ftp/lptestset/misc/)
- QP: Maros-Meszaros QP test set
- Conic/QP-style workloads: portfolio optimization and MPC-like dense classes generated in `benchmarks/generators`

## Real LP workflow

1. Download and index raw files:

```bash
bash scripts/fetch_real_lp.sh
```

2. Run real-LP benchmark runner (single invocation):

```bash
python benchmarks/reproduce_real_lp.py \
  --manifest benchmarks/datasets/real_lp_manifest.csv \
  --output results/real_lp_manual_$(date +%Y%m%d_%H%M%S)
```

3. Run 10-hour multi-case sweep:

```bash
bash scripts/reproduce_real_lp_10h.sh
```

Artifacts include JSONL records (including failed/timeout rows), markdown/LaTeX summaries, and PDF plots.
