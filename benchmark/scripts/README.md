# Benchmark Scripts

Run both benchmark binaries, write CSV files, and generate the SVG chart with one command:

```bash
python3 benchmark/scripts/run_and_make_tables.py \
  --build-dir build \
  --csv-dir benchmark/csv \
  --table-dir benchmark/tables \
  --sizes 24 \
  --warmup 5 \
  --iterations 30
```

The script writes the measured GPU name into the generated SVG chart.
By default this is detected with `nvidia-smi`; override it with `--gpu-name` if needed.

If you only want to create the SVG chart from existing CSV files:

```bash
python3 benchmark/scripts/run_and_make_tables.py \
  --csv-dir benchmark/csv \
  --table-dir benchmark/tables \
  --skip-run
```

Outputs:

- `benchmark/csv/aes_benchmark.csv`
- `benchmark/csv/cuda_benchmark.csv`
- `benchmark/tables/benchmark_tables.svg`
