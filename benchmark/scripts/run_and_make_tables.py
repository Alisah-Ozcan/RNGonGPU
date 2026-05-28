#!/usr/bin/env python3
# Copyright 2025-2026 Alişah Özcan
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import subprocess
from collections import defaultdict
from pathlib import Path


def split_args(value: str) -> list[str]:
    return [item for item in value.split() if item]


def run_command(command: list[str]) -> None:
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def detect_gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "Unknown GPU"

    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not names:
        return "Unknown GPU"
    return " / ".join(dict.fromkeys(names))


def ensure_output_dir_capable(executable: Path) -> None:
    if not executable.exists():
        raise RuntimeError(
            f"Benchmark executable not found: {executable}. "
            "Build benchmarks first with CMake."
        )

    result = subprocess.run(
        [str(executable), "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    if "--output-dir" not in result.stdout:
        raise RuntimeError(
            f"{executable} does not support --output-dir. "
            "Rebuild RNGonGPU benchmarks so the updated benchmark sources are used."
        )
    if "--data-types" not in result.stdout:
        raise RuntimeError(
            f"{executable} does not support --data-types. "
            "Rebuild RNGonGPU benchmarks so the updated benchmark sources are used."
        )


def load_rows(csv_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(csv_dir.glob("*.csv")):
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if "data_type" not in row and "type" in row:
                    row["data_type"] = row["type"]
                row.pop("type", None)
                row["source_file"] = path.name
                rows.append(row)
    return rows


def sort_key(row: dict[str, str]) -> tuple:
    return (
        row.get("backend", ""),
        row.get("variant", ""),
        row.get("distribution", ""),
        row.get("data_type", ""),
        int(row.get("size_log") or 0),
        int(row.get("security_level") or 0),
        int(row.get("stddev") or 0),
    )


def xml_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def write_svg(rows: list[dict[str, str]], output: Path, gpu_name: str) -> None:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("distribution", "unknown")].append(row)

    width = 1280
    left = 290
    right = 170
    bar_area_width = width - left - right
    row_height = 28
    panel_gap = 54
    title_height = 70
    panel_header = 44
    height = title_height
    for distribution in sorted(grouped):
        height += panel_header + row_height * len(grouped[distribution]) + panel_gap

    palette = {"aes": "#2563eb", "curand": "#16a34a"}

    with output.open("w") as handle:
        handle.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}">\n'
        )
        handle.write(
            "<style>"
            "text{font-family:monospace;font-size:12px;fill:#17202a}"
            ".title{font-size:24px;font-weight:800}"
            ".subtitle{font-size:13px;fill:#475569}"
            ".panel{font-size:18px;font-weight:700;fill:#0f172a}"
            ".axis{stroke:#94a3b8;stroke-width:1}"
            ".grid{stroke:#e2e8f0;stroke-width:1}"
            ".label{fill:#334155}"
            ".value{font-weight:700;fill:#0f172a}"
            "</style>\n"
        )
        handle.write('<rect x="0" y="0" width="100%" height="100%" fill="#ffffff"/>\n')
        handle.write('<text class="title" x="24" y="34">RNGonGPU Throughput Comparison</text>\n')
        handle.write(f'<text class="subtitle" text-anchor="end" x="{width - 24}" y="34">GPU: {xml_escape(gpu_name)}</text>\n')
        handle.write('<text class="subtitle" x="24" y="56">Each distribution compares backend + variant + data_type rows by throughput (GiB/s).</text>\n')

        y = title_height
        for distribution in sorted(grouped):
            dist_rows = sorted(
                grouped[distribution],
                key=lambda row: float(row.get("throughput_gib_s") or 0.0),
                reverse=True,
            )
            max_value = max(float(row.get("throughput_gib_s") or 0.0) for row in dist_rows)
            max_value = max(max_value, 1e-9)

            handle.write(f'<text class="panel" x="24" y="{y + 24}">{xml_escape(distribution)} distribution</text>\n')
            axis_y = y + panel_header
            handle.write(f'<line class="axis" x1="{left}" y1="{axis_y - 8}" x2="{left + bar_area_width}" y2="{axis_y - 8}"/>\n')
            for tick in range(5):
                x = left + (bar_area_width * tick / 4)
                value = max_value * tick / 4
                handle.write(f'<line class="grid" x1="{x:.1f}" y1="{axis_y - 14}" x2="{x:.1f}" y2="{axis_y + row_height * len(dist_rows)}"/>\n')
                handle.write(f'<text class="subtitle" x="{x - 18:.1f}" y="{axis_y - 18}">{value:.2f}</text>\n')

            for idx, row in enumerate(dist_rows):
                row_y = axis_y + idx * row_height
                value = float(row.get("throughput_gib_s") or 0.0)
                bar_width = bar_area_width * value / max_value
                backend = row.get("backend", "")
                color = palette.get(backend, "#64748b")
                label = (
                    f'{backend}/{row.get("variant", "")}/{row.get("data_type", "")}'
                    f' logN={row.get("size_log", "")}'
                )
                if row.get("security_level", "0") not in ("", "0"):
                    label += f' sec={row.get("security_level")}'
                handle.write(f'<text class="label" x="24" y="{row_y + 18}">{xml_escape(label)}</text>\n')
                handle.write(f'<rect x="{left}" y="{row_y + 6}" width="{bar_width:.1f}" height="16" rx="3" fill="{color}"/>\n')
                handle.write(f'<text class="value" x="{left + bar_width + 8:.1f}" y="{row_y + 19}">{value:.3f}</text>\n')

            y += panel_header + row_height * len(dist_rows) + panel_gap

        handle.write("</svg>\n")


def build_chart(csv_dir: Path, out_dir: Path, gpu_name: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    gpu_name = gpu_name or detect_gpu_name()

    rows = sorted(load_rows(csv_dir), key=sort_key)
    if not rows:
        raise RuntimeError(
            f"No CSV files found under {csv_dir}. "
            "If the benchmarks printed results but no CSV files were created, "
            "rebuild the benchmark binaries and run this script again."
        )

    write_svg(rows, out_dir / "benchmark_tables.svg", gpu_name)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run RNGonGPU benchmarks, write CSV files, and generate the SVG chart."
    )
    parser.add_argument("--build-dir", default="build")
    parser.add_argument("--csv-dir", default="benchmark/csv")
    parser.add_argument("--table-dir", default="benchmark/tables")
    parser.add_argument(
        "--gpu-name",
        default=None,
        help="GPU name written to the generated SVG chart. Defaults to nvidia-smi detection.",
    )
    parser.add_argument("--sizes", default="24")
    parser.add_argument("--warmup", default="3")
    parser.add_argument("--iterations", default="10")
    parser.add_argument("--distributions", default="uniform,normal,ternary")
    parser.add_argument(
        "--data-types",
        dest="data_types",
        default="u32,u64,f32,f64",
    )
    parser.add_argument("--types", dest="data_types", help=argparse.SUPPRESS)
    parser.add_argument(
        "--aes-extra",
        default="--security-levels 128,192,256 --stddevs 3",
    )
    parser.add_argument(
        "--cuda-extra",
        default="--curand-states xorwow,mrg32k3a,philox --stddevs 3",
    )
    parser.add_argument("--skip-aes", action="store_true")
    parser.add_argument("--skip-cuda", action="store_true")
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Only build the SVG chart from existing CSV files.",
    )

    args = parser.parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    build_dir = (repo_root / args.build_dir).resolve()
    csv_dir = (repo_root / args.csv_dir).resolve()
    table_dir = (repo_root / args.table_dir).resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    common = [
        "--sizes",
        args.sizes,
        "--warmup",
        args.warmup,
        "--iterations",
        args.iterations,
        "--distributions",
        args.distributions,
        "--data-types",
        args.data_types,
        "--output-dir",
        str(csv_dir),
    ]

    if not args.skip_run:
        if not args.skip_aes:
            ensure_output_dir_capable(build_dir / "bin/benchmark/aes_benchmark")
        if not args.skip_cuda:
            ensure_output_dir_capable(build_dir / "bin/benchmark/cuda_benchmark")

        if not args.skip_aes:
            run_command(
                [
                    str(build_dir / "bin/benchmark/aes_benchmark"),
                    *common,
                    *split_args(args.aes_extra),
                ]
            )

        if not args.skip_cuda:
            run_command(
                [
                    str(build_dir / "bin/benchmark/cuda_benchmark"),
                    *common,
                    *split_args(args.cuda_extra),
                ]
            )

    build_chart(csv_dir, table_dir, args.gpu_name)
    print(f"CSV results: {csv_dir}")
    print(f"SVG chart: {table_dir / 'benchmark_tables.svg'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
