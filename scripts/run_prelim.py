#!/usr/bin/env python3
"""
Batch runner for A experiments (prelim group).
No training — inference + evaluation only.

Usage:
    # Run A3 with all 3 base VLMs (default)
    python scripts/run_prelim.py --experiment A3

    # Run A1 for specific models
    python scripts/run_prelim.py --experiment A1 \
        --models OpenGVLab/InternVL3_5-8B Qwen/Qwen3.5-9B

    # Build probes first (A1 black frames, A2 relation annotation placeholders)
    python scripts/run_prelim.py --build-probes

    # MIRage row (once checkpoint path is known)
    python scripts/run_prelim.py --experiment A3 \
        --models /path/to/mirage/checkpoint --arch internvl --size-b 8.0
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from harness.utils.logger import init_session

MIS_TEST_ROOT = Path("/mnt/hdd/xuran/MIS/mis_test")

A_EXPERIMENT_CONFIGS = {
    "A1": "prelim/A1_textual_shortcut.yaml",
    "A2": "prelim/A2_pattern_coverage.yaml",
    "A3": "prelim/A3_synthetic_real_gap.yaml",
    "A4": "prelim/A4_counterfactual.yaml",
}

# Known model HF IDs → (short_name, architecture, size_b, trust_remote_code)
MODEL_INFO: dict[str, tuple] = {
    # Base VLMs (no safety SFT)
    "OpenGVLab/InternVL2_5-8B":                      ("internvl2_5_8b",       "internvl",  8.0, True),
    "OpenGVLab/InternVL3_5-8B":                      ("internvl3_5_8b",       "internvl",  8.5, True),
    "Qwen/Qwen3.5-9B":                               ("qwen3_5_9b",           "qwen2vl",   9.0, False),
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct":     ("llava_ov_8b",          "llava",     8.0, False),
    # MIRage safety-SFT models (InternVL2.5-8B + MIS fine-tune)
    "Tuwhy/InternVL2.5-8B-MIRage":                  ("internvl2_5_8b_mirage","internvl",  8.0, True),
    "Tuwhy/Qwen2-VL-7B-MIRage":                     ("qwen2_vl_7b_mirage",   "qwen2vl",   7.0, False),
}

DEFAULT_MODELS = [
    "OpenGVLab/InternVL3_5-8B",
    "Qwen/Qwen3.5-9B",
    "lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default="all", choices=["A1", "A2", "A3", "A4", "all"])
    p.add_argument(
        "--models", nargs="+", default=None, metavar="HF_ID",
        help=(
            "Model HF IDs or local checkpoint paths to evaluate. "
            "Known IDs auto-resolve to (arch, size). "
            "For unknown paths, also supply --arch and --size-b."
        ),
    )
    p.add_argument("--arch", default=None,
                   help="Architecture override for unknown model paths (internvl/qwen2vl/llava/phi)")
    p.add_argument("--size-b", type=float, default=None,
                   help="Model size in billions for unknown model paths")
    p.add_argument("--limit", type=int, default=None, help="Limit samples per benchmark")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--build-probes", action="store_true",
                   help="Build probe datasets only (A1 black frames, A2 placeholders)")
    p.add_argument("--build-probes-experiment", default=None, choices=["A1", "A2"],
                   help="Build probes for specific experiment only")
    p.add_argument("--data", default=None,
                   help="Override MIS input JSON when building probes (A1/A2)")
    p.add_argument("--output", default=None,
                   help="Override probe output path when building probes")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def build_probes(
    output_root: Path,
    experiment: str | None = None,
    data_override: str | None = None,
    output_override: str | None = None,
) -> None:
    from harness.data.probe_builder import ProbeBuilder

    output_root.mkdir(parents=True, exist_ok=True)
    data_path = Path(data_override) if data_override else MIS_TEST_ROOT / "mis_easy.json"

    if experiment in (None, "A1"):
        print("[probe] Building A1 text-only probes (easy + hard)...")
        probe_output = Path(output_override) if output_override else output_root / "probe_text_only.json"
        ProbeBuilder.build_text_only_probe(
            mis_test_json=data_path,
            output_path=probe_output,
        )
        print(f"  → {probe_output}")
        if not output_override:
            ProbeBuilder.build_text_only_probe(
                mis_test_json=MIS_TEST_ROOT / "mis_hard.json",
                output_path=output_root / "probe_text_only_hard.json",
            )
            print(f"  → {output_root}/probe_text_only_hard.json")

    if experiment in (None, "A2"):
        print("[probe] Building A2 relation-type placeholders...")
        ProbeBuilder.annotate_relation_types(
            mis_hard_json=Path(data_override) if data_override else MIS_TEST_ROOT / "mis_hard.json",
            output_path=Path(output_override) if output_override else output_root / "mis_hard_relation_annotated.json",
        )
        print(f"  → {Path(output_override) if output_override else output_root / 'mis_hard_relation_annotated.json'}")
        print("  NOTE: run GPT-4o to fill 'relation_type' fields.")

        extra_probe = output_root / "extra_relation_probe.jsonl"
        ProbeBuilder.build_relation_type_probe(
            mis_hard_json=Path(data_override) if data_override else MIS_TEST_ROOT / "mis_hard.json",
            extra_probe_jsonl=extra_probe if extra_probe.exists() else None,
            output_path=Path(output_override) if output_override else output_root / "relation_type_probe.json",
        )
        print(f"  → {Path(output_override) if output_override else output_root / 'relation_type_probe.json'}")


def _model_overrides(hf_id: str, arch: str | None, size_b: float | None) -> list[str]:
    """Build --override list for run_experiment.py from model HF ID."""
    if hf_id in MODEL_INFO:
        name, architecture, sz, trc = MODEL_INFO[hf_id]
    else:
        # Unknown path — require explicit --arch / --size-b
        if not arch:
            print(f"[warn] Model '{hf_id}' not in MODEL_INFO. "
                  "Provide --arch and --size-b, or add entry to MODEL_INFO.")
            architecture = arch or "internvl"
        else:
            architecture = arch
        sz = size_b or 8.0
        trc = architecture in ("internvl", "phi", "minicpm")
        name = Path(hf_id).name.lower().replace("-", "_").replace(".", "_")

    return [
        f"model.hf_path={hf_id}",
        f"model.name={name}",
        f"model.architecture={architecture}",
        f"model.size_b={sz}",
        f"model.trust_remote_code={str(trc).lower()}",
    ]


def run_a_experiment(exp_id: str, args: argparse.Namespace) -> None:
    config = A_EXPERIMENT_CONFIGS[exp_id]
    models = args.models or DEFAULT_MODELS

    for hf_id in models:
        overrides = _model_overrides(hf_id, args.arch, args.size_b)

        cmd = [
            "python", "scripts/run_experiment.py", config,
            "--skip-train",
            "--override", *overrides,
        ]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.skip_eval:
            cmd += ["--skip-eval"]
        if args.dry_run:
            cmd += ["--dry-run"]

        print(f"\n{'='*60}")
        print(f"[{exp_id}] model={hf_id}")
        print(f"[{exp_id}] config={config}")
        print(f"{'='*60}")

        if args.dry_run:
            print(f"[dry-run] Would run: {' '.join(cmd)}")
        else:
            subprocess.run(cmd, check=True, cwd=str(Path(__file__).parent.parent))


def main() -> None:
    args = parse_args()
    init_session("run_prelim", tag=args.experiment, category="prelim")

    harness_root = Path(__file__).parent.parent
    probes_dir = harness_root / "results" / "prelim" / "probes"

    if args.build_probes:
        build_probes(probes_dir, args.build_probes_experiment, args.data, args.output)
        return

    exps = list(A_EXPERIMENT_CONFIGS.keys()) if args.experiment == "all" else [args.experiment]

    for exp_id in exps:
        run_a_experiment(exp_id, args)

    print("\n[done] All requested A experiments complete.")


if __name__ == "__main__":
    main()
