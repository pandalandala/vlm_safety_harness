#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/hdd/xuran/vlm_safety_harness
SMOKE_JSON="$ROOT/tmp/our_test_smoke_minimal.json"
SMOKE_IMG_ROOT=/mnt/hdd/xuran/vlm_safety_harness/data_links/mis_test

latest_run_dir() {
  local run_name="$1"
  find "$ROOT/results/main/$run_name" -mindepth 1 -maxdepth 1 -type d | sort | tail -n 1
}

latest_checkpoint_dir() {
  local run_name="$1"
  local result_root
  result_root="$ROOT/results/main/$run_name"
  if [[ ! -d "$result_root" ]]; then
    echo "No run directory found for $run_name" >&2
    return 1
  fi

  local ckpt_dir
  ckpt_dir="$(
    find "$result_root" -mindepth 3 -maxdepth 3 -type d -path '*/checkpoint/checkpoint-*' \
      | sort \
      | tail -n 1
  )"
  if [[ -z "$ckpt_dir" ]]; then
    echo "No checkpoint directory found under $result_root" >&2
    return 1
  fi

  printf '%s\n' "$ckpt_dir"
}

run_dreams_retrain() {
  echo "[run] retrain main_dreams_llava_ov from hfcompat base"
  conda run -n mis_safety_llava bash -lc "
    cd '$ROOT' &&
    python scripts/run_experiment.py main/main_dreams_llava_ov.yaml \
      --skip-eval
  "
}

run_dreams_smoke() {
  local ckpt_dir
  ckpt_dir="$(latest_checkpoint_dir main_dreams_llava_ov)"
  echo "[run] smoke main_dreams_llava_ov checkpoint in mis_safety_llava453"
  conda run -n mis_safety_llava453 bash -lc "
    cd '$ROOT' &&
    python scripts/run_experiment.py main/main_dreams_llava_ov.yaml \
      --experiment-id E1_llava_dreams_retrain_smoke \
      --skip-train --skip-eval \
      --model-path '$ckpt_dir' \
      --limit 2 \
      --override \
        dataset.test_path='$SMOKE_JSON' \
        dataset.image_root='$SMOKE_IMG_ROOT' \
        dataset.test_image_root='$SMOKE_IMG_ROOT'
  "
}

run_mirage_retrain() {
  echo "[run] retrain main_baseline_mirage_data_llava_ov from hfcompat base"
  conda run -n mis_safety_llava bash -lc "
    cd '$ROOT' &&
    python scripts/run_experiment.py main/main_baseline_mirage_data_llava_ov.yaml \
      --skip-eval
  "
}

run_mirage_smoke() {
  local ckpt_dir
  ckpt_dir="$(latest_checkpoint_dir main_baseline_mirage_data_llava_ov)"
  echo "[run] smoke main_baseline_mirage_data_llava_ov checkpoint in mis_safety_llava453"
  conda run -n mis_safety_llava453 bash -lc "
    cd '$ROOT' &&
    python scripts/run_experiment.py main/main_baseline_mirage_data_llava_ov.yaml \
      --experiment-id E1_llava_mirage_retrain_smoke \
      --skip-train --skip-eval \
      --model-path '$ckpt_dir' \
      --limit 2 \
      --override \
        dataset.test_path='$SMOKE_JSON' \
        dataset.image_root='$SMOKE_IMG_ROOT' \
        dataset.test_image_root='$SMOKE_IMG_ROOT'
  "
}

run_base_8b_smoke() {
  echo "[run] smoke official LLaVA-OneVision-1.5-8B base in mis_safety_llava453"
  conda run -n mis_safety_llava453 bash -lc "
    cd '$ROOT' &&
    python scripts/run_experiment.py main/main_dreams_llava_ov.yaml \
      --experiment-id E1_llava_8b_base_refresh \
      --skip-train --skip-eval \
      --model-path lmms-lab/LLaVA-OneVision-1.5-8B-Instruct \
      --limit 2 \
      --override \
        dataset.test_path='$SMOKE_JSON' \
        dataset.image_root='$SMOKE_IMG_ROOT' \
        dataset.test_image_root='$SMOKE_IMG_ROOT'
  "
}

run_base_4b_smoke() {
  echo "[run] smoke official LLaVA-OneVision-1.5-4B base in mis_safety_llava453"
  conda run -n mis_safety_llava453 bash -lc "
    cd '$ROOT' &&
    python scripts/run_experiment.py main/main_baseline_llava_ov_1_5_4b.yaml \
      --experiment-id E1_llava_4b_base_refresh \
      --skip-train --skip-eval \
      --limit 2 \
      --override \
        dataset.test_path='$SMOKE_JSON' \
        dataset.image_root='$SMOKE_IMG_ROOT' \
        dataset.test_image_root='$SMOKE_IMG_ROOT'
  "
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run_6_1_pending_llava.sh <target>

Targets:
  dreams-retrain
  dreams-smoke
  mirage-retrain
  mirage-smoke
  base-8b-smoke
  base-4b-smoke
EOF
}

main() {
  local target="${1:-}"
  case "$target" in
    dreams-retrain) run_dreams_retrain ;;
    dreams-smoke) run_dreams_smoke ;;
    mirage-retrain) run_mirage_retrain ;;
    mirage-smoke) run_mirage_smoke ;;
    base-8b-smoke) run_base_8b_smoke ;;
    base-4b-smoke) run_base_4b_smoke ;;
    *) usage; exit 1 ;;
  esac
}

main "$@"
