"""Faithful repro via the harness LFInferenceBackend (LF async vLLM path)."""
import os, sys, json, traceback

os.environ.setdefault("HF_HOME", "/mnt2/xuran_hdd/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, "/mnt/hdd/xuran/vlm_safety_harness")
from harness.inference.lf_backend import LFInferenceBackend, EngineCrashedError
from harness.gpu.allocator import InferPlan

ROOT = "/mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    conc = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    util = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9
    gpu = int(sys.argv[4]) if len(sys.argv) > 4 else 4  # ABSOLUTE physical id (load() sets CVD)

    test = json.load(open(f"{ROOT}/test.json"))
    records = [{
        "id": r["id"], "question": r["question"],
        "image_path1": f"{ROOT}/{r['image_path1']}",
        "image_path2": f"{ROOT}/{r['image_path2']}",
        "category": r.get("category", ""), "benchmark": "our_test",
    } for r in test[:n]]

    print(f"[lf] n={n} conc={conc} ids={records[0]['id']}..{records[-1]['id']}", flush=True)

    be = LFInferenceBackend(
        model_path="moonshotai/Kimi-VL-A3B-Instruct",
        template="kimi_vl",
        infer_plan=InferPlan(gpu_ids=[gpu], tensor_parallel_size=1, gpu_memory_utilization=util),
        max_new_tokens=1024,
        temperature=0.0,
        max_model_len=8192,
        concurrency=conc,
        trust_remote_code=True,
        image_min_pixels=None,
        image_max_pixels=None,
        infer_backend="vllm",
    )
    be.load()
    print("[lf] backend loaded; running PER-BATCH (mirrors _run_benchmark) ...", flush=True)
    bs = 32
    for i in range(0, len(records), bs):
        batch = records[i:i + bs]
        print(f"[lf] --- batch {i//bs} (ids {batch[0]['id']}..{batch[-1]['id']}) calling generate_batch ---", flush=True)
        try:
            outs = be.generate_batch(batch)
            nerr = sum("[INFERENCE_ERROR]" in str(o.get("response", "")) for o in outs)
            print(f"[lf] batch {i//bs} OK -> {len(outs)} outputs, {nerr} inference-errors", flush=True)
        except EngineCrashedError:
            print(f"[lf] batch {i//bs} ENGINE CRASH (EngineCrashedError):", flush=True)
            traceback.print_exc()
            break
        except Exception:
            print(f"[lf] batch {i//bs} OTHER EXCEPTION:", flush=True)
            traceback.print_exc()
            break

    print("[lf] done", flush=True)


if __name__ == "__main__":
    main()
