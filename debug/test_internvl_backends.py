"""Compare InternVL output: vLLM vs HF backend, same records. Decide if vLLM garbles."""
import os, sys, json

os.environ.setdefault("HF_HOME", "/mnt2/xuran_hdd/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, "/mnt/hdd/xuran/vlm_safety_harness")
from harness.inference.lf_backend import LFInferenceBackend
from harness.gpu.allocator import InferPlan
from harness.training.trainer import ARCH_TO_TEMPLATE

ROOT = "/mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset"
MODEL = "/mnt/hdd/xuran/vlm_safety_harness/models/dreams_internvl3_5"


def main():
    backend = sys.argv[1]            # "vllm" or "huggingface"
    gpu = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4

    test = json.load(open(f"{ROOT}/test.json"))
    records = [{
        "id": r["id"], "question": r["question"],
        "image_path1": f"{ROOT}/{r['image_path1']}",
        "image_path2": f"{ROOT}/{r['image_path2']}",
        "category": r.get("category", ""), "benchmark": "our_test",
    } for r in test[:n]]

    be = LFInferenceBackend(
        model_path=MODEL,
        template=ARCH_TO_TEMPLATE["internvl"],
        infer_plan=InferPlan(gpu_ids=[gpu], tensor_parallel_size=1, gpu_memory_utilization=0.9),
        max_new_tokens=256,
        temperature=0.0,
        max_model_len=8192,
        concurrency=4,
        trust_remote_code=True,
        infer_backend=backend,
    )
    be.load()
    print(f"[test] backend={backend} loaded, running {n} records ...", flush=True)
    outs = be.generate_batch(records)
    print(f"\n========= RESULTS backend={backend} =========", flush=True)
    for o in outs:
        print(f"--- id={o['id']} ---", flush=True)
        print(repr(str(o.get("response", ""))[:400]), flush=True)
    print("[test] done", flush=True)


if __name__ == "__main__":
    main()
