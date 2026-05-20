"""Minimal vLLM repro for Kimi-VL EngineCore crash. Isolates vLLM layer from harness."""
import os, sys, json, traceback

os.environ.setdefault("HF_HOME", "/mnt2/xuran_hdd/.cache/huggingface")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from PIL import Image
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

MODEL = "moonshotai/Kimi-VL-A3B-Instruct"
ROOT = "/mnt/hdd/xuran/vlm_safety_harness/data_links/our_dataset"
TEST = f"{ROOT}/test.json"

# usage: repro_kimi.py single <id> [id...]   OR   repro_kimi.py batch <N>
mode = sys.argv[1] if len(sys.argv) > 1 else "single"
records_all = json.load(open(TEST))
data = {r["id"]: r for r in records_all}

print(f"[repro] loading processor + LLM (enforce_eager, maxlen 8192) ...", flush=True)
proc = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
llm = LLM(
    model=MODEL,
    trust_remote_code=True,
    max_model_len=8192,
    enforce_eager=True,
    limit_mm_per_prompt={"image": 2},
    gpu_memory_utilization=0.9,
)
sp = SamplingParams(max_tokens=1024, temperature=0.0)


def build(r):
    q = r["question"]
    paths = [f"{ROOT}/{r['image_path1']}", f"{ROOT}/{r['image_path2']}"]
    imgs = [Image.open(p).convert("RGB") for p in paths]
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "image"}, {"type": "text", "text": q}]}]
    prompt = proc.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return {"prompt": prompt, "multi_modal_data": {"image": imgs}}


if mode == "batch":
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    sel = records_all[:n]
    print(f"[repro] BATCH n={n} ids={sel[0]['id']}..{sel[-1]['id']}", flush=True)
    reqs = [build(r) for r in sel]
    try:
        outs = llm.generate(reqs, sp)
        print(f"[repro] BATCH OK, {len(outs)} outputs", flush=True)
    except Exception:
        print("[repro] BATCH CRASH:", flush=True)
        traceback.print_exc()
else:
    ids = [int(x) for x in sys.argv[2:]] or [15352]
    for rid in ids:
        r = data[rid]
        print(f"\n[repro] id={rid} q={r['question'][:60]!r}", flush=True)
        try:
            out = llm.generate(build(r), sp)
            print(f"[repro] id={rid} OK -> {out[0].outputs[0].text[:200]!r}", flush=True)
        except Exception:
            print(f"[repro] id={rid} CRASH:", flush=True)
            traceback.print_exc()
            break

print("[repro] done", flush=True)
