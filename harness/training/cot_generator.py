"""
CoTGenerator: generate structured Safety Rationale labels for DREAMS training data.

Uses a capable VLM (e.g., Qwen2-VL-72B or InternVL2.5-78B) to annotate each sample
with a 3-step CoT following MIS's safety reasoning structure.

Output CoT format (structured):
  <safety_analysis>
  Step 1 [Object Recognition]: Identify the objects/entities in each image.
  Step 2 [Relationship Analysis]: Analyze how the objects relate to the question.
  Step 3 [Harm Assessment]: Determine whether the combination constitutes a safety risk.
  Conclusion: [Safe / Unsafe] — {reason}
  </safety_analysis>
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

COT_SYSTEM_PROMPT = """You are a multimodal safety annotator. Given two images and a user question, generate a structured safety analysis following exactly this format:

<safety_analysis>
Step 1 [Object Recognition]: Describe the key objects/entities visible in Image 1 and Image 2.
Step 2 [Relationship Analysis]: Explain how the objects/entities in Image 1 and Image 2 relate to the user's question.
Step 3 [Harm Assessment]: Determine whether fulfilling this request using these specific images would constitute a safety violation. Consider: (a) the combined effect of both images, (b) counterfactual — would swapping either image make the request safe?
Conclusion: [Safe/Unsafe] — {brief reason}
</safety_analysis>

Be concise. Focus on safety-relevant reasoning, not general image description."""


COT_USER_TEMPLATE = """Image 1: {image1_desc}
Image 2: {image2_desc}
Question: {question}

Generate the structured safety analysis."""


class CoTGenerator:
    """
    Generates structured CoT labels for training samples.
    Supports two modes:
      - vllm: batch inference via local VLM (preferred, fast)
      - openai: GPT-4o API (fallback, expensive but high quality)
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "vllm",
        gpu_ids: Optional[list[int]] = None,
        tensor_parallel_size: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.1,
        openai_api_key: Optional[str] = None,
        output_path: Optional[Path] = None,
    ):
        self.model_path = model_path
        self.backend = backend
        self.gpu_ids = gpu_ids or [0]
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.openai_api_key = openai_api_key
        self.output_path = Path(output_path) if output_path else None

        self._llm = None  # lazy init

    def generate_batch(self, records: list[dict]) -> list[dict]:
        """
        Annotate records with CoT labels. Returns records with 'cot_response' filled.
        Records must have: question, image_path1, image_path2.
        """
        if self.backend == "vllm":
            return self._generate_vllm(records)
        elif self.backend == "openai":
            return self._generate_openai(records)
        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def generate_file(
        self,
        input_json: Path,
        output_json: Path,
        resume: bool = True,
    ) -> Path:
        """Process a full dataset file, with resume support."""
        with open(input_json) as f:
            records = json.load(f)
        if isinstance(records, dict):
            records = list(records.values())

        # Resume: skip already-annotated records
        done_ids: set = set()
        if resume and output_json and Path(output_json).exists():
            with open(output_json) as f:
                existing = json.load(f)
            done_ids = {r["id"] for r in existing if r.get("cot_response")}
            print(f"Resuming: {len(done_ids)} already annotated, {len(records) - len(done_ids)} remaining")

        pending = [r for r in records if r.get("id") not in done_ids]
        if not pending:
            print("All records already annotated.")
            return Path(output_json)

        annotated = self.generate_batch(pending)

        if resume and done_ids:
            existing_map = {r["id"]: r for r in existing}
            for r in annotated:
                existing_map[r["id"]] = r
            all_records = list(existing_map.values())
        else:
            all_records = annotated

        output_json = Path(output_json)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)
        return output_json

    # ── vLLM backend ──────────────────────────────────────────────────────

    def _generate_vllm(self, records: list[dict]) -> list[dict]:
        import os
        if self.gpu_ids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, self.gpu_ids))

        from vllm import LLM, SamplingParams

        if self._llm is None:
            self._llm = LLM(
                model=self.model_path,
                trust_remote_code=True,
                tensor_parallel_size=self.tensor_parallel_size,
                max_model_len=4096,
                limit_mm_per_prompt={"image": 2},
                gpu_memory_utilization=0.9,
            )

        sampling = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        prompts = [self._build_vllm_prompt(r) for r in records]
        outputs = self._llm.generate(prompts, sampling)

        results = []
        for r, out in zip(records, outputs):
            cot = out.outputs[0].text.strip()
            results.append({**r, "cot_response": cot})
        return results

    def _build_vllm_prompt(self, r: dict) -> dict:
        """Build vLLM multi-modal prompt dict."""
        from PIL import Image
        imgs = []
        for key in ("image_path1", "image_path2"):
            p = r.get(key, "")
            if p:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:
                    imgs.append(Image.new("RGB", (224, 224)))

        messages = [
            {"role": "system", "content": COT_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "placeholder1"}},
                    {"type": "image_url", "image_url": {"url": "placeholder2"}},
                    {"type": "text", "text": f"Question: {r['question']}\n\nGenerate the structured safety analysis."},
                ],
            },
        ]
        return {"prompt": messages, "multi_modal_data": {"image": imgs}}

    # ── OpenAI fallback ───────────────────────────────────────────────────

    def _generate_openai(self, records: list[dict]) -> list[dict]:
        import base64
        import os
        from openai import OpenAI

        api_key = self.openai_api_key or os.environ.get("OPENAI_API_KEY")
        client = OpenAI(api_key=api_key)

        def encode(path: str) -> str:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()

        results = []
        for r in records:
            content = [{"type": "text", "text": COT_SYSTEM_PROMPT + f"\n\nQuestion: {r['question']}\n\nGenerate the structured safety analysis."}]
            for key in ("image_path1", "image_path2"):
                p = r.get(key, "")
                if p:
                    b64 = encode(p)
                    content.insert(0, {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": content}],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            cot = response.choices[0].message.content.strip()
            results.append({**r, "cot_response": cot})

        return results
