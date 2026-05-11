"""
CoTGenerator: generate structured Safety Rationale labels for DREAMS training data.

Two modes:
  cot_only      — Original mode. Generates <safety_analysis> CoT block only.
                  Uses COT_SYSTEM_PROMPT. Output field: 'cot_response'.
  full_response — New mode. Generates <safety_analysis> CoT + natural safety
                  response in a single pass. Uses RESPONSE_GENERATION_SYSTEM.
                  Designed for Qwen3.5-122B-A10B (VLM). Output field: 'gpt_response'.

Output CoT format (structured):
  <safety_analysis>
  Step 1 [Visual Description]: ...
  Step 2 [Harm Source Analysis]: (A)... (B)... (C)...
  Step 3 [Harm Assessment]: ...
  Conclusion: [Unsafe] — [Harm Category]: ...
  </safety_analysis>
  {natural safety response prose}
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


RESPONSE_GENERATION_SYSTEM = """You are an expert AI safety researcher generating high-quality safety training data for vision-language models. Analyze each request with rigorous, grounded reasoning to detect both explicit and subtle harms while maintaining a constructive, educational response style.

## Harm Detection: Three-Source Framework

For every request, systematically check ALL three potential harm sources before forming your judgment:

[A] Single-Image Harm — Does either image, viewed in isolation, depict:
- Weapons, explosives, or tools designed to cause injury
- Controlled substances, precursor chemicals, or drug paraphernalia
- Graphic violence, severe injury, or death
- Explicit or sexualized content
- Vulnerable individuals (children, incapacitated persons, identifiable private individuals)
- Critical infrastructure in a vulnerable or compromised state

[B] Text-Level Harm — Does the question text itself signal harmful intent, independent of image content? Warning signs:
- Operational specificity: "detailed steps," "exact quantities," "how to synthesize/build/bypass/modify"
- Euphemistic phrasing that conceals violence or illegality: "affect," "handle with force," "convert," "release near"
- Targeting language: requests directed at specific persons, groups, locations, or infrastructure
- Scale language: phrases implying mass effects, distribution, or systemic impact

[C] Multimodal Combined Harm — Do the image pair + question together create an implicit harmful scenario absent from any single element alone? This is the most subtle and important category:
- A benign-looking object in one image + a specific environment in the other + an operational question = covert harmful procedure
- Two individually innocuous images whose combination reveals a dangerous method or exploit
- Visual context that disambiguates an ambiguous text request toward a clearly harmful interpretation

## Harm Category Reference
Identify which of these categories apply — this sharpens your assessment and grounds your response:

- Violent Activity: assault, weapon-enabled harm, incitement to physical violence
- Illegal Activity: theft, hacking/cyberattack, fraud, trafficking, property destruction, smuggling
- Self-Harm: suicide method facilitation, self-injury guidance, eating disorder promotion
- Privacy Violation: unauthorized surveillance, doxxing, non-consensual recording/tracking, identity theft
- Hate Speech: content targeting or inciting against race, religion, gender, nationality, sexuality, disability
- Erotic / Sexual Content: non-consensual sexual material, exploitation, sexual coercion
- WMD / Dangerous Weapons: explosives, chemical/biological/radiological agents, illegal weapon modification
- Controlled Substances: drug synthesis routes, trafficking logistics, abuse facilitation
- Misinformation: election manipulation, medical disinformation, fabricated evidence, deepfake production
- Intellectual Property: counterfeiting procedures, large-scale piracy instructions
- Child Safety: ANY content that could endanger, exploit, or sexualize minors — absolute prohibition, no negotiation
- Regulated Items: items legal in some jurisdictions but whose acquisition/use is being facilitated in harmful ways

## Response Format

PART 1 — Structured Analysis (always include, use XML tags exactly as shown):
<safety_analysis>
Step 1 [Visual Description]: Describe the content of Image 1 and Image 2 — objects, settings, persons, substances, or any visually relevant elements. Be specific; do not generalize.
Step 2 [Harm Source Analysis]:
  (A) Single-image: Is either image harmful in isolation? Describe what makes it dangerous if so.
  (B) Text-level: Does the question language itself signal harmful intent? Quote specific phrases if present.
  (C) Combined: What specific danger emerges from this particular image pair + question that would not exist from any element alone? Describe the implied harmful procedure or scenario.
Step 3 [Harm Assessment]: (a) Identify the harm category and describe the concrete harm if this request were fulfilled. (b) Is this harm specifically grounded in these images, or would it generalize to many similar images? (c) Counterfactual — would the request be substantially safer if either image depicted something benign instead?
Conclusion: [Unsafe] — [Harm Category]: {one-sentence specific risk statement}
</safety_analysis>

PART 2 — Safety Response (natural prose, no section headers, immediately after the closing tag):
Write 3–5 sentences in natural, non-preachy conversational style that:
- Opens by describing specific visual observations from both images (ground your response — a generic refusal that ignores the images is unacceptable)
- Names the specific safety concern and harm category in accessible language
- Declines the harmful aspect of the request with concise, clear reasoning
- Offers a genuinely applicable constructive alternative where one exists: safety information, legal approaches, professional resources, or educational context — do NOT fabricate alternatives that do not exist
- For Self-Harm: include relevant support resources (crisis lines, professional help)
- For Child Safety and WMD: firm refusal without offering alternatives or negotiating
- Does NOT provide step-by-step instructions, synthesis routes, technical specifications, bypass methods, or any operationally useful harmful guidance"""


COT_USER_TEMPLATE = """Image 1: {image1_desc}
Image 2: {image2_desc}
Question: {question}

Generate the structured safety analysis."""


class CoTGenerator:
    """
    Generates structured CoT labels or full safety responses for training samples.

    mode="cot_only"      — structured <safety_analysis> block only (original behaviour)
    mode="full_response" — <safety_analysis> + natural safety response (for DREAMS SFT labels)

    Backends:
      vllm   — batch inference via local VLM (preferred)
      openai — GPT-4o API (fallback)
    """

    def __init__(
        self,
        model_path: str,
        backend: str = "vllm",
        gpu_ids: Optional[list[int]] = None,
        tensor_parallel_size: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.1,
        mode: str = "cot_only",
        openai_api_key: Optional[str] = None,
        output_path: Optional[Path] = None,
    ):
        self.model_path = model_path
        self.backend = backend
        self.gpu_ids = gpu_ids or [0]
        self.tensor_parallel_size = tensor_parallel_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.mode = mode  # "cot_only" | "full_response"
        self.openai_api_key = openai_api_key
        self.output_path = Path(output_path) if output_path else None

        self._llm = None  # lazy init

    def generate_batch(self, records: list[dict]) -> list[dict]:
        """
        Annotate records. Returns records with output field filled.
          mode=cot_only      → 'cot_response' field
          mode=full_response → 'gpt_response' field
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

        output_field = "gpt_response" if self.mode == "full_response" else "cot_response"

        done_ids: set = set()
        if resume and output_json and Path(output_json).exists():
            with open(output_json) as f:
                existing = json.load(f)
            done_ids = {r["id"] for r in existing if r.get(output_field)}
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

        output_field = "gpt_response" if self.mode == "full_response" else "cot_response"
        results = []
        for r, out in zip(records, outputs):
            text = out.outputs[0].text.strip()
            results.append({**r, output_field: text})
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

        system = (
            RESPONSE_GENERATION_SYSTEM
            if self.mode == "full_response"
            else COT_SYSTEM_PROMPT
        )

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "placeholder1"}},
                    {"type": "image_url", "image_url": {"url": "placeholder2"}},
                    {"type": "text", "text": r["question"]},
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

        system = (
            RESPONSE_GENERATION_SYSTEM
            if self.mode == "full_response"
            else COT_SYSTEM_PROMPT
        )
        output_field = "gpt_response" if self.mode == "full_response" else "cot_response"

        def encode(path: str) -> str:
            with open(path, "rb") as f:
                return base64.b64encode(f.read()).decode()

        results = []
        for r in records:
            content: list[dict] = []
            for key in ("image_path1", "image_path2"):
                p = r.get(key, "")
                if p:
                    b64 = encode(p)
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    })
            content.append({"type": "text", "text": r["question"]})

            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
            text = response.choices[0].message.content.strip()
            results.append({**r, output_field: text})

        return results
