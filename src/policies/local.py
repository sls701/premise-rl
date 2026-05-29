"""
Local policy: Qwen3-4B with optional LoRA adapters.
Used for the trained policy in Phase 7 eval and during GRPO training rollouts.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class _BatchCoordinator:
    """Batches concurrent model.generate() calls from multiple async episodes.

    Multiple chat() coroutines submit prompts to an asyncio queue.  A single
    background task drains the queue in batches: it waits up to `window_s`
    seconds for more requests to arrive, then runs one model.generate() for
    all collected prompts together.  This gives ~concurrency-fold throughput
    improvement over serialised single-prompt generation.
    """

    def __init__(self, model, tokenizer, temperature: float, max_new_tokens: int,
                 window_s: float = 0.05):
        self._model = model
        self._tokenizer = tokenizer
        self._temperature = temperature
        self._max_new_tokens = max_new_tokens
        self._window_s = window_s
        self._device = next(model.parameters()).device
        self._queue: asyncio.Queue | None = None
        self._task: asyncio.Task | None = None

    def _ensure_started(self) -> None:
        if self._queue is None:
            self._queue = asyncio.Queue()
            self._task = asyncio.create_task(self._run_forever())

    async def generate(self, prompt: str) -> str:
        self._ensure_started()
        assert self._queue is not None
        future: asyncio.Future[str] = asyncio.get_event_loop().create_future()
        await self._queue.put((prompt, future))
        return await future

    async def _run_forever(self) -> None:
        assert self._queue is not None
        while True:
            first_prompt, first_fut = await self._queue.get()
            batch: list[tuple[str, asyncio.Future]] = [(first_prompt, first_fut)]

            deadline = asyncio.get_event_loop().time() + self._window_s
            while True:
                remaining = deadline - asyncio.get_event_loop().time()
                if remaining <= 0:
                    break
                try:
                    item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
                    batch.append(item)
                except asyncio.TimeoutError:
                    break

            prompts = [p for p, _ in batch]
            futures = [f for _, f in batch]
            logger.info("batch_generate: n=%d", len(prompts))

            loop = asyncio.get_event_loop()
            try:
                outputs = await loop.run_in_executor(None, self._generate_batch, prompts)
                for fut, text in zip(futures, outputs):
                    if not fut.done():
                        fut.set_result(text)
            except Exception as exc:
                for fut in futures:
                    if not fut.done():
                        fut.set_exception(exc)

    def _generate_batch(self, prompts: list[str]) -> list[str]:
        import torch

        self._tokenizer.padding_side = "left"
        enc = self._tokenizer(
            prompts, return_tensors="pt", padding=True,
            truncation=True, max_length=1536, pad_to_multiple_of=64,
        )
        input_ids = enc["input_ids"].to(self._device)
        attn_mask = enc["attention_mask"].to(self._device)
        padded_len = input_ids.shape[1]

        do_sample = self._temperature > 0
        torch.cuda.empty_cache()
        with torch.no_grad():
            out = self._model.generate(
                input_ids,
                attention_mask=attn_mask,
                max_new_tokens=self._max_new_tokens,
                temperature=self._temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        torch.cuda.synchronize()

        results = []
        for j in range(len(prompts)):
            new_ids = out[j, padded_len:]
            results.append(self._tokenizer.decode(new_ids, skip_special_tokens=True))
        return results

SEARCH_TOOL = [{
    "type": "function",
    "function": {
        "name": "search_theorems",
        "description": "Search the theorem corpus by natural language query",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language search query"},
                "k": {"type": "integer", "description": "Number of results to return", "default": 10},
            },
            "required": ["query"],
        },
    },
}]


def _to_qwen_native(messages: list[dict]) -> list[dict]:
    """Convert the pipeline's custom [tool_result]/[search_theorems] messages to
    Qwen's native <tool_call>/<tool_response> format so the model can use its
    pretrained tool-use behaviour instead of relearning the custom format."""
    import re
    result = []
    for msg in messages:
        content = msg.get("content", "") or ""
        if msg["role"] == "user" and content.startswith("[tool_result]\n"):
            result.append({
                "role": "tool",
                "name": "search_theorems",
                "content": content[len("[tool_result]\n"):],
            })
        elif msg["role"] == "assistant" and content.startswith("[search_theorems]"):
            m = re.search(r"query='([^']*)'", content)
            query = m.group(1) if m else ""
            result.append({
                "role": "assistant",
                "content": f'<tool_call>\n{json.dumps({"name": "search_theorems", "arguments": {"query": query}})}\n</tool_call>',
            })
        else:
            result.append(msg)
    return result


class LocalAgent:
    def __init__(
        self,
        model_id: str,
        checkpoint_dir: str | Path | None = None,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
        use_vllm: bool = False,
    ):
        self.model_id = model_id
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.use_vllm = use_vllm
        self._model = None
        self._tokenizer = None
        self._vllm_client = None
        self._coordinator: _BatchCoordinator | None = None

    def _load(self) -> None:
        if self._model is not None:
            return

        if self.use_vllm:
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_transformers(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        logger.info("Loading %s with transformers...", self.model_id)
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        if self.checkpoint_dir and self.checkpoint_dir.exists():
            logger.info("Loading LoRA from %s", self.checkpoint_dir)
            self._model = PeftModel.from_pretrained(self._model, str(self.checkpoint_dir))
        self._model.eval()
        self._coordinator = _BatchCoordinator(
            self._model, self._tokenizer, self.temperature, self.max_new_tokens,
        )

    def _load_vllm(self) -> None:
        from vllm import LLM, SamplingParams
        logger.info("Loading %s with vLLM...", self.model_id)
        self._vllm_client = LLM(model=self.model_id, dtype="bfloat16")

    def _build_prompt(self, messages: list[dict]) -> str:
        assert self._tokenizer is not None
        return self._tokenizer.apply_chat_template(
            _to_qwen_native(messages), tools=SEARCH_TOOL, tokenize=False,
            add_generation_prompt=True, enable_thinking=False,
        )

    async def chat(self, messages: list[dict]) -> tuple[str | None, int | None]:
        self._load()

        if self.use_vllm:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._chat_vllm, messages
            )

        assert self._coordinator is not None
        prompt = self._build_prompt(messages)
        text = await self._coordinator.generate(prompt)
        return self._parse_tool_call(text)

    def _chat_transformers(self, messages: list[dict]) -> tuple[str | None, int | None]:
        import torch
        prompt = self._build_prompt(messages)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        do_sample = self.temperature > 0
        with torch.no_grad():
            outputs = self._model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature if do_sample else None,
                do_sample=do_sample,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._parse_tool_call(text)

    def _chat_vllm(self, messages: list[dict]) -> tuple[str | None, int | None]:
        from vllm import SamplingParams
        prompt = self._build_prompt(messages)
        params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )
        outputs = self._vllm_client.generate([prompt], params)
        text = outputs[0].outputs[0].text
        return self._parse_tool_call(text)

    def _parse_tool_call(self, text: str) -> tuple[str | None, int | None]:
        # Qwen3's chat template emits tool calls as JSON inside <tool_call> tags
        import re
        m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                if data.get("name") == "search_theorems":
                    args = data.get("arguments", data.get("parameters", {}))
                    query = args.get("query", "")
                    k = int(args.get("k", 10))
                    return query, k
            except (json.JSONDecodeError, KeyError):
                pass

        # Fallback: look for JSON blob with a "query" key
        m2 = re.search(r'\{"query":\s*"([^"]+)"', text)
        if m2:
            return m2.group(1), 10

        return None, None


def make_local_agent(config: dict, checkpoint_dir: str | None = None) -> LocalAgent:
    return LocalAgent(
        model_id=config["model"],
        checkpoint_dir=checkpoint_dir or config.get("checkpoint_dir"),
        temperature=config.get("temperature", 0.0),
        max_new_tokens=config.get("max_tokens", 512),
        use_vllm=config.get("use_vllm", False),
    )
