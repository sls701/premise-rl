# GSPO CUDA Illegal Memory Access Debugging Implementation Plan

## Goal

Fix or isolate the `RuntimeError: CUDA error: an illegal memory access was encountered` occurring around:

```python
torch.cuda.synchronize()
```

in `src/train/gspo.py` after rollout and before the GSPO update step.

Important: the `synchronize()` call is probably not the root cause. It is where a previous asynchronous CUDA failure is finally being reported. The implementation should make CUDA errors surface at the true call site, then apply safety fixes around generation, token indexing, attention kernels, and training forwards.

The user is running on an NVIDIA L40S GPU. Treat L40S + bf16 + SDPA/Flash-style attention as a possible instability source, but do not assume hardware failure before checking the software issues below.

---

## Priority 0: Preserve the Original File

Before editing:

```bash
cp src/train/gspo.py src/train/gspo.py.bak
```

Then make changes incrementally and test after each priority block.

---

## Priority 1: Stop Swallowing CUDA Exceptions During Rollout

### Problem

The current rollout uses `asyncio.gather(..., return_exceptions=True)`. If a CUDA operation fails inside one rollout task, the exception may be returned as a value and filtered out later. The CUDA context can remain poisoned, causing the next `torch.cuda.synchronize()` to crash with a misleading traceback.

### Current Pattern

```python
return await asyncio.gather(*[
    _run_group_batched(
        t, model, tokenizer, sc,
        system_prompt, group_size, horizon, top_k,
        alpha, beta, temperature, max_new_tokens, device,
    )
    for t in targets
], return_exceptions=True)
```

Later:

```python
episode_groups = [g for g in results if not isinstance(g, Exception) and g]
```

### Required Change

Remove `return_exceptions=True` from the rollout-level gather.

```python
return await asyncio.gather(*[
    _run_group_batched(
        t, model, tokenizer, sc,
        system_prompt, group_size, horizon, top_k,
        alpha, beta, temperature, max_new_tokens, device,
    )
    for t in targets
])
```

Then simplify:

```python
episode_groups = [g for g in results if g]
```

### Required Change to Exception Handler

In the outer rollout exception handler, do not try to recover from CUDA illegal memory access. Re-raise CUDA errors.

Replace the current rollout `except` block with something like:

```python
try:
    results = asyncio.run(_rollout(batch))
except RuntimeError as exc:
    logger.exception("Rollout failed")
    if "CUDA" in str(exc) or "cuda" in str(exc):
        raise
    torch.cuda.empty_cache()
    continue
except Exception:
    logger.exception("Rollout failed")
    torch.cuda.empty_cache()
    continue
```

### Acceptance Criteria

- A rollout CUDA failure should stop the program immediately.
- The traceback should point closer to the actual failing operation, likely inside `generate()` or an indexing/logits operation.
- The code should no longer filter exception objects out of `results`.

---

## Priority 2: Add Token ID Bounds Checks Before CUDA Indexing

### Problem

These lines can trigger CUDA device-side asserts or illegal memory access if `comp_ids_t` contains an invalid token ID:

```python
cur_token_lp = cur_lp[range(n_c), comp_ids_t]
ref_token_lp = ref_lp[range(n_c), comp_ids_t]
```

Even if the invalid index occurs here, the error may surface later at `torch.cuda.synchronize()`.

### Required Change

Before indexing into `cur_lp` or `ref_lp`, validate completion token IDs against the logits vocabulary size.

Add this after computing `logit_slice` and before `F.log_softmax`:

```python
if logit_slice.shape[0] != n_c:
    logger.error(
        "Bad current-policy slice: n_p=%d n_c=%d slice_shape=%s seq_len=%d",
        n_p, n_c, tuple(logit_slice.shape), seq.shape[1],
    )
    del seq, logits, comp_ids_t, logit_slice
    continue

vocab_size = logit_slice.shape[-1]
if min(step.completion_ids) < 0 or max(step.completion_ids) >= vocab_size:
    logger.error(
        "Completion token out of range for current policy: min=%d max=%d vocab_size=%d n_c=%d",
        min(step.completion_ids), max(step.completion_ids), vocab_size, n_c,
    )
    del seq, logits, comp_ids_t, logit_slice
    continue
```

Then replace advanced indexing with `gather`:

```python
cur_lp = F.log_softmax(logit_slice, dim=-1)
cur_token_lp = cur_lp.gather(1, comp_ids_t.view(-1, 1)).squeeze(1)
```

Do the equivalent checks for the reference-policy logits:

```python
ref_logit_slice = ref_logits[0, n_p - 1: n_p + n_c - 1].contiguous().float()

if ref_logit_slice.shape[0] != n_c:
    logger.error(
        "Bad reference-policy slice: n_p=%d n_c=%d slice_shape=%s seq_len=%d",
        n_p, n_c, tuple(ref_logit_slice.shape), seq.shape[1],
    )
    del seq, logits, ref_logits, comp_ids_t, logit_slice, cur_lp, cur_token_lp
    continue

ref_vocab_size = ref_logit_slice.shape[-1]
if min(step.completion_ids) < 0 or max(step.completion_ids) >= ref_vocab_size:
    logger.error(
        "Completion token out of range for reference policy: min=%d max=%d vocab_size=%d n_c=%d",
        min(step.completion_ids), max(step.completion_ids), ref_vocab_size, n_c,
    )
    del seq, logits, ref_logits, comp_ids_t, logit_slice, cur_lp, cur_token_lp, ref_logit_slice
    continue

ref_lp = F.log_softmax(ref_logit_slice, dim=-1)
ref_token_lp = ref_lp.gather(1, comp_ids_t.view(-1, 1)).squeeze(1)
```

### Optional Cleanup

Use `torch.arange` if keeping advanced indexing, but `gather` is preferred:

```python
idx = torch.arange(n_c, device=device)
cur_token_lp = cur_lp[idx, comp_ids_t]
```

### Acceptance Criteria

- No GPU indexing should occur before token IDs are validated.
- If bad token IDs exist, the logs should show `min`, `max`, `vocab_size`, and `n_c`.
- The code should use `gather` instead of `cur_lp[range(n_c), comp_ids_t]`.

---

## Priority 3: Log Tokenizer and Model Vocabulary Compatibility

### Problem

If tokenizer IDs can exceed the model embedding/logits vocabulary, generation or update indexing can fail.

### Required Change

After loading tokenizer and model, log vocabulary sizes and special token IDs:

```python
logger.info(
    "tokenizer_len=%d model_vocab=%d eos_token_id=%s pad_token_id=%s",
    len(tokenizer),
    model.config.vocab_size,
    tokenizer.eos_token_id,
    tokenizer.pad_token_id,
)

assert tokenizer.eos_token_id is not None, "tokenizer.eos_token_id must be set"
assert tokenizer.eos_token_id < model.config.vocab_size, "eos_token_id exceeds model vocab"
```

If `len(tokenizer) > model.config.vocab_size`, resize embeddings:

```python
if len(tokenizer) > model.config.vocab_size:
    logger.warning(
        "Tokenizer has more tokens than model vocab: len(tokenizer)=%d model_vocab=%d. Resizing embeddings.",
        len(tokenizer), model.config.vocab_size,
    )
    model.resize_token_embeddings(len(tokenizer))
```

### Acceptance Criteria

- Training logs clearly show tokenizer length, model vocabulary size, EOS token ID, and PAD token ID.
- If the tokenizer is larger than the model vocab, embeddings are resized before LoRA wrapping or before training begins.

---

## Priority 4: Disable KV Cache During Training Forwards

### Problem

The update step performs forward + backward passes, but the model may still use cache behavior intended for generation. This can increase memory pressure and sometimes interact badly with attention backends.

### Required Change

After loading the model:

```python
model.config.use_cache = False
```

In the GSPO update step, change current-policy forward:

```python
logits = model(seq, use_cache=False).logits
```

Change reference-policy forward:

```python
with model.disable_adapter(), torch.no_grad():
    ref_logits = model(seq, use_cache=False).logits
```

During generation, explicitly request cache if desired:

```python
out = model.generate(
    input_ids,
    attention_mask=attn_mask,
    max_new_tokens=max_new_tokens,
    temperature=temperature,
    do_sample=True,
    pad_token_id=tokenizer.eos_token_id,
    logits_processor=LogitsProcessorList([_SafeLogitsProcessor()]),
    use_cache=True,
)
```

### Acceptance Criteria

- Training forwards pass `use_cache=False`.
- Generation explicitly passes `use_cache=True` or uses the intended default.
- `model.config.use_cache = False` is set after model load.

---

## Priority 5: Force Eager/Math Attention on L40S While Debugging

### Problem

On L40S, bf16 + SDPA/Flash/memory-efficient attention can be a source of hard-to-localize CUDA errors. Temporarily force eager/math attention to isolate whether the problem is kernel/backend-related.

### Required Change

Before creating the model, add:

```python
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
```

Change model loading from SDPA to eager attention:

```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="eager",
)
```

Note: use `torch_dtype`, not `dtype`, unless the installed Transformers version specifically requires otherwise.

### Acceptance Criteria

- Debug run uses eager attention.
- If the illegal memory access disappears under eager attention, document that SDPA/Flash backend is the likely source.
- Once stable, optionally benchmark returning to SDPA.

---

## Priority 6: Remove Manual Padding to Multiple of 8 in Update Step

### Problem

The current code manually appends token ID `0` to pad sequence length to a multiple of 8:

```python
_n_raw = n_p + n_c
_pad8 = (-_n_raw) % 8
_ids = step.prompt_ids + step.completion_ids + [0] * _pad8
seq = torch.tensor(
    _ids, dtype=torch.long, device=device,
).unsqueeze(0)
```

This is not necessary for normal transformer forwards and is risky without an attention mask. It changes the actual token sequence fed to the model.

### Required Change

Replace with:

```python
_ids = step.prompt_ids + step.completion_ids
seq = torch.tensor(_ids, dtype=torch.long, device=device).unsqueeze(0)
```

### Acceptance Criteria

- Update-step sequence contains only prompt IDs followed by completion IDs.
- No artificial token ID `0` padding is appended in the single-sequence update forward.

---

## Priority 7: Add Finite Checks Around Logits, Log-Probs, and Loss

### Problem

The code checks `logit_slice.isnan().any()`, but this misses positive/negative infinities and does not check reference logits, log-probs, ratio, or loss.

### Required Change

Replace:

```python
if logit_slice.isnan().any():
```

with:

```python
if not torch.isfinite(logit_slice).all():
    logger.warning("Non-finite current logits in backward (n_p=%d, n_c=%d); skipping", n_p, n_c)
    del seq, logits, comp_ids_t, logit_slice
    continue
```

Add similar checks:

```python
if not torch.isfinite(ref_logit_slice).all():
    logger.warning("Non-finite reference logits in backward (n_p=%d, n_c=%d); skipping", n_p, n_c)
    del seq, logits, ref_logits, comp_ids_t, logit_slice, ref_logit_slice
    continue
```

After log-probs:

```python
if not torch.isfinite(cur_token_lp).all() or not torch.isfinite(ref_token_lp).all():
    logger.warning("Non-finite token logprobs; skipping step")
    del seq, logits, ref_logits, comp_ids_t, logit_slice, ref_logit_slice
    del cur_lp, ref_lp, cur_token_lp, ref_token_lp
    continue
```

Before backward:

```python
if not torch.isfinite(step_loss):
    logger.warning("Non-finite step loss; skipping backward")
    del seq, logits, ref_logits, comp_ids_t
    del cur_lp, ref_lp, cur_token_lp, ref_token_lp
    continue
```

### Acceptance Criteria

- Non-finite values are detected and skipped before `backward()`.
- Logs distinguish current-policy logits, reference-policy logits, token log-probs, and loss failures.

---

## Priority 8: Make Synchronization Debug-Friendly, Not Recovery-Oriented

### Problem

`torch.cuda.synchronize()` is used many times. That is fine for debugging, especially with `CUDA_LAUNCH_BLOCKING=1`, but once an illegal memory access has occurred, later cleanup/synchronize calls can obscure the source.

### Required Change

Keep synchronization near suspected operations, especially after `generate()` while debugging:

```python
out = model.generate(...)
torch.cuda.synchronize()
```

But avoid doing this in exception handlers after CUDA RuntimeErrors:

```python
except RuntimeError as exc:
    logger.exception("...")
    if "CUDA" in str(exc) or "cuda" in str(exc):
        raise
```

### Acceptance Criteria

- CUDA exceptions are not swallowed and then followed by more CUDA calls.
- The program fails fast on CUDA RuntimeErrors.

---

## Priority 9: Optional Larger Refactor — Use a Separate Frozen Reference Model

### Problem

The current update step computes reference logits by temporarily disabling the LoRA adapter:

```python
with model.disable_adapter(), torch.no_grad():
    ref_logits = model(seq).logits
```

This is convenient but mutates adapter state inside the training loop. It is probably not the main illegal-memory source, but a separate reference model is cleaner and more robust.

### Optional Change

Load a separate frozen base model:

```python
ref_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="eager",
)
ref_model.eval()
ref_model.requires_grad_(False)
ref_model.config.use_cache = False
```

Pass `ref_model` into `gspo_update_step` and replace:

```python
with model.disable_adapter(), torch.no_grad():
    ref_logits = model(seq, use_cache=False).logits
```

with:

```python
with torch.no_grad():
    ref_logits = ref_model(seq, use_cache=False).logits
```

### Acceptance Criteria

- No adapter toggling occurs inside the update loop.
- Reference logits come from a frozen model.
- Only implement this if GPU memory is sufficient.

---

## Priority 10: Suggested Debug Environment Variables

Run with:

```bash
CUDA_LAUNCH_BLOCKING=1 \
TORCH_SHOW_CPP_STACKTRACES=1 \
python -m src.train.gspo --config configs/gspo_intra.yaml
```

For deeper device-side assertions, PyTorch must be built with:

```bash
TORCH_USE_CUDA_DSA=1
```

Note: setting `TORCH_USE_CUDA_DSA=1` at runtime usually does not help unless PyTorch was compiled with that flag.

---

## Priority 11: Suggested First Test Run

Use a very small config to reproduce quickly:

```yaml
max_targets: 2
batch_size: 1
group_size: 2
horizon: 2
grad_accum: 1
max_completion_length: 64
```

Run one debug pass with eager attention and no swallowed exceptions.

Expected outcomes:

1. If the crash disappears: likely SDPA/Flash/bf16 backend instability or memory pressure.
2. If the crash moves into `generate()`: focus on sampling/logits validity and attention backend.
3. If the crash moves into update indexing: token ID bounds or shape mismatch was likely the culprit.
4. If logs show bad token IDs: fix tokenizer/model vocabulary mismatch or generated-token handling.

---

## Final Recommended Patch Order

1. Remove `return_exceptions=True` and re-raise CUDA errors.
2. Force eager/math attention on L40S while debugging.
3. Set `model.config.use_cache = False`; pass `use_cache=False` during update forwards.
4. Add tokenizer/model vocab logging and assertions.
5. Add token ID bounds checks before CUDA indexing.
6. Replace advanced indexing with `gather`.
7. Remove manual multiple-of-8 padding in the update step.
8. Add finite checks for logits, log-probs, and loss.
9. Consider separate frozen reference model if instability persists and memory allows.

---

## Notes for the Implementing Agent

- Do not treat the line containing `torch.cuda.synchronize()` as the root cause by default.
- Do not keep running after a CUDA illegal memory access; fail fast.
- Prefer changes that make the true failure site visible before attempting performance optimizations.
- Keep eager attention until correctness is established, then benchmark SDPA separately.
- Make the smallest possible change per test run so the cause can be isolated.
