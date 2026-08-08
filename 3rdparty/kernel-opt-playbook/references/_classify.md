# New operator → category decision table

Given a **new operator**, decide its category from the **compute pattern** (not the
name), then pull optimization experiences with `query.py --category <cat>`.

| Category | Compute-pattern signature (match → this category) | Typical forms |
|---|---|---|
| `attention` | Q·Kᵀ → softmax → ·V; causal/mask; KV cache; GQA/MHA/MLA | flash attention, paged decode, prefill, mask prep |
| `normalization` | mean/variance/RMS over the last dim then scale; per-row normalize | rmsnorm, layernorm, groupnorm, instance norm |
| `gemm_matmul` | dense matmul A·B (optional bias/activation); projection layers | gemm, linear/projection, bmm |
| `moe` | token→expert routing, top-k gating, group scoring, expert dispatch | moe routing, expert compute, group score |
| `rope_position` | rotary position embedding, cos/sin generation, position indexing | rope, multimodal rope, freq computation |
| `activation_elementwise` | elementwise/pointwise ops, gated MLP, residual add | gelu/silu/swiglu, gated mlp, residual add |
| `conv` | convolution, sliding window, NHWC/NCHW, fused norm/activation | conv3x3, vae conv, causal conv |
| `mamba_ssm` | state-space recurrence, time decay, selective scan | mamba, ssm, time decay |
| `fft` | frequency-domain transform, rfft, frequency-domain conv | fft, hyena |
| `reduction_scatter` | reduction/scatter/gather/sort/split | scatter, gather, cumsum, topk sort |

## Priority rules (when ambiguous)
1. Any **Q·Kᵀ + softmax + ·V** structure → `attention` (even if it also contains rope/norm).
2. Dominated by **matmul** → `gemm_matmul`.
3. Dominated by **per-row normalization** → `normalization`.
4. Dominated by **elementwise / gating / residual** → `activation_elementwise`.
5. None fit → pick the category matching the heaviest compute; if truly unclear,
   pick the closest and also query the neighboring categories.

## attention subcategories (optional, for more precise retrieval)
`--subcategory` values: `paged_decode` / `ragged_prefill` / `prefill` / `mla` /
`mask` / `qk_norm_rope` / `projection` / `generic`

> Note: experiences in this library have their **original operator names removed** —
> you get "technique + representative code snippet" and must **transfer** it to your
> new operator, not copy an existing answer.
