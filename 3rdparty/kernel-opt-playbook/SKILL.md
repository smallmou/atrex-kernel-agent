---
name: kernel-opt-playbook
description: >
  Optimize a GPU kernel (CUDA or Triton) by looking up proven optimization
  techniques for its operator category. Use when the user wants to "optimize a
  kernel/operator", asks "how do I speed up this kernel", "any known tricks for
  attention/rmsnorm/gemm/moe/rope/conv", or wants category-level optimization
  patterns. Experiences are organized by operator category, carry no original
  operator names, and are tagged cuda/triton — you transfer the technique to your
  new kernel rather than copy a specific answer.
---

# Kernel Optimization Playbook

A library of **3889 effective GPU-kernel optimization experiences**, organized by
**operator category**. Each experience is anonymized (no original operator names) so
you learn and transfer the *technique* instead of reverse-looking-up a specific
answer. Every experience is tagged with its framework (cuda / triton / gluon).

## Workflow

### 1. Classify the new operator
Read `references/_classify.md` and decide the category from the operator's
**compute pattern** (not its name): `attention` / `normalization` / `gemm_matmul` /
`moe` / `rope_position` / `activation_elementwise` / `conv` / `mamba_ssm` / `fft` /
`reduction_scatter`.

### 2. Fix the framework
Determine whether the new kernel is **CUDA** or **Triton** — pull experiences from
the same framework only (the techniques differ a lot).
```bash
python3 query.py --list          # categories, counts, triton/cuda split
```

### 3. Pull techniques for the category (ranked by reliability)
```bash
python3 query.py --category <cat> --framework <triton|cuda>
# attention supports subcategories:
python3 query.py --category attention --framework triton --subcategory paged_decode
```
Output is a list of techniques ranked by **(usage count × median speedup)** — the
most reliable first — each with "what was done / why it worked / common pitfall+fix /
a representative code snippet".

### 4. Dig into a specific technique
```bash
python3 query.py --category <cat> --framework <fw> --tag <technique_tag> --full
```
Returns every anonymized record under that technique (full code snippets, pitfalls).

### 5. Apply + verify
- Apply techniques from most to least reliable, **transferring** each to your kernel
  (adapt variable names / shapes to fit your operator).
- After each change: **check correctness first** (diff against a reference), **then
  measure geomean**, and keep only changes that are genuinely faster.
- Use each experience's `pitfalls` to avoid known traps (e.g. too many warps hurting
  SM utilization).

## Common technique tags (cross-category)
`fusion` (kernel fusion — highest frequency and reliability), `warp_count_tuning`
(num_warps tuning), `dispatch_threshold` (shape-based kernel dispatch),
`launch_overhead_reduction`, `framework_migration`, `cuda_graph`, `tf32`,
`vectorization`, `block_size_tuning`, `persistent_kernel`, `cache_modifier`,
`loop_unroll`, `division_optimization`, `hybrid_dispatch`.

## Data layout
- `data/index.json` — global index (per-category counts, framework split,
  subcategories, technique frequencies, speedups).
- `data/by_category/<cat>.jsonl` — all records for the category (sorted by speedup),
  fields: `operator_category / operator_subcategory / framework / hardware /
  technique_tags / speedup_vs_prev_kept / what_changed / why_it_worked / pitfalls /
  evidence / key_snippets`.
- All text is anonymized: original operator names, workload UUIDs, and file paths
  are stripped.
