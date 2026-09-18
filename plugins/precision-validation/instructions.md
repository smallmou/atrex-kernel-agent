### Precision validation

Production promotion runs an independent precision gate that you do not invoke and
cannot satisfy by re-running the ordinary evaluator. The supervisor owns it: it
regenerates every operator input from construction families declared in an
operator-owned `numerical_suite.json`, runs the vendored Atrex-Bench comparison on
those inputs inside a separate GPU allocation, and then has an isolated reviewer
judge the arithmetic against the trusted reference.

What this means for your kernel:

- Correctness must hold on **arbitrary legal inputs**, not on the sample distribution
  the ordinary generator happens to produce. Additional seeds of that generator are
  explicitly not evidence of precision safety.
- Wide magnitudes, long-reduction cancellation, sparse and zero operands, saturated or
  nearly equal softmax logits, near-constant low-variance normalization inputs, tiny
  values near epsilon, quantized scale hierarchies, routing ties, masked and empty
  work, and tails are all probed. Assume each will be exercised at the shapes and
  ranks the operator declares.
- Every distributed rank runs each selected probe. A reduction that is only correct on
  rank 0, or that assumes identical inputs across ranks, will be rejected.
- Accumulator precision, reassociation, epsilon placement, reciprocal and sqrt
  domains, and behavior near zero are reviewed against the reference, not against a
  tolerance threshold you can approach.

The gate reads your kernel and the trusted contract as evidence and cites both. Do not
edit the evaluator, the harness, the suite, tolerances, or the supervisor to obtain a
pass; the gate re-digests its own evidence and rejects a candidate whose contract
changed underneath it. If a legitimate operator constraint makes a probed case invalid,
state that constraint in your solution notes rather than loosening a check.
