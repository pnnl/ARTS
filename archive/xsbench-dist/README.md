# xsbench_dist (retired driver)

A distributed SPMD rewrite of the XSBench OCR port: one deterministic table
replica per rank, inline lookup chains, reusing the intel-sharedDB physics
sources.  Never registered in the build (its `add_benchmark_app` block was
held back) and never carried by the application submodule.

Retired 2026-08-22: any distribution-friendly rebuild of `XSBench_intel`
converges on `XSBench_intel_sharedDB` — per-lookup tasking costs more
runtime work per lookup than the lookup kernel itself, so the only way up
is coarse inline chains over a per-instance replica, which is exactly what
that sibling already is (and what this driver was).  The intel/sharedDB
pair is the decomposition ablation; a third row would duplicate the second.

Note if ever revived: the sharedDB sources this driver borrowed have since
replaced the unionized grid's embedded `xs_ptrs` pointers with flat
`xs_grid` index arithmetic — `set_grid_ptrs` takes the xs_grid pointer now,
so this file's call (and its own kernel plumbing) needs the same adaptation.
