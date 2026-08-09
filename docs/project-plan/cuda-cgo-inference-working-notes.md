# CUDA/cgo inference working notes

These notes record the repository paths selected before implementation. They are
an implementation map, not the final operator documentation.

- `wordleml/policy` defines the exact GoMLX reference graph and stable variable
  scopes. Its dense checkpoint weights are input-major and therefore need one
  transpose into the exported output-major CUDA layout.
- `wordleml/proofeval`, `wordleml/supervised`, `wordleml/runstate`, and
  `wordleml/runmetadata` provide the supported run validation and checkpoint
  restoration path used by the offline exporter.
- `wordleml/vocabulary` and `wordleml/modelstate` remain the canonical word-ID,
  hash, and four-input encoding boundary. CUDA-specific commands use the sealed
  loader which never opens the final-test split.
- `wordleml/gameeval` remains authoritative for gameplay, availability masking,
  repeat suppression, and deterministic lower-action-ID tie breaking.
- `wordleml/serving` and `wordleml/cmd/serve` remain the GoMLX fallback. The new
  runtime must not import them because that would pull GoMLX/XLA into its binary.
- `web/internal/server/static` is the visual reference for the new one-process
  browser demo. The assets must be copied or extracted because `web` is a
  separate Go module with an `internal` package.
- `wordleml/cuda/smoke.cu`, the root `Makefile`, `docker/Dockerfile`, and
  `docker-compose.yml` define the existing one-approved-GPU, `sm_120`,
  container-only build and execution policy.

The selected source is run `seed-replication-20260809-132505Z`, selector
`best`, update 2,600, trained at commit
`2718164bb80460757592b90aa86b96eb6d596018`. Its requested checkpoint is the
latest complete pair in `checkpoints/best` and records 1,046,596 trainable FP32
parameters. The run deliberately records no final-test artifact.
