# Inference serving

The local visualiser can select the best checkpoint from any completed, passed
run beneath `runs/`. The neural-network forward pass runs through GoMLX on the
configured CUDA device; HTTP, input validation, state encoding, legal-action
selection, and the authoritative Wordle engine remain in Go on the host. A
game therefore makes up to six small GPU inference calls as its board state
changes.

## Runtime and API

`cmd/serve` validates each run's data hashes, effective configuration,
CUDA/PJRT identity, checkpoint step, best-validation state, and materialized
parameter count. It records the original training commit in every response but
permits later compatible serving code, rather than requiring the repository
`HEAD` to equal that training commit. The configured initial checkpoint is
loaded and warmed at startup. When another run is selected, its best checkpoint
is restored and warmed on CUDA before it atomically becomes active. A failed
load leaves the prior model active. Existing games can finish on the old model,
and that runtime is released only after the swap. Requests against each GoMLX
session remain serialized because its executor and Store are not safe for
concurrent use.

The internal service exposes:

- `GET /healthz`: readiness and model identity;
- `GET /v1/models`: the active model and completed compatible runs discovered
  beneath `runs/`;
- `PUT /v1/models`: load and activate the requested best checkpoint for
  `{"run_id":"production-20260809-005026Z"}`;
- `GET /v1/solutions`: the 100 allowed validation solutions;
- `POST /v1/games`: one complete game for `{"solution":"VODKA"}`.

The game response contains the run/checkpoint/update identity, solved status,
guess count, and every accepted guess, feedback pattern, and shortlist-size
transition. Invalid or final-test solutions are rejected. Model selection can
only name a run returned by `GET /v1/models`; there are no training or
arbitrary-checkpoint endpoints.

## Browser flow

Compose keeps `inference:8090` private. The unprivileged web container proxies
`/api/models`, `/api/solutions`, and `/api/games`, so browser requests remain
same-origin and the GPU endpoint is never published. Selecting a model waits
for its on-device load and warm-up. A game REST response contains the completed
trajectory; JavaScript animates those turns locally rather than holding a
streaming connection.

Set the initially selected completed run in `.env` and start the stack:

```text
WORDLEML_INFERENCE_RUN_ID=proof-full-20260808
```

```console
make monitoring
```

Open <http://127.0.0.1:8082>. For an API-only process use `make inference`.
The model picker re-scans `runs/`, so a newly completed compatible run appears
without restarting the inference service. The final-test split remains sealed:
only the fixed validation list is exposed or accepted.
