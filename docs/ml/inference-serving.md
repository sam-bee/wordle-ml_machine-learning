# Inference serving

The local visualiser serves one immutable, passed full-run best checkpoint. The
neural-network forward pass runs through GoMLX on the configured CUDA device;
HTTP, input validation, state encoding, legal-action selection, and the
authoritative Wordle engine remain in Go on the host. A game therefore makes up
to six small GPU inference calls as its board state changes.

## Runtime and API

`cmd/serve` validates the run's data hashes, effective configuration, CUDA/PJRT
identity, checkpoint step, best-validation state, and materialized parameter
count. It records the original training commit in every response but permits
later compatible serving code, rather than requiring the repository `HEAD` to
equal that training commit. The checkpoint is loaded and warmed once at
startup. Requests are serialized because a GoMLX session and Store are not
safe for concurrent use.

The internal service exposes:

- `GET /healthz`: readiness and model identity;
- `GET /v1/solutions`: the 100 allowed validation solutions;
- `POST /v1/games`: one complete game for `{"solution":"VODKA"}`.

The game response contains the run/checkpoint/update identity, solved status,
guess count, and every accepted guess, feedback pattern, and shortlist-size
transition. Invalid or final-test solutions are rejected. There are no
training or model-replacement endpoints.

## Browser flow

Compose keeps `inference:8090` private. The unprivileged web container proxies
`/api/solutions` and `/api/games`, so browser requests remain same-origin and
the GPU endpoint is never published. The REST response contains the completed
game; JavaScript animates those turns locally rather than holding a streaming
connection.

Set the passed full run in `.env` and start the stack:

```text
WORDLEML_INFERENCE_RUN_ID=proof-full-20260808
```

```console
make monitoring
```

Open <http://127.0.0.1:8082>. For an API-only process use `make inference`.
The final-test split remains sealed: only the fixed validation list is exposed
or accepted.
