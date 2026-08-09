# Policy model structure

The GoMLX model in [`wordleml/policy`](../../wordleml/policy/) assigns one raw logit to every word in the fixed action
vocabulary. It contains the policy architecture only: state generation, training, loss calculation, action selection,
and gameplay are separate work. The hand-written CUDA/cgo backend implements
this exact fixed FP32 graph from an exported artifact; see
[CUDA/cgo inference](cuda-cgo-inference.md).

The current data contains 2,309 possible solutions (`S`) and 4,739 actions (`A`). These values are passed in through
`policy.Config`; the model does not hard-code either vocabulary size. Wordle's fixed five-letter words, six turns, and
209 candidate statistics are intentionally fixed in the architecture.

## Inputs and output

All floating-point inputs and weights are FP32. The batch dimension may vary between compiled GoMLX graphs.

| Input | Shape | Meaning |
| --- | --- | --- |
| `candidateMask` | `[batch, S]` | One for each solution compatible with the board, otherwise zero. Each row is divided by its sum before projection, making the linear projection a mean over remaining candidates. A valid state therefore needs at least one candidate. |
| `candidateStats` | `[batch, 209]` | Precomputed summary of the remaining candidates: 130 positional letter frequencies, 78 letter-multiplicity frequencies, and one normalized log candidate-count feature. |
| `turn` | `[batch]` | Integer turn index from zero through five. |
| `remainingActionMask` | `[batch, A]` | One where an action is also a remaining solution. It controls a learned bonus only; it is not a legality mask. |

The output is raw FP32 logits with shape `[batch, A]`. Softmax belongs with the future loss or action-selection code,
not in the model.

## Network

```text
candidateMask / row sum -> Linear(S, 96) -> ReLU --+
candidateStats          -> Linear(209, 48) -> ReLU -+-> concatenate -> h [batch, 160]
turn                    -> Embedding(6, 16) --------+

r = ReLU(Linear(160, 160)(h))
r = Linear(160, 160)(r)
h = ReLU(h + r)

baseLogits = Linear(160, A)(h)
beta       = Linear(160, 1)(h)
logits     = baseLogits + beta * remainingActionMask
```

The scalar `beta` is learned separately for every position in the batch. It can raise or lower all actions which are
still candidate solutions while leaving other action logits unchanged. In particular, a zero in
`remainingActionMask` never becomes negative infinity: non-candidate probe words remain playable. There is no hard-mode
filter in this model.

The residual block works without a projection on its skip path because the three feature branches total exactly 160
features (`96 + 48 + 16`). The implementation deliberately has no attention, normalization, dropout, action
embeddings, value head, or additional hidden layers.

## Parameter count

Every linear layer includes a bias. The trainable count is:

| Component | Parameters |
| --- | ---: |
| Candidate projection | `96S + 96` |
| Statistics projection | `209 * 48 + 48 = 10,080` |
| Turn embedding | `6 * 16 = 96` |
| Two residual linear layers | `2 * (160 * 160 + 160) = 51,520` |
| Base policy logits | `160A + A = 161A` |
| Candidate bonus | `160 + 1 = 161` |
| **Total** | **`96S + 161A + 61,953`** |

For the repository vocabularies, `S=2,309` and `A=4,739`, so the model has exactly **1,046,596 trainable parameters**.
Its FP32 weights occupy 4,186,384 bytes: about 4.19 MB, or 3.99 MiB. The architecture brief's reference dimensions of
`S=2,315` and `A=4,800` produce exactly 1,056,993 parameters.

The tests build and run the real GoMLX graph on the configured backend, then count the materialized trainable variables.
This deliberately excludes GoMLX's internal, non-trainable random-number-generator state. Both vocabulary configurations
above are asserted so changes to layer widths, biases, or added heads make the architectural drift visible.

## GoMLX implementation details

`Model.Forward` uses GoMLX's scoped `DenseWithBias` and `Embedding` layers and returns a graph node directly. The turn
vector is explicitly expanded from `[batch]` to `[batch, 1]` before embedding lookup; this avoids GoMLX interpreting a
single-example `[1]` input as one index tuple and dropping the batch axis. Variables use GoMLX's default initializer,
with zero-initialized biases, and the stable scopes give checkpoints readable layer names.
