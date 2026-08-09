# Model-facing Wordle state

The shared `wordleml/modelstate` package is the sole host-side encoder for
training records, retained GoMLX serving, and the CUDA/cgo demo. It accepts a
289-byte, LSB-first bitset over the canonical solution IDs plus a turn from
zero through five. It rejects empty sets and non-zero padding bits, then
produces the four values consumed by `policy.Model.Forward` and the
hand-written CUDA forward pass:

- FP32 `CandidateMask[2309]`;
- FP32 `CandidateStats[209]`;
- int32 `Turn`;
- FP32 `RemainingActionMask[4739]`.

`wordleml/vocabulary.Load(dataDir)` loads the five fixed files in `data/`, preserves their file order as the canonical
IDs, validates the 4,739/2,309 vocabulary dimensions and the disjoint 2,109/100/100 split, and maps every solution ID
to its action ID. `Vocabulary.Hashes()` exposes SHA-256 hashes over normalized uppercase words, one newline per word, for
dataset metadata checks.

`candidateMask` identifies which of the 2,309 solution words remain compatible with all feedback. The model divides this
mask by the number of candidates before its first linear layer. Each of that layer's 96 outputs is therefore a learned
mean over the remaining solution set, rather than a sum whose scale grows with the candidate count.

`candidateStats` supplies 209 explicit aggregate features:

- 130 frequencies for each of 26 letters at each of five positions;
- 78 frequencies for each letter occurring at least one, two, or three times in a candidate word;
- one normalized logarithmic candidate-count feature.

The first 130 values are position-major (`position * 26 + letter`). The next 78 are letter-major
(`130 + letter * 3 + threshold - 1`) for thresholds one, two, and three. Each is a fraction of the remaining candidates;
the final value is `log(candidateCount) / log(2309)`.

These statistics retain useful letter and multiplicity information after the much larger candidate mask is compressed.
The candidate count is included explicitly because normalizing `candidateMask` removes its magnitude.

`turn` is an integer from zero through five and receives a learned 16-value embedding. It lets the same candidate state
lead to a different policy early or late in the six-guess game.

`remainingActionMask` is indexed by the 4,739-word action vocabulary. An entry is one only when that action word is among
the remaining candidate solutions. It does not describe legality: the model adds `beta * remainingActionMask` to its
ordinary action logits, so words used as information-gathering probes keep their base logits and remain selectable.
Go retains a separate available-action mask for history-based repeat
suppression and deterministic legal selection after either inference backend
returns raw logits.
