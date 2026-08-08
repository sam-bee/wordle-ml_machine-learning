# Model-facing Wordle state

The policy does not consume the coloured board or raw letters directly. A future data/state component will translate a
board into four tensors; the current model treats all four as precomputed inputs and does not implement that translation.

`candidateMask` identifies which of the 2,309 solution words remain compatible with all feedback. The model divides this
mask by the number of candidates before its first linear layer. Each of that layer's 96 outputs is therefore a learned
mean over the remaining solution set, rather than a sum whose scale grows with the candidate count.

`candidateStats` supplies 209 explicit aggregate features:

- 130 frequencies for each of 26 letters at each of five positions;
- 78 frequencies for each letter occurring at least one, two, or three times in a candidate word;
- one normalized logarithmic candidate-count feature.

These statistics retain useful letter and multiplicity information after the much larger candidate mask is compressed.
The candidate count is included explicitly because normalizing `candidateMask` removes its magnitude.

`turn` is an integer from zero through five and receives a learned 16-value embedding. It lets the same candidate state
lead to a different policy early or late in the six-guess game.

`remainingActionMask` is indexed by the 4,739-word action vocabulary. An entry is one only when that action word is among
the remaining candidate solutions. It does not describe legality: the model adds `beta * remainingActionMask` to its
ordinary action logits, so words used as information-gathering probes keep their base logits and remain selectable.
