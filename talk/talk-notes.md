# Talk notes

The talk is one hour for Go developers who may have little machine-learning experience. It must build the ML concepts
gradually and keep the Wordle problem visible throughout, rather than turning into a tour of infrastructure.

## Reproducible starting point

- Begin demonstrations from the same containerised environment used during development.
- Show that GPU selection is a real engineering concern: the development desktop has two NVIDIA cards, but Compose
  passes only the RTX 5070 Ti by UUID.
- Use the small CUDA smoke kernel to make `sm_120`, compute capability 12.0, host/device memory, and kernel launch syntax
  concrete before introducing model code.
- Follow it with the GoMLX Euclidean-distance graph to introduce symbolic graphs, XLA compilation, and the CUDA backend
  without pretending that the Wordle model already exists.
- Mention TensorBoard and the visualiser only briefly at this stage. Their deliberately empty states make a useful
  before-and-after contrast once training metrics and gameplay data are implemented.

## Later structure

[To be expanded alongside the model: Wordle data and encoding, first baseline, loss and optimisation, Go/CUDA boundary,
training results, failures and lessons, live visualisation, and conclusion.]
