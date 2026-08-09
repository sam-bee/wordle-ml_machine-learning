# From Go to the GPU: Integrating with CUDA

Go is an appealing choice for machine learning projects, with a fast, efficient systems programming approach and a mature
concurrency model. Integrating Go with CUDA for GPU programming allows developers to connect conventional software with
bespoke AI/ML tooling.

In this talk, a real example project will be used to explore that journey. Join us as we solve word puzzles with Go
code, train and validate a compact policy with Go libraries, export one fixed model, and connect it to a hand-written
CUDA inference implementation via cgo.

Using a concrete case study, the talk will show how Go can handle orchestration, data generation, evaluation, and the
authoritative Wordle rules while CUDA runs the numerical forward pass. We will look
closely at the components and architecture of a custom-built ML model, and the engineering boundary between on-host Go
code and on-device GPU workloads in CUDA.

Along the way, the talk will cover practical lessons from building a mixed Go/CUDA project from scratch - including code
structure, profiling with NVIDIA’s tools, and the role that Go libraries in the machine learning space can play as the
project evolves.

This isn't a how-to guide on running a popular Python library or using the latest cloud tools. But if you want to look
at the nuts and bolts of how Go and CUDA combine to deploy a custom neural-net forward pass, measure it honestly, and
keep the surrounding application understandable, join us as we move from Go to the GPU.
