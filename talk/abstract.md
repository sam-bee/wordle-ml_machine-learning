# From Go to the GPU: Integrating with CUDA

Go is an appealing choice machine learning projects, with a fast, efficient systems programming approach and a mature
concurrency model. Integrating Go with CUDA for GPU programming allows developers to connect conventional software with
bespoke AI/ML tooling.

In this talk, a real example project will be used to explore that journey. Join us as we solve word puzzles with Go
code, design a machine learning model from scratch in CUDA, and connect the two via cgo to train our model with
synthetic data.

Using a concrete case study, the talk will show how Go can handle orchestration, data generation, and evaluation -
calling CUDA code inline to take over the performance-critical work of running and training our model. We will look
closely at the components and architecture of a custom-built ML model, and the engineering boundary between on-host Go
code and on-device GPU workloads in CUDA.

Along the way, the talk will cover practical lessons from building a mixed Go/CUDA project from scratch - including code
structure, profiling with NVIDIA’s tools, and the role that Go libraries in the machine learning space can play as the
project evolves.

This isn't a how-to guide on running a popular Python library or using the latest cloud tools. But if you want to look
at the nuts and bolts of how Go and CUDA combine to build and train a custom neural net from scratch, join us as we move
from Go to the GPU.
