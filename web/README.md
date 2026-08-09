# Web visualiser

This Go application visualises complete games played by the trained policy. It
serves embedded HTML, CSS, and JavaScript, proxies `/api/models`,
`/api/solutions`, and `/api/games` to the internal CUDA inference service, and
retains a lightweight `/healthz` endpoint. The model selector lists completed,
compatible runs beneath `runs/`; changing it loads and warms that run's best
checkpoint on the CUDA device before subsequent games use it.

The browser receives a completed trajectory and animates it turn by turn; it
does not connect to the GPU service directly. Tests and compilation happen in
`docker/Dockerfile.web`, so no host front-end toolchain is required. From the
repository root, use `make web` and open <http://127.0.0.1:8082>.
