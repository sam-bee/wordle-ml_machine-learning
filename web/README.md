# Web visualiser

This Go application will visualise training and gameplay data. It currently serves a static splash page and a
`/healthz` endpoint.

The HTML and CSS are embedded into the Go binary. Tests and compilation happen in `docker/Dockerfile.web`; no host
front-end toolchain is required. From the repository root, use `make web` and open <http://127.0.0.1:8082>.
