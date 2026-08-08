package runmetadata

import (
	"context"
	"errors"
	"fmt"
	"os/exec"
	"regexp"
	"strings"
	"time"
)

const (
	nvidiaSMICommand  = "nvidia-smi"
	nvccCommand       = "nvcc"
	defaultProbeLimit = 5 * time.Second
)

var nvccReleasePattern = regexp.MustCompile(`(?m)\brelease\s+([^,\s]+)`)

// GPUDetails is the read-only nvidia-smi description of the single GPU made
// visible to the training process. Its keys are name, uuid, driver_version,
// and compute_capability.
type GPUDetails = map[string]string

// CommandRunner executes a probe command. It is injectable so tests can check
// parsing without running CUDA tools or needing a GPU.
type CommandRunner interface {
	Run(context.Context, string, ...string) ([]byte, error)
}

// CommandRunnerFunc adapts a function to CommandRunner.
type CommandRunnerFunc func(context.Context, string, ...string) ([]byte, error)

// Run invokes the function.
func (function CommandRunnerFunc) Run(ctx context.Context, name string, args ...string) ([]byte, error) {
	return function(ctx, name, args...)
}

type osCommandRunner struct{}

func (osCommandRunner) Run(ctx context.Context, name string, args ...string) ([]byte, error) {
	return exec.CommandContext(ctx, name, args...).Output()
}

// ProbeOptions configures a bounded, read-only runtime probe. Detail maps are
// supplied by the runner after it has selected its PJRT/GoMLX environment; the
// probe never creates a GoMLX backend. A zero Timeout uses five seconds.
type ProbeOptions struct {
	Runner       CommandRunner
	Timeout      time.Duration
	PJRTDetails  map[string]string
	GoMLXDetails map[string]string
}

// RuntimeProbe is the hardware and environment evidence that a runner can
// copy into RuntimeMetadata before calling Collect.
type RuntimeProbe struct {
	GPUDetails   GPUDetails
	CUDADetails  map[string]string
	PJRTDetails  map[string]string
	GoMLXDetails map[string]string
}

// ProbeGPU queries nvidia-smi with a five-second limit and accepts only one
// project-approved Blackwell GPU: an RTX 5070 Ti or RTX 5050 with CUDA compute
// capability 12.0. In particular, it cannot silently select a visible RTX
// 3060 or any other GPU.
func ProbeGPU(ctx context.Context, runner CommandRunner) (GPUDetails, error) {
	if ctx == nil {
		return nil, errors.New("probe context must not be nil")
	}
	boundedContext, cancel := context.WithTimeout(ctx, defaultProbeLimit)
	defer cancel()
	return probeGPU(boundedContext, runner)
}

func probeGPU(ctx context.Context, runner CommandRunner) (GPUDetails, error) {
	if runner == nil {
		runner = osCommandRunner{}
	}
	output, err := runner.Run(ctx, nvidiaSMICommand,
		"--query-gpu=name,uuid,driver_version,compute_cap",
		"--format=csv,noheader,nounits",
	)
	if err != nil {
		return nil, fmt.Errorf("query visible GPU with nvidia-smi: %w", err)
	}
	return parseGPUDetails(output)
}

// ProbeRuntime collects GPU and CUDA-compiler evidence without initialising a
// PJRT or GoMLX backend. nvcc is optional: a missing compiler is recorded as
// such, while any other nvcc failure is returned to the caller.
func ProbeRuntime(ctx context.Context, options ProbeOptions) (RuntimeProbe, error) {
	if ctx == nil {
		return RuntimeProbe{}, errors.New("probe context must not be nil")
	}
	if options.Timeout < 0 {
		return RuntimeProbe{}, fmt.Errorf("probe timeout must not be negative: %s", options.Timeout)
	}
	limit := options.Timeout
	if limit == 0 {
		limit = defaultProbeLimit
	}
	runner := options.Runner
	if runner == nil {
		runner = osCommandRunner{}
	}

	gpuContext, cancelGPU := context.WithTimeout(ctx, limit)
	defer cancelGPU()
	gpu, err := probeGPU(gpuContext, runner)
	if err != nil {
		return RuntimeProbe{}, err
	}

	nvccContext, cancelNVCC := context.WithTimeout(ctx, limit)
	defer cancelNVCC()
	cuda, err := probeNVCC(nvccContext, runner)
	if err != nil {
		return RuntimeProbe{}, err
	}
	return RuntimeProbe{
		GPUDetails:   gpu,
		CUDADetails:  cuda,
		PJRTDetails:  cloneDetails(options.PJRTDetails),
		GoMLXDetails: cloneDetails(options.GoMLXDetails),
	}, nil
}

func parseGPUDetails(output []byte) (GPUDetails, error) {
	lines := strings.Split(strings.TrimSpace(string(output)), "\n")
	if len(lines) != 1 || strings.TrimSpace(lines[0]) == "" {
		return nil, fmt.Errorf("nvidia-smi reported %d visible GPUs, want exactly one", countNonEmpty(lines))
	}
	fields := strings.Split(lines[0], ",")
	if len(fields) != 4 {
		return nil, fmt.Errorf("malformed nvidia-smi GPU row %q: got %d columns, want 4", lines[0], len(fields))
	}
	for index := range fields {
		fields[index] = strings.TrimSpace(fields[index])
		if fields[index] == "" {
			return nil, fmt.Errorf("malformed nvidia-smi GPU row %q: column %d is empty", lines[0], index+1)
		}
	}
	model, approved := approvedGPUModel(fields[0])
	if !approved {
		return nil, fmt.Errorf("visible GPU %q is not an approved RTX 5070 Ti or RTX 5050", fields[0])
	}
	if fields[3] != "12.0" {
		return nil, fmt.Errorf("%s compute capability is %q, want 12.0", model, fields[3])
	}
	return GPUDetails{
		"name":               fields[0],
		"uuid":               fields[1],
		"driver_version":     fields[2],
		"compute_capability": fields[3],
	}, nil
}

// approvedGPUModel recognises the model words reported by nvidia-smi rather
// than requiring one vendor-specific display string. For example, it accepts
// both "NVIDIA GeForce RTX 5050 Laptop GPU" and "RTX-5050". It deliberately
// does not accept different RTX variants such as an RTX 3060 or a hypothetical
// 5050 Ti/SUPER.
func approvedGPUModel(name string) (string, bool) {
	tokens := gpuNameTokens(name)
	for index := 0; index+1 < len(tokens); index++ {
		if tokens[index] != "rtx" {
			continue
		}
		switch tokens[index+1] {
		case "5070":
			if index+2 < len(tokens) && tokens[index+2] == "ti" && !hasUnsupportedModelSuffix(tokens, index+3) {
				return "RTX 5070 Ti", true
			}
		case "5050":
			if hasUnsupportedModelSuffix(tokens, index+2) {
				continue
			}
			return "RTX 5050", true
		}
	}
	return "", false
}

func hasUnsupportedModelSuffix(tokens []string, index int) bool {
	return index < len(tokens) && (tokens[index] == "ti" || tokens[index] == "super")
}

func gpuNameTokens(name string) []string {
	return strings.FieldsFunc(strings.ToLower(name), func(character rune) bool {
		return !(character >= 'a' && character <= 'z' || character >= '0' && character <= '9')
	})
}

func probeNVCC(ctx context.Context, runner CommandRunner) (map[string]string, error) {
	output, err := runner.Run(ctx, nvccCommand, "--version")
	if err != nil {
		if errors.Is(err, exec.ErrNotFound) {
			return map[string]string{"nvcc": "not_found"}, nil
		}
		return nil, fmt.Errorf("query CUDA compiler with nvcc --version: %w", err)
	}
	text := strings.TrimSpace(string(output))
	if text == "" {
		return nil, errors.New("nvcc --version returned no version information")
	}
	details := map[string]string{"nvcc": "available"}
	if match := nvccReleasePattern.FindStringSubmatch(text); len(match) == 2 {
		details["nvcc_version"] = match[1]
	} else {
		details["nvcc_output"] = text
	}
	return details, nil
}

func countNonEmpty(lines []string) int {
	count := 0
	for _, line := range lines {
		if strings.TrimSpace(line) != "" {
			count++
		}
	}
	return count
}
