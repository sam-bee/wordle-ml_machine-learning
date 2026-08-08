package runmetadata

import (
	"context"
	"errors"
	"os/exec"
	"reflect"
	"strings"
	"testing"
	"time"
)

const (
	approved5070GPU = "NVIDIA GeForce RTX 5070 Ti, GPU-5070, 580.65.06, 12.0\n"
	approved5050GPU = "NVIDIA GeForce RTX-5050 Laptop GPU, GPU-5050, 580.65.06, 12.0\n"
)

func TestProbeRuntimeRecordsApprovedGPUAndCUDAEvidence(t *testing.T) {
	var calls []commandCall
	runner := CommandRunnerFunc(func(ctx context.Context, name string, arguments ...string) ([]byte, error) {
		if _, ok := ctx.Deadline(); !ok {
			t.Fatal("probe command has no deadline")
		}
		calls = append(calls, commandCall{name: name, arguments: append([]string(nil), arguments...)})
		switch name {
		case nvidiaSMICommand:
			return []byte(approved5070GPU), nil
		case nvccCommand:
			return []byte("Cuda compilation tools, release 13.1, V13.1.80\n"), nil
		default:
			return nil, errors.New("unexpected command")
		}
	})
	pjrt := map[string]string{"plugin": "cuda", "xla_flags": "--xla_gpu_enable_triton_gemm=false"}
	gomlx := map[string]string{"compute_backend": "xla:cuda"}
	probe, err := ProbeRuntime(context.Background(), ProbeOptions{
		Runner:       runner,
		Timeout:      time.Second,
		PJRTDetails:  pjrt,
		GoMLXDetails: gomlx,
	})
	if err != nil {
		t.Fatalf("ProbeRuntime: %v", err)
	}
	if got, want := probe.GPUDetails, (GPUDetails{
		"name": "NVIDIA GeForce RTX 5070 Ti", "uuid": "GPU-5070", "driver_version": "580.65.06", "compute_capability": "12.0",
	}); !reflect.DeepEqual(got, want) {
		t.Fatalf("GPU details = %#v, want %#v", got, want)
	}
	if got, want := probe.CUDADetails, map[string]string{"nvcc": "available", "nvcc_version": "13.1"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("CUDA details = %#v, want %#v", got, want)
	}
	if !reflect.DeepEqual(probe.PJRTDetails, pjrt) || !reflect.DeepEqual(probe.GoMLXDetails, gomlx) {
		t.Fatalf("caller environment details were not preserved: %#v %#v", probe.PJRTDetails, probe.GoMLXDetails)
	}
	pjrt["plugin"] = "changed"
	if probe.PJRTDetails["plugin"] != "cuda" {
		t.Fatal("probe retained caller map instead of copying it")
	}
	if got, want := calls, ([]commandCall{
		{name: nvidiaSMICommand, arguments: []string{"--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv,noheader,nounits"}},
		{name: nvccCommand, arguments: []string{"--version"}},
	}); !reflect.DeepEqual(got, want) {
		t.Fatalf("commands = %#v, want %#v", got, want)
	}
}

func TestProbeGPUAcceptsRTX5050(t *testing.T) {
	runner := staticRunner{responses: map[string]commandResponse{
		nvidiaSMICommand: {output: approved5050GPU},
	}}
	got, err := ProbeGPU(context.Background(), runner)
	if err != nil {
		t.Fatalf("ProbeGPU RTX 5050: %v", err)
	}
	if got["name"] != "NVIDIA GeForce RTX-5050 Laptop GPU" || got["compute_capability"] != "12.0" {
		t.Fatalf("RTX 5050 details = %#v", got)
	}
}

func TestProbeGPURejectsMultipleVisibleDevices(t *testing.T) {
	runner := staticRunner{responses: map[string]commandResponse{
		nvidiaSMICommand: {output: approved5070GPU + "NVIDIA GeForce RTX 5050, GPU-other, 580.65.06, 12.0\n"},
	}}
	_, err := ProbeGPU(context.Background(), runner)
	if err == nil || !strings.Contains(err.Error(), "exactly one") {
		t.Fatalf("ProbeGPU multiple GPUs error = %v", err)
	}
}

func TestProbeGPURejectsRTX3060(t *testing.T) {
	runner := staticRunner{responses: map[string]commandResponse{
		nvidiaSMICommand: {output: "NVIDIA GeForce RTX 3060, GPU-3060, 580.65.06, 12.0\n"},
	}}
	_, err := ProbeGPU(context.Background(), runner)
	if err == nil || !strings.Contains(err.Error(), "approved RTX 5070 Ti or RTX 5050") {
		t.Fatalf("ProbeGPU RTX 3060 error = %v", err)
	}
}

func TestProbeGPURejectsAnyOtherGPUModel(t *testing.T) {
	runner := staticRunner{responses: map[string]commandResponse{
		nvidiaSMICommand: {output: "NVIDIA GeForce RTX 5080, GPU-5080, 580.65.06, 12.0\n"},
	}}
	_, err := ProbeGPU(context.Background(), runner)
	if err == nil || !strings.Contains(err.Error(), "approved RTX 5070 Ti or RTX 5050") {
		t.Fatalf("ProbeGPU RTX 5080 error = %v", err)
	}
}

func TestProbeGPURejectsMalformedOutputAndWrongCapability(t *testing.T) {
	for name, output := range map[string]string{
		"missing column":   "NVIDIA GeForce RTX 5070 Ti, GPU-5070, 580.65.06\n",
		"empty UUID":       "NVIDIA GeForce RTX 5070 Ti, , 580.65.06, 12.0\n",
		"wrong capability": "NVIDIA GeForce RTX 5070 Ti, GPU-5070, 580.65.06, 8.6\n",
	} {
		t.Run(name, func(t *testing.T) {
			runner := staticRunner{responses: map[string]commandResponse{nvidiaSMICommand: {output: output}}}
			if _, err := ProbeGPU(context.Background(), runner); err == nil {
				t.Fatal("ProbeGPU accepted malformed nvidia-smi output")
			}
		})
	}
}

func TestProbeRuntimeRecordsMissingNVCC(t *testing.T) {
	runner := staticRunner{responses: map[string]commandResponse{
		nvidiaSMICommand: {output: approved5070GPU},
		nvccCommand:      {err: exec.ErrNotFound},
	}}
	probe, err := ProbeRuntime(context.Background(), ProbeOptions{Runner: runner})
	if err != nil {
		t.Fatalf("ProbeRuntime missing nvcc: %v", err)
	}
	if got, want := probe.CUDADetails, map[string]string{"nvcc": "not_found"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("CUDA details = %#v, want %#v", got, want)
	}
}

type commandCall struct {
	name      string
	arguments []string
}

type commandResponse struct {
	output string
	err    error
}

type staticRunner struct {
	responses map[string]commandResponse
}

func (runner staticRunner) Run(_ context.Context, name string, _ ...string) ([]byte, error) {
	response, found := runner.responses[name]
	if !found {
		return nil, errors.New("unexpected command " + name)
	}
	return []byte(response.output), response.err
}
