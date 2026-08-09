package main

import (
	"bytes"
	"path/filepath"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/cudamodel"
	"github.com/sam-bee/wordle-ml_machine-learning/cudaref"
)

func TestParseConfig(t *testing.T) {
	var stderr bytes.Buffer
	got, err := parseConfig([]string{
		"-run-id=seed-replication-20260809-132505Z",
		"-data-dir=/fixture/data",
		"-runs-dir=/fixture/runs",
	}, &stderr)
	if err != nil {
		t.Fatalf("parseConfig: %v", err)
	}
	if got.checkpoint != defaultCheckpoint {
		t.Fatalf("checkpoint = %q, want %q", got.checkpoint, defaultCheckpoint)
	}
	wantOutput := filepath.Join("/fixture/runs", "seed-replication-20260809-132505Z", "exports", exportDirectoryFormat, defaultCheckpoint)
	if got.output != wantOutput {
		t.Fatalf("output = %q, want %q", got.output, wantOutput)
	}
}

func TestParseConfigRejectsUnsupportedCheckpoint(t *testing.T) {
	_, err := parseConfig([]string{"-run-id=run-1", "-checkpoint=latest"}, &bytes.Buffer{})
	if err == nil {
		t.Fatal("parseConfig accepted latest checkpoint")
	}
}

func TestValidateSourceShapeReversesDenseTensors(t *testing.T) {
	var dense cudamodel.Tensor
	for _, tensor := range cudamodel.ExpectedTensors() {
		if len(tensor.Shape) == 2 {
			dense = tensor
			break
		}
	}
	if dense.Name == "" {
		t.Fatal("expected at least one dense tensor")
	}
	if err := validateSourceShape([]int{dense.Shape[1], dense.Shape[0]}, dense); err != nil {
		t.Fatalf("validate source dense shape: %v", err)
	}
	if err := validateSourceShape(dense.Shape, dense); err == nil {
		t.Fatal("validateSourceShape accepted output-major dense source")
	}
}

func TestValidateSourceShapeKeepsTurnEmbeddingOrientation(t *testing.T) {
	var embedding cudamodel.Tensor
	for _, tensor := range cudamodel.ExpectedTensors() {
		if tensor.Name == cudamodel.TurnEmbedding {
			embedding = tensor
			break
		}
	}
	if embedding.Name == "" {
		t.Fatal("turn embedding tensor is missing")
	}
	if denseSourceIsInputMajor(embedding) {
		t.Fatal("turn embedding was classified as a dense kernel")
	}
	if err := validateSourceShape(embedding.Shape, embedding); err != nil {
		t.Fatalf("validate turn embedding source shape: %v", err)
	}
	if err := validateSourceShape([]int{embedding.Shape[1], embedding.Shape[0]}, embedding); err == nil {
		t.Fatal("validateSourceShape accepted transposed turn embedding source")
	}
}

func TestChooseRepresentativesCoversRequiredGameplay(t *testing.T) {
	vectors := make([]cudaref.Vector, 0, 32)
	appendVector := func(turn, candidates int, selectedCandidate bool) {
		vectors = append(vectors, cudaref.Vector{Provenance: cudaref.Provenance{
			Turn:                 turn,
			CandidateCount:       candidates,
			SelectedWasCandidate: selectedCandidate,
		}})
	}
	appendVector(0, 2309, true) // opening
	for turn := 0; turn < cudamodel.NumTurns; turn++ {
		appendVector(turn, 10, true)
	}
	appendVector(0, 2309, true) // large after opening/turn coverage
	appendVector(0, 10, true)   // medium after turn coverage
	appendVector(0, 3, true)    // small after turn coverage
	appendVector(0, 10, true)   // selected candidate after prior picks
	appendVector(0, 3, false)   // selected probe after prior picks
	for len(vectors) < cap(vectors) {
		appendVector(len(vectors)%cudamodel.NumTurns, 10, true)
	}
	got, err := chooseRepresentatives(vectors, 20)
	if err != nil {
		t.Fatalf("chooseRepresentatives: %v", err)
	}
	if len(got) != 20 {
		t.Fatalf("representative count = %d, want 20", len(got))
	}
	seenTurns := make(map[int]bool)
	seenLarge, seenMedium, seenSmall, seenCandidate, seenProbe := false, false, false, false, false
	for _, vector := range got {
		provenance := vector.Provenance
		seenTurns[provenance.Turn] = true
		switch {
		case provenance.CandidateCount > 100:
			seenLarge = true
		case provenance.CandidateCount >= 6:
			seenMedium = true
		default:
			seenSmall = true
		}
		seenCandidate = seenCandidate || provenance.SelectedWasCandidate
		seenProbe = seenProbe || !provenance.SelectedWasCandidate
	}
	for turn := 0; turn < cudamodel.NumTurns; turn++ {
		if !seenTurns[turn] {
			t.Errorf("missing turn %d", turn)
		}
	}
	if !seenLarge || !seenMedium || !seenSmall || !seenCandidate || !seenProbe {
		t.Fatalf("coverage large=%t medium=%t small=%t candidate=%t probe=%t", seenLarge, seenMedium, seenSmall, seenCandidate, seenProbe)
	}
}
