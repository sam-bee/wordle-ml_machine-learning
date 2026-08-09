package cudaref

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

func TestGoldenVectorsRoundTrip(t *testing.T) {
	dir := t.TempDir()
	vectors := []Vector{testVector("opening", 0), testVector("late", 5)}
	manifest, err := WriteGoldenVectors(dir, vectors)
	if err != nil {
		t.Fatalf("WriteGoldenVectors: %v", err)
	}
	if manifest.Format != GoldenFormat || manifest.ValuesCount == 0 {
		t.Fatalf("manifest = %+v", manifest)
	}
	set, err := LoadGoldenVectors(dir)
	if err != nil {
		t.Fatalf("LoadGoldenVectors: %v", err)
	}
	if len(set.Vectors) != len(vectors) {
		t.Fatalf("loaded %d vectors, want %d", len(set.Vectors), len(vectors))
	}
	for index, want := range vectors {
		got := set.Vectors[index]
		if got.ID != want.ID || got.Inputs.Turn != want.Inputs.Turn || got.RawTopActionID != want.RawTopActionID || got.SelectedActionID != want.SelectedActionID || got.TopTwoMargin != want.TopTwoMargin {
			t.Fatalf("vector %d metadata = %+v, want %+v", index, got, want)
		}
		if got.Inputs.CandidateMask[0] != want.Inputs.CandidateMask[0] || got.Inputs.CandidateStats[12] != want.Inputs.CandidateStats[12] || got.Inputs.RemainingActionMask[19] != want.Inputs.RemainingActionMask[19] || got.AvailableActionMask[20] != want.AvailableActionMask[20] || got.RawLogits[21] != want.RawLogits[21] {
			t.Fatalf("vector %d payload did not round trip", index)
		}
	}
}

func TestLoadGoldenVectorsRejectsWeightPayloadTamper(t *testing.T) {
	dir := t.TempDir()
	if _, err := WriteGoldenVectors(dir, []Vector{testVector("opening", 0)}); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(dir, GoldenValuesFilename)
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	contents[0] ^= 0xff
	if err := os.WriteFile(path, contents, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := LoadGoldenVectors(dir); err == nil {
		t.Fatal("tampered values payload was accepted")
	}
}

func TestWriteGoldenVectorsRejectsInvalidInput(t *testing.T) {
	vector := testVector("opening", 0)
	vector.Inputs.CandidateMask = vector.Inputs.CandidateMask[:len(vector.Inputs.CandidateMask)-1]
	if _, err := WriteGoldenVectors(t.TempDir(), []Vector{vector}); err == nil {
		t.Fatal("short candidate mask was accepted")
	}
	vector = testVector("opening", 0)
	vector.RawLogits[0] = float32(math.NaN())
	if _, err := WriteGoldenVectors(t.TempDir(), []Vector{vector}); err == nil {
		t.Fatal("non-finite logit was accepted")
	}
	vector = testVector("opening", 0)
	vector.SelectedActionID = 19
	if _, err := WriteGoldenVectors(t.TempDir(), []Vector{vector}); err == nil {
		t.Fatal("inconsistent selected action was accepted")
	}
}

func testVector(id string, turn int32) Vector {
	inputs := modelstate.Inputs{
		CandidateMask:       make([]float32, vocabulary.NumSolutions),
		CandidateStats:      make([]float32, modelstate.CandidateStatsSize),
		Turn:                turn,
		RemainingActionMask: make([]float32, vocabulary.NumActions),
	}
	inputs.CandidateMask[0] = 1
	inputs.CandidateStats[12] = .25
	inputs.RemainingActionMask[19] = 1
	available := make([]float32, vocabulary.NumActions)
	available[20] = 1
	logits := make([]float32, vocabulary.NumActions)
	logits[21] = 7.5
	return Vector{
		ID: id, Inputs: inputs, AvailableActionMask: available, RawLogits: logits,
		RawTopActionID: 21, SelectedActionID: 20, TopTwoMargin: 7.5,
		Provenance: Provenance{Solution: "ADEPT", Turn: int(turn), CandidateCount: 1, SelectionKind: "probe", ShortlistSizeBefore: 1, ShortlistSizeAfter: 1},
	}
}
