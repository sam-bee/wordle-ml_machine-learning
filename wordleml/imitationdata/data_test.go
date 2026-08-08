package imitationdata

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/policy"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

func TestLoadMiniMetadataContract(t *testing.T) {
	data := loadMini(t)
	if data.Split() != Mini || data.Len() != 1600 {
		t.Fatalf("loaded split=%s records=%d, want mini/1600", data.Split(), data.Len())
	}
	header := data.Header()
	metadata := data.Metadata()
	if header.Version != 3 || header.RecordSize != 437 || metadata.BinaryFile != "wordle-mini.bin" {
		t.Fatalf("unexpected mini metadata: header=%+v metadata=%+v", header, metadata)
	}
	if metadata.IncludesOpeningState {
		t.Fatal("mini must not contain the train opening record")
	}
}

func TestExampleSeparatesAvailabilityFromCandidateBonusAndUsesTeacherLabel(t *testing.T) {
	data := loadMini(t)
	var example Example
	found := false
	for index := 0; index < data.Len(); index++ {
		candidate, err := data.Example(index)
		if err != nil {
			t.Fatal(err)
		}
		if candidate.Record.TurnDepth != 0 {
			example, found = candidate, true
			break
		}
	}
	if !found {
		t.Fatal("mini has no record with history")
	}
	if example.TeacherTopAction != int32(example.Record.TopKActionIDs[0]) {
		t.Fatalf("teacher label=%d, want top action=%d", example.TeacherTopAction, example.Record.TopKActionIDs[0])
	}
	for i := 0; i < int(example.Record.TurnDepth); i++ {
		if got := example.AvailableActionMask[example.Record.History[i].GuessID]; got != 0 {
			t.Fatalf("history action %d has availability %v, want 0", example.Record.History[i].GuessID, got)
		}
	}
	for actionID, available := range example.AvailableActionMask {
		wasUsed := false
		for i := 0; i < int(example.Record.TurnDepth); i++ {
			wasUsed = wasUsed || actionID == int(example.Record.History[i].GuessID)
		}
		want := float32(1)
		if wasUsed {
			want = 0
		}
		if available != want {
			t.Fatalf("availability[%d]=%v, want %v", actionID, available, want)
		}
	}
	different := false
	for actionID := range example.AvailableActionMask {
		if example.AvailableActionMask[actionID] != example.RemainingActionMask[actionID] {
			different = true
			break
		}
	}
	if !different {
		t.Fatal("availability mask unexpectedly equals the candidate-bonus mask")
	}
}

func TestIndexOrderIsDeterministic(t *testing.T) {
	data := loadMini(t)
	first := data.IndexOrder(17)
	if !reflect.DeepEqual(first, data.IndexOrder(17)) {
		t.Fatal("same shuffle seed produced a different order")
	}
	if reflect.DeepEqual(first, data.IndexOrder(18)) {
		t.Fatal("different shuffle seeds produced the same order")
	}
	seen := make([]bool, data.Len())
	for _, index := range first {
		if index < 0 || index >= data.Len() || seen[index] {
			t.Fatalf("invalid shuffled index %d", index)
		}
		seen[index] = true
	}
}

func TestUnbatchedDatasetShapesAndDTypes(t *testing.T) {
	data := loadMini(t)
	for batch, err := range data.Dataset(1).Iter() {
		if err != nil {
			t.Fatal(err)
		}
		defer batch.Finalize()
		wantInputs := []struct {
			dtype dtypes.DType
			dims  []int
		}{
			{dtypes.Float32, []int{vocabulary.NumSolutions}},
			{dtypes.Float32, []int{modelstate.CandidateStatsSize}},
			{dtypes.Int32, nil},
			{dtypes.Float32, []int{vocabulary.NumActions}},
			{dtypes.Float32, []int{vocabulary.NumActions}},
		}
		for i, want := range wantInputs {
			if err := batch.Inputs[i].Shape().Check(want.dtype, want.dims...); err != nil {
				t.Fatalf("input %d: %v", i, err)
			}
		}
		if err := batch.Labels[0].Shape().Check(dtypes.Int32, 1); err != nil {
			t.Fatalf("label: %v", err)
		}
		return
	}
	t.Fatal("unbatched dataset produced no batch")
}

func TestBatchedMiniShapesDTypesAndFiniteValues(t *testing.T) {
	data := loadMini(t)
	backend := compute.MustNew()
	batched := Batch(backend, data.Dataset(9), 3, true)
	for batch, err := range batched.Iter() {
		if err != nil {
			t.Fatal(err)
		}
		defer batch.Finalize()
		wantInputs := []struct {
			dtype dtypes.DType
			dims  []int
		}{
			{dtypes.Float32, []int{3, vocabulary.NumSolutions}},
			{dtypes.Float32, []int{3, modelstate.CandidateStatsSize}},
			{dtypes.Int32, []int{3}},
			{dtypes.Float32, []int{3, vocabulary.NumActions}},
			{dtypes.Float32, []int{3, vocabulary.NumActions}},
		}
		if len(batch.Inputs) != len(wantInputs) || len(batch.Labels) != 1 {
			t.Fatalf("inputs=%d labels=%d", len(batch.Inputs), len(batch.Labels))
		}
		for i, want := range wantInputs {
			if err := batch.Inputs[i].Shape().Check(want.dtype, want.dims...); err != nil {
				t.Fatalf("input %d: %v", i, err)
			}
			if want.dtype == dtypes.Float32 {
				for _, row := range batch.Inputs[i].Value().([][]float32) {
					for _, value := range row {
						if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
							t.Fatalf("input %d has non-finite value %v", i, value)
						}
					}
				}
			}
		}
		if err := batch.Labels[0].Shape().Check(dtypes.Int32, 3, 1); err != nil {
			t.Fatalf("label: %v", err)
		}
		return
	}
	t.Fatal("batched dataset produced no batch")
}

func TestRejectsMismatchedMetadataHashAndSplit(t *testing.T) {
	v := loadVocabulary(t)
	for _, mutation := range []struct {
		name string
		edit func(*map[string]any)
	}{
		{"hash", func(m *map[string]any) {
			(*m)["action_vocabulary_sha256"] = "0000000000000000000000000000000000000000000000000000000000000000"
		}},
		{"split", func(m *map[string]any) { (*m)["split"] = "validation" }},
	} {
		t.Run(mutation.name, func(t *testing.T) {
			directory := copyMini(t)
			metadataPath := filepath.Join(directory, "wordle-mini.json")
			contents, err := os.ReadFile(metadataPath)
			if err != nil {
				t.Fatal(err)
			}
			var metadata map[string]any
			if err := json.Unmarshal(contents, &metadata); err != nil {
				t.Fatal(err)
			}
			mutation.edit(&metadata)
			contents, err = json.Marshal(metadata)
			if err != nil {
				t.Fatal(err)
			}
			if err := os.WriteFile(metadataPath, contents, 0o644); err != nil {
				t.Fatal(err)
			}
			if _, err := Load(v, directory, Mini); err == nil {
				t.Fatal("Load accepted mismatched metadata")
			}
		})
	}
}

func TestModelStateAndPolicyAgreeOn209CandidateStatistics(t *testing.T) {
	if modelstate.CandidateStatsSize != 209 {
		t.Fatalf("modelstate candidate stats=%d, want 209", modelstate.CandidateStatsSize)
	}
	if policy.CandidateStatsSize != modelstate.CandidateStatsSize {
		t.Fatalf("policy candidate stats=%d, want modelstate value %d", policy.CandidateStatsSize, modelstate.CandidateStatsSize)
	}
}

func loadMini(t *testing.T) *Data {
	t.Helper()
	data, err := Load(loadVocabulary(t), filepath.Join(repositoryRoot(t), "data", "imitation"), Mini)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func loadVocabulary(t *testing.T) *vocabulary.Vocabulary {
	t.Helper()
	v, err := vocabulary.Load(filepath.Join(repositoryRoot(t), "data"))
	if err != nil {
		t.Fatal(err)
	}
	return v
}

func copyMini(t *testing.T) string {
	t.Helper()
	directory := t.TempDir()
	for _, filename := range []string{"wordle-mini.bin", "wordle-mini.json"} {
		contents, err := os.ReadFile(filepath.Join(repositoryRoot(t), "data", "imitation", filename))
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(directory, filename), contents, 0o644); err != nil {
			t.Fatal(err)
		}
	}
	return directory
}

func repositoryRoot(t *testing.T) string {
	t.Helper()
	root, err := filepath.Abs(filepath.Join("..", ".."))
	if err != nil {
		t.Fatal(err)
	}
	return root
}
