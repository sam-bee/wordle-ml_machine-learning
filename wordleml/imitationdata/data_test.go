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
	synthetic "github.com/sam-bee/wordle-ml_synthetic-data-creation/dataset"
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
	if example.RecordIndex < 0 {
		t.Fatalf("record index=%d, want non-negative", example.RecordIndex)
	}
	for rank, actionID := range example.Record.TopKActionIDs {
		if got := example.TeacherTopActions[rank]; got != int32(actionID) {
			t.Fatalf("teacher rank %d=%d, want %d", rank+1, got, actionID)
		}
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
		if len(batch.Labels) != 1 {
			t.Fatalf("labels=%d, want top-1", len(batch.Labels))
		}
		if err := batch.Labels[0].Shape().Check(dtypes.Int32, 1); err != nil {
			t.Fatalf("top-1 label: %v", err)
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
			t.Fatalf("top-1 label: %v", err)
		}
		return
	}
	t.Fatal("batched dataset produced no batch")
}

func TestTrainingSamplerInjectsOpeningAndResumesExactly(t *testing.T) {
	training := loadTrain(t)
	opening, found, err := training.FindOpening()
	if err != nil {
		t.Fatalf("FindOpening(train): %v", err)
	}
	if !found || opening.Record.SolutionID != synthetic.OpeningSolutionID || opening.Record.TurnDepth != 0 {
		t.Fatalf("opening = %+v, found=%t", opening.Record, found)
	}
	mini := loadMini(t)
	if _, found, err := mini.FindOpening(); err != nil || found {
		t.Fatalf("FindOpening(mini) found=%t err=%v, want no opening", found, err)
	}
	validationSource := *mini
	validationSource.split = Validation
	if _, err := NewTrainingSampler(&validationSource, opening, 128, 71, Cursor{}); err == nil {
		t.Fatal("NewTrainingSampler accepted validation data")
	}

	const batchSize = 129
	sampler, err := NewTrainingSampler(mini, opening, batchSize, 71, Cursor{})
	if err != nil {
		t.Fatalf("NewTrainingSampler: %v", err)
	}
	seenFirstEpoch := make([]bool, mini.Len())
	seenCount := 0
	for batchNumber := 0; batchNumber < 13; batchNumber++ {
		examples, err := sampler.Next()
		if err != nil {
			t.Fatalf("Next batch %d: %v", batchNumber, err)
		}
		if len(examples) != batchSize || examples[0].Record.SolutionID != synthetic.OpeningSolutionID {
			t.Fatalf("batch %d did not reserve its first slot for opening", batchNumber)
		}
		for index, example := range examples[1:] {
			if example.Record.SolutionID == synthetic.OpeningSolutionID {
				t.Fatalf("batch %d source slot %d duplicated opening", batchNumber, index+1)
			}
			if seenCount >= mini.Len() {
				continue
			}
			if seenFirstEpoch[example.RecordIndex] {
				t.Fatalf("source record %d repeated before epoch was exhausted", example.RecordIndex)
			}
			seenFirstEpoch[example.RecordIndex] = true
			seenCount++
		}
	}
	if seenCount != mini.Len() {
		t.Fatalf("first epoch yielded %d unique records, want %d", seenCount, mini.Len())
	}

	saved := sampler.Cursor()
	wantNext := sampler.Peek()
	restored, err := NewTrainingSampler(mini, opening, batchSize, 71, saved)
	if err != nil {
		t.Fatalf("restore sampler: %v", err)
	}
	if got := restored.Peek(); !reflect.DeepEqual(got, wantNext) {
		t.Fatalf("restored next IDs=%v, want %v", got, wantNext)
	}
	wantBatch, err := sampler.Next()
	if err != nil {
		t.Fatal(err)
	}
	gotBatch, err := restored.Next()
	if err != nil {
		t.Fatal(err)
	}
	for index := range wantBatch {
		if gotBatch[index].RecordIndex != wantBatch[index].RecordIndex ||
			gotBatch[index].Record.SolutionID != wantBatch[index].Record.SolutionID {
			t.Fatalf("restored batch record %d=(%d,%d), want (%d,%d)", index,
				gotBatch[index].RecordIndex, gotBatch[index].Record.SolutionID,
				wantBatch[index].RecordIndex, wantBatch[index].Record.SolutionID)
		}
	}
}

func TestMaterializeBatchCopiesFixedInputsAndTeacherRanking(t *testing.T) {
	mini := loadMini(t)
	examples := make([]Example, 3)
	for index := range examples {
		var err error
		examples[index], err = mini.Example(index)
		if err != nil {
			t.Fatal(err)
		}
	}
	batch, err := MaterializeBatch(examples)
	if err != nil {
		t.Fatalf("MaterializeBatch: %v", err)
	}
	defer batch.Finalize()
	if err := batch.Inputs[0].Shape().Check(dtypes.Float32, 3, vocabulary.NumSolutions); err != nil {
		t.Fatal(err)
	}
	if err := batch.Inputs[4].Shape().Check(dtypes.Float32, 3, vocabulary.NumActions); err != nil {
		t.Fatal(err)
	}
	if err := batch.Labels[0].Shape().Check(dtypes.Int32, 3, 1); err != nil {
		t.Fatal(err)
	}
	if err := batch.Labels[1].Shape().Check(dtypes.Int32, 3, 16); err != nil {
		t.Fatal(err)
	}
	top16 := batch.Labels[1].Value().([][]int32)
	for row := range examples {
		if !reflect.DeepEqual(top16[row], examples[row].TeacherTopActions[:]) {
			t.Fatalf("teacher row %d=%v, want %v", row, top16[row], examples[row].TeacherTopActions)
		}
	}
	// The batch must not alias Example storage because Trainer may donate it.
	before := batch.Inputs[0].Value().([][]float32)[0][0]
	examples[0].CandidateMask[0] = before + 10
	if got := batch.Inputs[0].Value().([][]float32)[0][0]; got != before {
		t.Fatalf("materialized candidate mask aliased source: got %v, want %v", got, before)
	}
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

func loadTrain(t *testing.T) *Data {
	t.Helper()
	data, err := Load(loadVocabulary(t), filepath.Join(repositoryRoot(t), "data", "imitation"), Train)
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
