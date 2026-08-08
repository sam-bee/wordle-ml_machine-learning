package proofrun

import (
	"encoding/json"
	"path/filepath"
	"slices"
	"testing"

	"github.com/gomlx/gomlx/core/tensors"
	"github.com/sam-bee/wordle-ml_machine-learning/imitationdata"
	"github.com/sam-bee/wordle-ml_machine-learning/runstate"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

func TestTrainingMetricsUseEMALossAndTopKOffsets(t *testing.T) {
	values := []*tensors.Tensor{
		tensors.FromScalar(float32(9)), // per-batch loss, intentionally ignored
		tensors.FromScalar(float32(8)), // EMA loss
		tensors.FromScalar(float32(1)),
		tensors.FromScalar(float32(2)),
		tensors.FromScalar(float32(3)),
	}
	metrics, err := trainingMetricsFromTensors(values)
	if err != nil {
		t.Fatalf("trainingMetricsFromTensors: %v", err)
	}
	if metrics != (Metrics{Loss: 8, Top1: 1, Top5: 2, Top16: 3}) {
		t.Fatalf("metrics = %#v", metrics)
	}
}

func TestResumedSamplerMustMatchCheckpointPeek(t *testing.T) {
	dataDir := filepath.Join("..", "..", "data")
	vocabulary, err := vocabulary.Load(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	trainData, err := imitationdata.Load(vocabulary, filepath.Join(dataDir, "imitation"), imitationdata.Train)
	if err != nil {
		t.Fatal(err)
	}
	miniData, err := imitationdata.Load(vocabulary, filepath.Join(dataDir, "imitation"), imitationdata.Mini)
	if err != nil {
		t.Fatal(err)
	}
	opening, found, err := trainData.FindOpening()
	if err != nil || !found {
		t.Fatalf("FindOpening = found %t, err %v", found, err)
	}
	sampler, err := imitationdata.NewTrainingSampler(miniData, opening, 128, Seed, imitationdata.Cursor{})
	if err != nil {
		t.Fatal(err)
	}
	state := runstate.State{ShuffleSeed: Seed, NextRecordIDs: sampler.Peek()}
	if err := validateSamplerPeek(state, sampler); err != nil {
		t.Fatalf("matching sampler peek: %v", err)
	}
	state.NextRecordIDs[0]++
	if err := validateSamplerPeek(state, sampler); err == nil {
		t.Fatal("mismatched sampler peek was accepted")
	}
}

func TestMiniResumeProofFollowsUninterruptedSampler(t *testing.T) {
	dataDir := filepath.Join("..", "..", "data")
	vocabulary, err := vocabulary.Load(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	trainData, err := imitationdata.Load(vocabulary, filepath.Join(dataDir, "imitation"), imitationdata.Train)
	if err != nil {
		t.Fatal(err)
	}
	miniData, err := imitationdata.Load(vocabulary, filepath.Join(dataDir, "imitation"), imitationdata.Mini)
	if err != nil {
		t.Fatal(err)
	}
	opening, found, err := trainData.FindOpening()
	if err != nil || !found {
		t.Fatalf("FindOpening = found %t, err %v", found, err)
	}
	checkpointSampler, err := imitationdata.NewTrainingSampler(miniData, opening, 128, Seed, imitationdata.Cursor{})
	if err != nil {
		t.Fatal(err)
	}
	if err := checkpointSampler.AdvanceBatches(500); err != nil {
		t.Fatal(err)
	}
	state := runstate.State{GlobalUpdate: 500, ShuffleSeed: Seed, NextRecordIDs: checkpointSampler.Peek()}
	uninterrupted, err := imitationdata.NewTrainingSampler(miniData, opening, 128, Seed, imitationdata.Cursor{})
	if err != nil {
		t.Fatal(err)
	}
	proof, err := resumeProofForStop(state, uninterrupted, 500)
	if err != nil {
		t.Fatalf("resumeProofForStop: %v", err)
	}
	if !slices.Equal(proof.CheckpointNextRecordIDs, proof.UninterruptedReferenceNextRecordIDs) {
		t.Fatalf("checkpoint IDs %v differ from uninterrupted IDs %v", proof.CheckpointNextRecordIDs, proof.UninterruptedReferenceNextRecordIDs)
	}
	resumed, err := resumeProofForResume(state, proof)
	if err != nil {
		t.Fatalf("resumeProofForResume: %v", err)
	}
	if resumed.ResumeFromUpdate != 500 || resumed.FirstResumedScalarUpdate != 0 || resumed.Completed {
		t.Fatalf("initial resumed proof = %#v", resumed)
	}
	// This assignment mirrors the point immediately after successful scalar
	// telemetry publication in Run.
	resumed.FirstResumedScalarUpdate = 510
	resumed.Completed = true
	contents, err := json.Marshal(Result{Stage: Mini, GlobalUpdate: 1000, ResumeProof: resumed})
	if err != nil {
		t.Fatalf("marshal final mini result: %v", err)
	}
	var decoded Result
	if err := json.Unmarshal(contents, &decoded); err != nil {
		t.Fatalf("unmarshal final mini result: %v", err)
	}
	if decoded.ResumeProof == nil || decoded.ResumeProof.ResumeFromUpdate != 500 || decoded.ResumeProof.FirstResumedScalarUpdate != 510 || !decoded.ResumeProof.Completed {
		t.Fatalf("persisted completed resume proof = %#v", decoded.ResumeProof)
	}
}

func TestMiniGateRequiresCompletedResumeProof(t *testing.T) {
	passing := Result{
		Stage:           Mini,
		GlobalUpdate:    1000,
		InitialTraining: Metrics{Loss: 4, Top1: 0.1, Top5: 0.2, Top16: 0.3},
		FinalTraining:   Metrics{Loss: 1.9, Top1: 0.3, Top5: 0.4, Top16: 0.5},
		ResumeProof: &ResumeProof{
			CheckpointNextRecordIDs:             []int{1, 2, 3},
			UninterruptedReferenceNextRecordIDs: []int{1, 2, 3},
			ResumeFromUpdate:                    500,
			FirstResumedScalarUpdate:            510,
			Completed:                           true,
		},
		TelemetryProof: &MiniTelemetryProof{TrainingSteps: []int64{10, 500, 510, 1000}},
	}
	if !miniGatePassed(passing) {
		t.Fatal("complete learning and resume evidence did not pass the mini gate")
	}
	for name, mutate := range map[string]func(*Result){
		"nil proof":         func(result *Result) { result.ResumeProof = nil },
		"not completed":     func(result *Result) { result.ResumeProof.Completed = false },
		"wrong resume step": func(result *Result) { result.ResumeProof.ResumeFromUpdate = 400 },
		"wrong scalar step": func(result *Result) { result.ResumeProof.FirstResumedScalarUpdate = 500 },
		"wrong IDs":         func(result *Result) { result.ResumeProof.CheckpointNextRecordIDs[0]++ },
		"no event proof":    func(result *Result) { result.TelemetryProof = nil },
	} {
		t.Run(name, func(t *testing.T) {
			candidate := passing
			candidate.ResumeProof = cloneResumeProof(passing.ResumeProof)
			mutate(&candidate)
			if miniGatePassed(candidate) {
				t.Fatal("incomplete resume evidence passed the mini gate")
			}
		})
	}
}

func TestFullGateRequiresOptimizationHeldOutTopKAndBroadGroups(t *testing.T) {
	passing := Result{
		GlobalUpdate:           2000,
		InitialTraining:        Metrics{Loss: 8, Top1: .1, Top5: .2, Top16: .3},
		FinalTraining:          Metrics{Loss: 4, Top1: .4, Top5: .5, Top16: .6},
		InitialValidation:      Metrics{Loss: 8, Top1: .1, Top5: .2, Top16: .3},
		BestValidation:         Metrics{Loss: 7.5, Top1: .11, Top5: .21, Top16: .31},
		ValidationImprovements: 2,
		MajorGroupLearning:     MajorGroupLearning{TurnCount: 2, ShortlistCount: 2},
	}
	if !fullGatePassed(passing) {
		t.Fatal("complete full-stage learning evidence did not pass")
	}
	for name, mutate := range map[string]func(*Result){
		"training loss":   func(result *Result) { result.FinalTraining.Loss = result.InitialTraining.Loss },
		"validation loss": func(result *Result) { result.BestValidation.Loss = result.InitialValidation.Loss * .96 },
		"validation top1": func(result *Result) { result.BestValidation.Top1 = result.InitialValidation.Top1 },
		"checkpoints":     func(result *Result) { result.ValidationImprovements = 1 },
		"turn groups":     func(result *Result) { result.MajorGroupLearning.TurnCount = 1 },
		"shortlist groups": func(result *Result) {
			result.MajorGroupLearning.ShortlistCount = 1
		},
	} {
		t.Run(name, func(t *testing.T) {
			candidate := passing
			mutate(&candidate)
			if fullGatePassed(candidate) {
				t.Fatal("incomplete full-stage evidence passed")
			}
		})
	}
}
