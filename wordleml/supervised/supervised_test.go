package supervised

import (
	"math"
	"testing"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/sam-bee/wordle-ml_machine-learning/policy"
)

var testBackend = compute.MustNew()

func TestApplyAvailabilityMask(t *testing.T) {
	exec := model.MustNewExec(testBackend, model.NewStore(), func(_ *model.Scope, logits, available *graph.Node) *graph.Node {
		return ApplyAvailabilityMask(logits, available)
	})
	defer exec.Finalize()

	output := exec.MustCall1(
		[][]float32{{10, 20, 30}},
		[][]float32{{1, 0, 1}},
	)
	defer func() { _ = output.FinalizeAll() }()
	got := output.Value().([][]float32)[0]
	if got[0] != 10 || got[2] != 30 {
		t.Fatalf("available actions changed: %v", got)
	}
	if got[1] > -1e8 {
		t.Fatalf("used action was not suppressed: %v", got)
	}
}

func TestTopKAccuracyGraph(t *testing.T) {
	exec := model.MustNewExec(testBackend, model.NewStore(), func(scope *model.Scope, labels, logits *graph.Node) (*graph.Node, *graph.Node, *graph.Node) {
		return topKAccuracyGraph(1)(scope, []*graph.Node{labels}, []*graph.Node{logits}),
			topKAccuracyGraph(5)(scope, []*graph.Node{labels}, []*graph.Node{logits}),
			topKAccuracyGraph(16)(scope, []*graph.Node{labels}, []*graph.Node{logits})
	})
	defer exec.Finalize()

	logits := make([][]float32, 4)
	for row := range logits {
		logits[row] = make([]float32, 16)
		for action := range logits[row] {
			logits[row][action] = float32(16 - action)
		}
	}
	got := exec.MustCall([][]int32{{0}, {4}, {10}, {15}}, logits)
	defer finalizeAll(t, got)
	assertClose(t, scalar(got[0]), 0.25)
	assertClose(t, scalar(got[1]), 0.5)
	assertClose(t, scalar(got[2]), 1)
}

func TestTrainStepUpdatesPolicyAndCheckpointRestoresIt(t *testing.T) {
	config := Config{
		Policy:       policy.Config{NumSolutions: 4, NumActions: 16},
		LearningRate: 0.01,
		Seed:         42,
	}
	session, err := New(config, testBackend)
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	// Eval builds the policy variables without taking an optimisation step.
	metrics, err := session.Trainer.EvalStep(testBatch())
	if err != nil {
		t.Fatalf("EvalStep: %v", err)
	}
	finalizeAll(t, metrics)
	bias := session.Store.GetVariable("/wordle_policy/base_logits/dense/biases")
	before := tensors.MustCopyFlatData[float32](bias.MustValue())
	weights := session.Store.GetVariable("/wordle_policy/base_logits/dense/weights")
	weightsBefore := tensors.MustCopyFlatData[float32](weights.MustValue())

	metrics, err = session.Trainer.TrainStep(testBatch())
	if err != nil {
		t.Fatalf("TrainStep: %v", err)
	}
	if !isFinite(scalar(metrics[0])) {
		t.Fatalf("loss is not finite: %v", scalar(metrics[0]))
	}
	finalizeAll(t, metrics)
	if got := session.Trainer.GlobalStep(); got != 1 {
		t.Fatalf("global step = %d, want 1", got)
	}
	after := tensors.MustCopyFlatData[float32](bias.MustValue())
	if equalFloat32Slices(before, after) {
		t.Fatal("base logits bias did not change after TrainStep")
	}
	weightsAfter := tensors.MustCopyFlatData[float32](weights.MustValue())
	if equalFloat32Slices(weightsBefore, weightsAfter) {
		t.Fatal("base logits weights did not change after TrainStep")
	}

	dir := t.TempDir()
	handler, err := NewCheckpoint(session.Store, dir)
	if err != nil {
		t.Fatalf("NewCheckpoint: %v", err)
	}
	if err := handler.Save(); err != nil {
		t.Fatalf("checkpoint Save: %v", err)
	}

	restored, err := New(config, testBackend)
	if err != nil {
		t.Fatalf("new restored session: %v", err)
	}
	if _, err := NewCheckpoint(restored.Store, dir); err != nil {
		t.Fatalf("resume checkpoint: %v", err)
	}
	metrics, err = restored.Trainer.EvalStep(testBatch())
	if err != nil {
		t.Fatalf("evaluate restored session: %v", err)
	}
	finalizeAll(t, metrics)
	if got := restored.Trainer.GlobalStep(); got != 1 {
		t.Fatalf("restored global step = %d, want 1", got)
	}
	restoredBias := restored.Store.GetVariable("/wordle_policy/base_logits/dense/biases")
	if restoredBias == nil {
		t.Fatal("restored base logits bias is missing")
	}
	if !equalFloat32Slices(after, tensors.MustCopyFlatData[float32](restoredBias.MustValue())) {
		t.Fatal("restored base logits bias differs from checkpoint")
	}
	restoredWeights := restored.Store.GetVariable("/wordle_policy/base_logits/dense/weights")
	if restoredWeights == nil || !equalFloat32Slices(weightsAfter, tensors.MustCopyFlatData[float32](restoredWeights.MustValue())) {
		t.Fatal("restored base logits weights differ from checkpoint")
	}
	metrics, err = restored.Trainer.TrainStep(testBatch())
	if err != nil {
		t.Fatalf("train restored session: %v", err)
	}
	finalizeAll(t, metrics)
	if got := restored.Trainer.GlobalStep(); got != 2 {
		t.Fatalf("resumed global step = %d, want 2", got)
	}
}

func TestConfigValidation(t *testing.T) {
	base := Config{Policy: policy.Config{NumSolutions: 4, NumActions: 16}, LearningRate: 0.01, Seed: 1}
	for name, config := range map[string]Config{
		"zero learning rate": {Policy: base.Policy, Seed: base.Seed},
		"NaN learning rate":  {Policy: base.Policy, LearningRate: math.NaN(), Seed: base.Seed},
		"infinite learning rate": {
			Policy: base.Policy, LearningRate: math.Inf(1), Seed: base.Seed,
		},
		"zero seed":       {Policy: base.Policy, LearningRate: base.LearningRate},
		"too few actions": {Policy: policy.Config{NumSolutions: 4, NumActions: 15}, LearningRate: base.LearningRate, Seed: base.Seed},
	} {
		t.Run(name, func(t *testing.T) {
			if _, err := New(config, testBackend); err == nil {
				t.Fatal("New unexpectedly succeeded")
			}
		})
	}
}

func testBatch() train.Batch {
	return train.Batch{
		Inputs: []*tensors.Tensor{
			tensors.FromFlatDataAndDimensions([]float32{1, 0, 0, 0, 0, 1, 0, 0}, 2, 4),
			tensors.FromFlatDataAndDimensions(make([]float32, 2*policy.CandidateStatsSize), 2, policy.CandidateStatsSize),
			tensors.FromFlatDataAndDimensions([]int32{0, 1}, 2),
			tensors.FromFlatDataAndDimensions(make([]float32, 2*16), 2, 16),
			tensors.FromFlatDataAndDimensions(ones(2*16), 2, 16),
		},
		Labels: []*tensors.Tensor{
			tensors.FromFlatDataAndDimensions([]int32{1, 2}, 2, 1),
		},
	}
}

func ones(size int) []float32 {
	values := make([]float32, size)
	for i := range values {
		values[i] = 1
	}
	return values
}

func scalar(tensor *tensors.Tensor) float32 {
	return tensor.Value().(float32)
}

func assertClose(t *testing.T, got, want float32) {
	t.Helper()
	if math.Abs(float64(got-want)) > 1e-6 {
		t.Fatalf("got %v, want %v", got, want)
	}
}

func isFinite(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}

func equalFloat32Slices(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func finalizeAll(t *testing.T, tensors []*tensors.Tensor) {
	t.Helper()
	for _, tensor := range tensors {
		if err := tensor.FinalizeAll(); err != nil {
			t.Fatalf("finalize metric: %v", err)
		}
	}
}
