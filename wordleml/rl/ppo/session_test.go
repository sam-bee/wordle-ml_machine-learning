package ppo

import (
	"math"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/policy"
	"github.com/sam-bee/wordle-ml_machine-learning/supervised"
)

var sessionTestBackend = compute.MustNew()

func TestBaselineWrapCriticAndActorOnlyCompatibility(t *testing.T) {
	policyConfig := policy.Config{NumSolutions: 4, NumActions: 16}
	input := testActorInput(policyConfig)
	baseline, supervisedLogits := createBaselineCheckpoint(t, policyConfig, input)

	actor := newTestActor(t, policyConfig, 0)
	defer actor.Finalize()
	reference := newTestActor(t, policyConfig, 0)
	defer reference.Finalize()
	if err := LoadBaselineActor(baseline, actor, reference); err != nil {
		t.Fatal(err)
	}
	actorLogits := mustActorPredict(t, actor, []modelstate.Inputs{input})
	referenceLogits := mustActorPredict(t, reference, []modelstate.Inputs{input})
	if !reflect.DeepEqual(actorLogits[0], supervisedLogits) {
		t.Fatal("PPO wrapper does not reproduce direct supervised logits")
	}
	if !reflect.DeepEqual(actorLogits, referenceLogits) {
		t.Fatal("baseline actor clone changed logits")
	}

	critic, err := NewCriticSession(CriticConfig{Policy: policyConfig, LearningRate: 1e-4, ValueLossCoefficient: 0.5, MaximumGradientNorm: 1, Seed: 321}, sessionTestBackend)
	if err != nil {
		t.Fatal(err)
	}
	defer critic.Finalize()
	if _, err := critic.Predict([]modelstate.Inputs{input}); err != nil {
		t.Fatal(err)
	}
	afterCritic := mustActorPredict(t, actor, []modelstate.Inputs{input})
	if !reflect.DeepEqual(actorLogits, afterCritic) {
		t.Fatal("initializing separate critic changed actor logits")
	}

	exportDir := filepath.Join(t.TempDir(), "actor-only")
	if err := ExportActorOnly(actor, exportDir); err != nil {
		t.Fatal(err)
	}
	deployed, err := supervised.New(supervised.Config{Policy: policyConfig, LearningRate: 1e-3, Seed: 123}, sessionTestBackend)
	if err != nil {
		t.Fatal(err)
	}
	defer deployed.Finalize()
	if _, err := supervised.NewCheckpoint(deployed.Store, exportDir); err != nil {
		t.Fatal(err)
	}
	deployedRaw, _, _, err := deployed.PredictDiagnostics(
		[][]float32{input.CandidateMask}, [][]float32{input.CandidateStats}, []int32{input.Turn},
		[][]float32{input.RemainingActionMask}, [][]float32{ones32(policyConfig.NumActions)},
	)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = deployedRaw.FinalizeAll() }()
	if got := tensors.MustCopyFlatData[float32](deployedRaw); !reflect.DeepEqual(got, actorLogits[0]) {
		t.Fatal("actor-only export does not reproduce PPO actor")
	}
}

func TestZeroLearningRateLeavesActorParametersUnchanged(t *testing.T) {
	config := policy.Config{NumSolutions: 4, NumActions: 16}
	input := testActorInput(config)
	actor := newTestActor(t, config, 0)
	defer actor.Finalize()
	// Materialize parameters before taking the identity checksum.
	logits := mustActorPredict(t, actor, []modelstate.Inputs{input})
	before, err := VariableChecksum(actor.Store, actorVariablePrefix)
	if err != nil {
		t.Fatal(err)
	}
	distribution, _ := NewCategorical(logits[0], ones32(config.NumActions))
	action, _ := distribution.Greedy()
	if _, err := actor.TrainStep(ActorBatch{
		Inputs:          []modelstate.Inputs{input},
		Availability:    [][]float32{ones32(config.NumActions)},
		Actions:         []int32{int32(action)},
		OldLogProbs:     []float32{float32(distribution.LogProbability(action))},
		Advantages:      []float32{1},
		ReferenceLogits: logits,
	}); err != nil {
		t.Fatal(err)
	}
	after, err := VariableChecksum(actor.Store, actorVariablePrefix)
	if err != nil {
		t.Fatal(err)
	}
	if before != after {
		t.Fatalf("zero-learning-rate actor changed: %s != %s", before, after)
	}
}

func TestActorCriticCheckpointResumeIncludesOptimizerAndState(t *testing.T) {
	config := policy.Config{NumSolutions: 4, NumActions: 16}
	input := testActorInput(config)
	actor := newTestActor(t, config, 1e-3)
	defer actor.Finalize()
	logits := mustActorPredict(t, actor, []modelstate.Inputs{input})
	distribution, _ := NewCategorical(logits[0], ones32(config.NumActions))
	action, _ := distribution.Greedy()
	batch := ActorBatch{
		Inputs:          []modelstate.Inputs{input},
		Availability:    [][]float32{ones32(config.NumActions)},
		Actions:         []int32{int32(action)},
		OldLogProbs:     []float32{float32(distribution.LogProbability(action))},
		Advantages:      []float32{1},
		ReferenceLogits: logits,
	}
	if _, err := actor.TrainStep(batch); err != nil {
		t.Fatal(err)
	}
	actor.Store.SetParam("ppo_run_state", `{"iteration":3,"seed":99}`)
	actorDir := filepath.Join(t.TempDir(), "actor")
	if err := SaveStoreCheckpoint(actor.Store, actorDir); err != nil {
		t.Fatal(err)
	}
	if _, err := actor.TrainStep(batch); err != nil {
		t.Fatal(err)
	}
	want, err := VariableChecksum(actor.Store, actorVariablePrefix)
	if err != nil {
		t.Fatal(err)
	}

	restored := newTestActor(t, config, 1e-3)
	defer restored.Finalize()
	if err := LoadActorCheckpoint(restored, actorDir); err != nil {
		t.Fatal(err)
	}
	if value, found := restored.Store.GetParam("ppo_run_state"); !found || value != `{"iteration":3,"seed":99}` {
		t.Fatalf("restored PPO state = %v, %t", value, found)
	}
	if _, err := restored.TrainStep(batch); err != nil {
		t.Fatal(err)
	}
	got, err := VariableChecksum(restored.Store, actorVariablePrefix)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("resumed actor/Adam update differs: %s != %s", got, want)
	}

	critic, err := NewCriticSession(CriticConfig{Policy: config, LearningRate: 1e-3, ValueLossCoefficient: 0.5, MaximumGradientNorm: 1, Seed: 456}, sessionTestBackend)
	if err != nil {
		t.Fatal(err)
	}
	defer critic.Finalize()
	criticBatch := CriticBatch{Inputs: []modelstate.Inputs{input}, Returns: []float32{0.75}}
	if _, err := critic.TrainStep(criticBatch); err != nil {
		t.Fatal(err)
	}
	criticDir := filepath.Join(t.TempDir(), "critic")
	if err := SaveStoreCheckpoint(critic.Store, criticDir); err != nil {
		t.Fatal(err)
	}
	if _, err := critic.TrainStep(criticBatch); err != nil {
		t.Fatal(err)
	}
	wantValue, err := critic.Predict(criticBatch.Inputs)
	if err != nil {
		t.Fatal(err)
	}
	restoredCritic, err := NewCriticSession(CriticConfig{Policy: config, LearningRate: 1e-3, ValueLossCoefficient: 0.5, MaximumGradientNorm: 1, Seed: 456}, sessionTestBackend)
	if err != nil {
		t.Fatal(err)
	}
	defer restoredCritic.Finalize()
	if err := LoadCriticCheckpoint(restoredCritic, criticDir); err != nil {
		t.Fatal(err)
	}
	if _, err := restoredCritic.TrainStep(criticBatch); err != nil {
		t.Fatal(err)
	}
	gotValue, err := restoredCritic.Predict(criticBatch.Inputs)
	if err != nil || math.Abs(float64(gotValue[0]-wantValue[0])) > 1e-7 {
		t.Fatalf("restored critic value = %v, %v; want %v", gotValue, err, wantValue)
	}
}

func createBaselineCheckpoint(t *testing.T, config policy.Config, input modelstate.Inputs) (string, []float32) {
	t.Helper()
	session, err := supervised.New(supervised.Config{Policy: config, LearningRate: 1e-3, Seed: 123}, sessionTestBackend)
	if err != nil {
		t.Fatal(err)
	}
	defer session.Finalize()
	raw, masked, beta, err := session.PredictDiagnostics(
		[][]float32{input.CandidateMask}, [][]float32{input.CandidateStats}, []int32{input.Turn},
		[][]float32{input.RemainingActionMask}, [][]float32{ones32(config.NumActions)},
	)
	if err != nil {
		t.Fatal(err)
	}
	directLogits := tensors.MustCopyFlatData[float32](raw)
	_ = raw.FinalizeAll()
	_ = masked.FinalizeAll()
	_ = beta.FinalizeAll()
	dir := filepath.Join(t.TempDir(), "baseline")
	handler, err := supervised.NewCheckpoint(session.Store, dir)
	if err != nil || handler.Save() != nil {
		t.Fatalf("save baseline checkpoint: %v", err)
	}
	return dir, directLogits
}

func newTestActor(t *testing.T, config policy.Config, learningRate float64) *ActorSession {
	t.Helper()
	session, err := NewActorSession(ActorConfig{
		Policy: config, LearningRate: learningRate, ClipRange: 0.1,
		EntropyCoefficient: 0.001, SupervisedReferenceKLCoeff: 0.05,
		MaximumGradientNorm: 1, Seed: 987,
	}, sessionTestBackend)
	if err != nil {
		t.Fatal(err)
	}
	return session
}

func mustActorPredict(t *testing.T, actor *ActorSession, inputs []modelstate.Inputs) [][]float32 {
	t.Helper()
	logits, err := actor.Predict(inputs)
	if err != nil {
		t.Fatal(err)
	}
	return logits
}

func testActorInput(config policy.Config) modelstate.Inputs {
	candidate := make([]float32, config.NumSolutions)
	candidate[0] = 1
	remaining := make([]float32, config.NumActions)
	remaining[0] = 1
	return modelstate.Inputs{CandidateMask: candidate, CandidateStats: make([]float32, modelstate.CandidateStatsSize), Turn: 0, RemainingActionMask: remaining}
}

func ones32(size int) []float32 {
	values := make([]float32, size)
	for index := range values {
		values[index] = 1
	}
	return values
}
