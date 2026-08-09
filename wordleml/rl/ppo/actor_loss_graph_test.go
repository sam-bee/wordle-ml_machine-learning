package ppo

import (
	"math"
	"testing"

	"github.com/gomlx/gomlx/core/tensors"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/policy"
)

// These tests deliberately use ActorSession.TrainStep rather than the host
// loss helpers. They exercise the compiled GoMLX masked-log-softmax loss,
// global-gradient clipper, and Adam update together.

func TestActorLossGraphMovesSelectedProbabilityWithAdvantageSign(t *testing.T) {
	config := policy.Config{NumSolutions: 4, NumActions: 16}
	input := testActorInput(config)
	availability := graphTestAvailability(config.NumActions, 0, 1)

	positive := newGraphTestActor(t, config)
	defer positive.Finalize()
	positiveBefore, action := graphTestActionProbability(t, positive, input, availability)
	positiveLogits := mustActorPredict(t, positive, []modelstate.Inputs{input})
	positiveDistribution, err := NewCategorical(positiveLogits[0], availability)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := positive.TrainStep(graphTestBatch(input, availability, action, positiveDistribution.LogProbability(action), 1, positiveLogits)); err != nil {
		t.Fatalf("positive-advantage TrainStep: %v", err)
	}
	positiveAfter := graphTestProbability(t, positive, input, availability, action)
	if !(positiveAfter > positiveBefore) {
		t.Fatalf("positive advantage changed selected probability from %.9f to %.9f, want increase", positiveBefore, positiveAfter)
	}

	negative := newGraphTestActor(t, config)
	defer negative.Finalize()
	negativeBefore, negativeAction := graphTestActionProbability(t, negative, input, availability)
	if negativeAction != action || math.Abs(negativeBefore-positiveBefore) > 1e-7 {
		t.Fatalf("same-seed actor initialization changed: action/probability %d/%.9f vs %d/%.9f", action, positiveBefore, negativeAction, negativeBefore)
	}
	negativeLogits := mustActorPredict(t, negative, []modelstate.Inputs{input})
	negativeDistribution, err := NewCategorical(negativeLogits[0], availability)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := negative.TrainStep(graphTestBatch(input, availability, negativeAction, negativeDistribution.LogProbability(negativeAction), -1, negativeLogits)); err != nil {
		t.Fatalf("negative-advantage TrainStep: %v", err)
	}
	negativeAfter := graphTestProbability(t, negative, input, availability, negativeAction)
	if !(negativeAfter < negativeBefore) {
		t.Fatalf("negative advantage changed selected probability from %.9f to %.9f, want decrease", negativeBefore, negativeAfter)
	}
}

func TestActorLossGraphClipsExaggeratedPositiveRatio(t *testing.T) {
	config := policy.Config{NumSolutions: 4, NumActions: 16}
	input := testActorInput(config)
	availability := graphTestAvailability(config.NumActions, 0, 1)
	actor := newGraphTestActor(t, config)
	defer actor.Finalize()

	_, action := graphTestActionProbability(t, actor, input, availability)
	logits := mustActorPredict(t, actor, []modelstate.Inputs{input})
	distribution, err := NewCategorical(logits[0], availability)
	if err != nil {
		t.Fatal(err)
	}
	// pi/current divided by pi/old is 1.5 before the update, well above the
	// configured 1.1 upper clip boundary. With entropy and reference penalties
	// disabled, the clipped branch is constant and must have zero actor gradient.
	oldLogProbability := distribution.LogProbability(action) - math.Log(1.5)
	before, err := VariableChecksum(actor.Store, actorVariablePrefix)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := actor.TrainStep(graphTestBatch(input, availability, action, oldLogProbability, 1, logits)); err != nil {
		t.Fatalf("clipped TrainStep: %v", err)
	}
	after, err := VariableChecksum(actor.Store, actorVariablePrefix)
	if err != nil {
		t.Fatal(err)
	}
	if after != before {
		t.Fatalf("upper-clipped PPO sample changed actor parameters: %s != %s", after, before)
	}
}

func TestActorLossGraphUsesStoredMaskAndRejectsNonFiniteBatchValues(t *testing.T) {
	config := policy.Config{NumSolutions: 4, NumActions: 16}
	input := testActorInput(config)
	availability := graphTestAvailability(config.NumActions, 0, 1)
	const maskedAction = 2
	actor := newGraphTestActor(t, config)
	defer actor.Finalize()

	_, action := graphTestActionProbability(t, actor, input, availability)
	logits := mustActorPredict(t, actor, []modelstate.Inputs{input})
	distribution, err := NewCategorical(logits[0], availability)
	if err != nil {
		t.Fatal(err)
	}
	bias := actor.Store.GetVariable("/wordle_policy/base_logits/dense/biases")
	if bias == nil {
		t.Fatal("base-logit bias was not materialized")
	}
	beforeBias := tensors.MustCopyFlatData[float32](bias.MustValue())
	if _, err := actor.TrainStep(graphTestBatch(input, availability, action, distribution.LogProbability(action), 1, logits)); err != nil {
		t.Fatalf("masked TrainStep: %v", err)
	}
	afterBias := tensors.MustCopyFlatData[float32](bias.MustValue())
	if afterBias[maskedAction] != beforeBias[maskedAction] {
		t.Fatalf("masked action bias changed from %g to %g", beforeBias[maskedAction], afterBias[maskedAction])
	}
	afterLogits := mustActorPredict(t, actor, []modelstate.Inputs{input})
	afterDistribution, err := NewCategorical(afterLogits[0], availability)
	if err != nil {
		t.Fatal(err)
	}
	if afterDistribution.Probability(maskedAction) != 0 || !math.IsInf(afterDistribution.LogProbability(maskedAction), -1) {
		t.Fatalf("stored-mask action probability/logp = %v/%v, want 0/-Inf", afterDistribution.Probability(maskedAction), afterDistribution.LogProbability(maskedAction))
	}

	invalid := graphTestBatch(input, availability, action, math.NaN(), 1, logits)
	if _, err := actor.TrainStep(invalid); err == nil {
		t.Fatal("TrainStep accepted NaN old action log-probability")
	}
	invalid = graphTestBatch(input, availability, action, distribution.LogProbability(action), math.Inf(1), logits)
	if _, err := actor.TrainStep(invalid); err == nil {
		t.Fatal("TrainStep accepted infinite advantage")
	}
}

func newGraphTestActor(t *testing.T, config policy.Config) *ActorSession {
	t.Helper()
	actor, err := NewActorSession(ActorConfig{
		Policy:                     config,
		LearningRate:               1e-2,
		ClipRange:                  0.1,
		EntropyCoefficient:         0,
		SupervisedReferenceKLCoeff: 0,
		MaximumGradientNorm:        1,
		Seed:                       771,
	}, sessionTestBackend)
	if err != nil {
		t.Fatal(err)
	}
	return actor
}

func graphTestAvailability(actions int, legal ...int) []float32 {
	availability := make([]float32, actions)
	for _, action := range legal {
		availability[action] = 1
	}
	return availability
}

func graphTestActionProbability(t *testing.T, actor *ActorSession, input modelstate.Inputs, availability []float32) (float64, int) {
	t.Helper()
	logits := mustActorPredict(t, actor, []modelstate.Inputs{input})
	distribution, err := NewCategorical(logits[0], availability)
	if err != nil {
		t.Fatal(err)
	}
	action, err := distribution.Greedy()
	if err != nil {
		t.Fatal(err)
	}
	return distribution.Probability(action), action
}

func graphTestProbability(t *testing.T, actor *ActorSession, input modelstate.Inputs, availability []float32, action int) float64 {
	t.Helper()
	logits := mustActorPredict(t, actor, []modelstate.Inputs{input})
	distribution, err := NewCategorical(logits[0], availability)
	if err != nil {
		t.Fatal(err)
	}
	return distribution.Probability(action)
}

func graphTestBatch(input modelstate.Inputs, availability []float32, action int, oldLogProbability, advantage float64, referenceLogits [][]float32) ActorBatch {
	return ActorBatch{
		Inputs:          []modelstate.Inputs{input},
		Availability:    [][]float32{availability},
		Actions:         []int32{int32(action)},
		OldLogProbs:     []float32{float32(oldLogProbability)},
		Advantages:      []float32{float32(advantage)},
		ReferenceLogits: referenceLogits,
	}
}
