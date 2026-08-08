// Package supervised builds the small supervised-training wrapper around the
// Wordle policy. The policy remains responsible only for its four model inputs;
// action availability is a training-time concern.
package supervised

import (
	"fmt"
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/metric"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/sam-bee/wordle-ml_machine-learning/policy"
)

const unavailableActionPenalty = -1e9

// Config contains the fixed policy dimensions and the few supervised-training
// choices needed for an experiment.
type Config struct {
	Policy       policy.Config
	LearningRate float64
	Seed         int64
}

// Session owns one policy, its parameter store, and the trainer that updates it.
type Session struct {
	Policy  *policy.Model
	Store   *model.Store
	Trainer *train.Trainer
}

// New constructs a deterministic Adam training session.
func New(config Config, backend compute.Backend) (*Session, error) {
	if config.LearningRate <= 0 || math.IsNaN(config.LearningRate) || math.IsInf(config.LearningRate, 0) {
		return nil, fmt.Errorf("learning rate must be positive, got %g", config.LearningRate)
	}
	if config.Seed == 0 {
		return nil, fmt.Errorf("seed must be non-zero")
	}
	if config.Policy.NumActions < 16 {
		return nil, fmt.Errorf("number of actions must be at least 16 for top-16 accuracy, got %d", config.Policy.NumActions)
	}
	if backend == nil {
		return nil, fmt.Errorf("backend must not be nil")
	}
	policyModel, err := policy.New(config.Policy)
	if err != nil {
		return nil, err
	}

	store := model.NewStore()
	store.SetParam(model.ParamInitialSeed, config.Seed)
	session := &Session{Policy: policyModel, Store: store}
	session.Trainer = train.NewTrainer(
		backend,
		store,
		session.Forward,
		loss.SparseCategoricalCrossEntropyLogits,
		optimizer.Adam().LearningRate(config.LearningRate).Done(),
		trainMetrics(),
		evalMetrics(),
	)
	return session, nil
}

// Forward consumes the five tensors provided by imitationdata: candidate mask,
// candidate statistics, turn, remaining-solution action mask, and availability
// mask. The availability mask is deliberately applied after policy.Forward so it
// cannot change the policy architecture or suppress valid probe words.
func (s *Session) Forward(scope *model.Scope, inputs []*graph.Node) *graph.Node {
	if len(inputs) != 5 {
		panic(fmt.Sprintf("supervised model needs 5 inputs, got %d", len(inputs)))
	}
	logits := s.Policy.Forward(scope, inputs[0], inputs[1], inputs[2], inputs[3])
	return ApplyAvailabilityMask(logits, inputs[4])
}

// ApplyAvailabilityMask suppresses only actions which have already been used.
// It is outside policy.Model because remainingActionMask means "candidate
// solution", while availability means "not guessed earlier in this game".
func ApplyAvailabilityMask(logits, availableActions *graph.Node) *graph.Node {
	if err := logits.Shape().Check(dtypes.Float32, -1, -1); err != nil {
		panic(fmt.Errorf("logits: %w", err))
	}
	if err := availableActions.Shape().Check(dtypes.Float32, logits.Shape().Dimensions...); err != nil {
		panic(fmt.Errorf("available actions: %w", err))
	}
	penalty := graph.Mul(
		graph.Sub(graph.ScalarOne(availableActions.Graph(), dtypes.Float32), availableActions),
		graph.Scalar(availableActions.Graph(), dtypes.Float32, unavailableActionPenalty),
	)
	return graph.Add(logits, penalty)
}

// NewCheckpoint builds a checkpoint handler that retains the three newest
// checkpoints and resumes the newest one when it exists.
func NewCheckpoint(store *model.Store, dir string) (*checkpoint.Handler, error) {
	return checkpoint.Build(store).Dir(dir).Keep(3).Done()
}

func trainMetrics() []metric.Interface {
	return []metric.Interface{
		metric.NewExponentialMovingAverageMetric("Top-1 Accuracy", "top1", metric.AccuracyMetricType, topKAccuracyGraph(1), nil, 0.01),
		metric.NewExponentialMovingAverageMetric("Top-5 Accuracy", "top5", metric.AccuracyMetricType, topKAccuracyGraph(5), nil, 0.01),
		metric.NewExponentialMovingAverageMetric("Top-16 Accuracy", "top16", metric.AccuracyMetricType, topKAccuracyGraph(16), nil, 0.01),
	}
}

func evalMetrics() []metric.Interface {
	return []metric.Interface{
		metric.NewMeanMetric("Top-1 Accuracy", "top1", metric.AccuracyMetricType, topKAccuracyGraph(1), nil),
		metric.NewMeanMetric("Top-5 Accuracy", "top5", metric.AccuracyMetricType, topKAccuracyGraph(5), nil),
		metric.NewMeanMetric("Top-16 Accuracy", "top16", metric.AccuracyMetricType, topKAccuracyGraph(16), nil),
	}
}

func topKAccuracyGraph(k int) metric.BaseMetricGraph {
	return func(_ *model.Scope, labels, predictions []*graph.Node) *graph.Node {
		logits := predictions[0]
		label := graph.Squeeze(labels[0], -1)
		labelMask := graph.OneHot(label, logits.Shape().Dimensions[1], logits.DType())
		topKMask := graph.ConvertDType(graph.TopKMask(logits, k, -1), logits.DType())
		return graph.ReduceAllMean(graph.ReduceSum(graph.Mul(labelMask, topKMask), -1))
	}
}
