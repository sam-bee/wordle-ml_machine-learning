package ppo

import (
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/policy"
	"github.com/sam-bee/wordle-ml_machine-learning/supervised"
)

// ActorConfig is the PPO-specific training wrapper around the unchanged
// supervised policy architecture.
type ActorConfig struct {
	Policy                     policy.Config
	LearningRate               float64
	ClipRange                  float64
	EntropyCoefficient         float64
	SupervisedReferenceKLCoeff float64
	MaximumGradientNorm        float64
	Seed                       int64
}

// CriticConfig configures the separately checkpointed scalar value network.
type CriticConfig struct {
	Policy               policy.Config
	LearningRate         float64
	ValueLossCoefficient float64
	MaximumGradientNorm  float64
	Seed                 int64
}

// ActorBatch contains exactly the stored-mask PPO quantities used by one
// minibatch. ReferenceLogits come from the permanently frozen supervised
// actor on these same observations and masks.
type ActorBatch struct {
	Inputs          []modelstate.Inputs
	Availability    [][]float32
	Actions         []int32
	OldLogProbs     []float32
	Advantages      []float32
	ReferenceLogits [][]float32
}

// CriticBatch contains value inputs and realised complete-game returns.
type CriticBatch struct {
	Inputs  []modelstate.Inputs
	Returns []float32
}

// ActorSession owns only the actor and its PPO Adam state. The critic and the
// frozen supervised reference live in separate stores.
type ActorSession struct {
	Policy    *policy.Model
	Store     *model.Store
	Trainer   *train.Trainer
	Inference *model.Exec
	config    ActorConfig
}

// CriticSession owns only the scalar critic and its Adam state.
type CriticSession struct {
	Critic    Critic
	Store     *model.Store
	Trainer   *train.Trainer
	Inference *model.Exec
	config    CriticConfig
}

// Predict returns one raw actor-logit row per encoded observation. Legality is
// intentionally applied by Categorical from the caller's exact stored mask.
func (s *ActorSession) Predict(inputs []modelstate.Inputs) ([][]float32, error) {
	if s == nil || s.Inference == nil {
		return nil, errors.New("actor inference session is nil or finalized")
	}
	if len(inputs) == 0 {
		return nil, errors.New("actor prediction batch is empty")
	}
	candidateMasks := make([][]float32, len(inputs))
	candidateStats := make([][]float32, len(inputs))
	turns := make([]int32, len(inputs))
	remainingMasks := make([][]float32, len(inputs))
	for index, input := range inputs {
		if err := validateActorInput(input, s.config.Policy); err != nil {
			return nil, fmt.Errorf("actor input %d: %w", index, err)
		}
		candidateMasks[index] = input.CandidateMask
		candidateStats[index] = input.CandidateStats
		turns[index] = input.Turn
		remainingMasks[index] = input.RemainingActionMask
	}
	result, err := s.Inference.Call1(candidateMasks, candidateStats, turns, remainingMasks)
	if err != nil {
		return nil, err
	}
	defer func() { _ = result.FinalizeAll() }()
	flat, err := tensors.CopyFlatData[float32](result)
	if err != nil {
		return nil, err
	}
	if len(flat) != len(inputs)*s.config.Policy.NumActions {
		return nil, fmt.Errorf("actor output has %d values, want %d", len(flat), len(inputs)*s.config.Policy.NumActions)
	}
	rows := make([][]float32, len(inputs))
	for index := range rows {
		start := index * s.config.Policy.NumActions
		rows[index] = append([]float32(nil), flat[start:start+s.config.Policy.NumActions]...)
	}
	return rows, nil
}

// TrainStep applies one PPO actor minibatch and returns its finite total actor
// loss (policy - entropy + supervised-reference KL penalty).
func (s *ActorSession) TrainStep(batch ActorBatch) (float64, error) {
	if s == nil || s.Trainer == nil {
		return 0, errors.New("actor trainer is nil or finalized")
	}
	materialized, err := s.materializeBatch(batch)
	if err != nil {
		return 0, err
	}
	metrics, err := s.Trainer.TrainStep(materialized)
	if err != nil {
		return 0, err
	}
	return firstFiniteMetric(metrics, "actor loss")
}

func (s *ActorSession) materializeBatch(batch ActorBatch) (train.Batch, error) {
	size := len(batch.Inputs)
	if size == 0 || len(batch.Availability) != size || len(batch.Actions) != size || len(batch.OldLogProbs) != size || len(batch.Advantages) != size || len(batch.ReferenceLogits) != size {
		return train.Batch{}, fmt.Errorf("actor batch lengths inputs=%d availability=%d actions=%d old_logp=%d advantages=%d reference=%d", size, len(batch.Availability), len(batch.Actions), len(batch.OldLogProbs), len(batch.Advantages), len(batch.ReferenceLogits))
	}
	candidateMasks := make([]float32, 0, size*s.config.Policy.NumSolutions)
	candidateStats := make([]float32, 0, size*modelstate.CandidateStatsSize)
	turns := make([]int32, size)
	remainingMasks := make([]float32, 0, size*s.config.Policy.NumActions)
	availability := make([]float32, 0, size*s.config.Policy.NumActions)
	referenceLogits := make([]float32, 0, size*s.config.Policy.NumActions)
	for index, input := range batch.Inputs {
		if err := validateActorInput(input, s.config.Policy); err != nil {
			return train.Batch{}, fmt.Errorf("actor batch input %d: %w", index, err)
		}
		if len(batch.Availability[index]) != s.config.Policy.NumActions || len(batch.ReferenceLogits[index]) != s.config.Policy.NumActions {
			return train.Batch{}, fmt.Errorf("actor batch row %d availability/reference lengths are %d/%d, want %d", index, len(batch.Availability[index]), len(batch.ReferenceLogits[index]), s.config.Policy.NumActions)
		}
		if batch.Actions[index] < 0 || int(batch.Actions[index]) >= s.config.Policy.NumActions || batch.Availability[index][batch.Actions[index]] != 1 {
			return train.Batch{}, fmt.Errorf("actor batch row %d selected unavailable/invalid action %d", index, batch.Actions[index])
		}
		if !finite64(float64(batch.OldLogProbs[index])) || !finite64(float64(batch.Advantages[index])) {
			return train.Batch{}, fmt.Errorf("actor batch row %d has non-finite old log-probability or advantage", index)
		}
		candidateMasks = append(candidateMasks, input.CandidateMask...)
		candidateStats = append(candidateStats, input.CandidateStats...)
		turns[index] = input.Turn
		remainingMasks = append(remainingMasks, input.RemainingActionMask...)
		availability = append(availability, batch.Availability[index]...)
		referenceLogits = append(referenceLogits, batch.ReferenceLogits[index]...)
	}
	return train.Batch{
		Inputs: []*tensors.Tensor{
			tensors.FromFlatDataAndDimensions(candidateMasks, size, s.config.Policy.NumSolutions),
			tensors.FromFlatDataAndDimensions(candidateStats, size, modelstate.CandidateStatsSize),
			tensors.FromFlatDataAndDimensions(turns, size),
			tensors.FromFlatDataAndDimensions(remainingMasks, size, s.config.Policy.NumActions),
			tensors.FromFlatDataAndDimensions(availability, size, s.config.Policy.NumActions),
		},
		Labels: []*tensors.Tensor{
			tensors.FromFlatDataAndDimensions(batch.Actions, size),
			tensors.FromFlatDataAndDimensions(batch.OldLogProbs, size),
			tensors.FromFlatDataAndDimensions(batch.Advantages, size),
			tensors.FromFlatDataAndDimensions(referenceLogits, size, s.config.Policy.NumActions),
		},
	}, nil
}

// Predict returns one scalar critic value for every complete actor-encoded
// state. The critic deliberately accepts modelstate.Inputs rather than a
// reduced representation, so it sees the same four state tensors as the
// actor: candidate mask, candidate statistics, turn, and action mask.
func (s *CriticSession) Predict(inputs []modelstate.Inputs) ([]float32, error) {
	if s == nil || s.Inference == nil {
		return nil, errors.New("critic inference session is nil or finalized")
	}
	if len(inputs) == 0 {
		return nil, errors.New("critic prediction batch is empty")
	}
	candidateMasks := make([][]float32, len(inputs))
	candidateStats := make([][]float32, len(inputs))
	turns := make([]int32, len(inputs))
	remainingActionMasks := make([][]float32, len(inputs))
	for index, input := range inputs {
		if err := validateActorInput(input, s.config.Policy); err != nil {
			return nil, fmt.Errorf("critic input %d: %w", index, err)
		}
		candidateMasks[index] = input.CandidateMask
		candidateStats[index] = input.CandidateStats
		turns[index] = input.Turn
		remainingActionMasks[index] = input.RemainingActionMask
	}
	result, err := s.Inference.Call1(candidateMasks, candidateStats, turns, remainingActionMasks)
	if err != nil {
		return nil, err
	}
	defer func() { _ = result.FinalizeAll() }()
	values, err := tensors.CopyFlatData[float32](result)
	if err != nil {
		return nil, err
	}
	if len(values) != len(inputs) {
		return nil, fmt.Errorf("critic output has %d values, want %d", len(values), len(inputs))
	}
	for index, value := range values {
		if !finite64(float64(value)) {
			return nil, fmt.Errorf("critic value %d is not finite: %v", index, value)
		}
	}
	return values, nil
}

// TrainStep applies one value-regression minibatch.
func (s *CriticSession) TrainStep(batch CriticBatch) (float64, error) {
	if s == nil || s.Trainer == nil {
		return 0, errors.New("critic trainer is nil or finalized")
	}
	size := len(batch.Inputs)
	if size == 0 || len(batch.Returns) != size {
		return 0, fmt.Errorf("critic batch lengths inputs=%d returns=%d", size, len(batch.Returns))
	}
	flatCandidateMasks := make([]float32, 0, size*s.config.Policy.NumSolutions)
	flatStats := make([]float32, 0, size*modelstate.CandidateStatsSize)
	turns := make([]int32, size)
	flatRemainingActionMasks := make([]float32, 0, size*s.config.Policy.NumActions)
	for index, input := range batch.Inputs {
		if err := validateActorInput(input, s.config.Policy); err != nil || !finite64(float64(batch.Returns[index])) {
			return 0, fmt.Errorf("critic batch row %d is malformed", index)
		}
		flatCandidateMasks = append(flatCandidateMasks, input.CandidateMask...)
		flatStats = append(flatStats, input.CandidateStats...)
		turns[index] = input.Turn
		flatRemainingActionMasks = append(flatRemainingActionMasks, input.RemainingActionMask...)
	}
	metrics, err := s.Trainer.TrainStep(train.Batch{
		Inputs: []*tensors.Tensor{
			tensors.FromFlatDataAndDimensions(flatCandidateMasks, size, s.config.Policy.NumSolutions),
			tensors.FromFlatDataAndDimensions(flatStats, size, modelstate.CandidateStatsSize),
			tensors.FromFlatDataAndDimensions(turns, size),
			tensors.FromFlatDataAndDimensions(flatRemainingActionMasks, size, s.config.Policy.NumActions),
		},
		Labels: []*tensors.Tensor{
			tensors.FromFlatDataAndDimensions(batch.Returns, size),
		},
	})
	if err != nil {
		return 0, err
	}
	return firstFiniteMetric(metrics, "critic loss")
}

func firstFiniteMetric(metrics []*tensors.Tensor, name string) (float64, error) {
	defer func() {
		for _, metric := range metrics {
			_ = metric.FinalizeAll()
		}
	}()
	if len(metrics) == 0 {
		return 0, fmt.Errorf("%s produced no metrics", name)
	}
	values, err := tensors.CopyFlatData[float32](metrics[0])
	if err != nil || len(values) != 1 {
		if err == nil {
			err = fmt.Errorf("metric has %d values", len(values))
		}
		return 0, fmt.Errorf("read %s: %w", name, err)
	}
	value := float64(values[0])
	if !finite64(value) {
		return 0, fmt.Errorf("%s is not finite: %v", name, value)
	}
	return value, nil
}

func validateActorInput(input modelstate.Inputs, config policy.Config) error {
	if len(input.CandidateMask) != config.NumSolutions || len(input.CandidateStats) != modelstate.CandidateStatsSize || len(input.RemainingActionMask) != config.NumActions || input.Turn < 0 || input.Turn >= criticNumberOfTurns {
		return fmt.Errorf("dimensions candidate=%d stats=%d remaining=%d turn=%d", len(input.CandidateMask), len(input.CandidateStats), len(input.RemainingActionMask), input.Turn)
	}
	return nil
}

// NewActorSession creates a fresh actor wrapper. LearningRate zero is a
// deliberate mechanical-test mode: the full loss graph executes but a no-op
// optimiser leaves every actor parameter byte unchanged.
func NewActorSession(config ActorConfig, backend compute.Backend) (*ActorSession, error) {
	if backend == nil {
		return nil, errors.New("actor backend must not be nil")
	}
	if config.Seed == 0 {
		return nil, errors.New("actor seed must be non-zero")
	}
	if config.LearningRate < 0 || !finite64(config.LearningRate) {
		return nil, fmt.Errorf("actor learning rate must be finite and non-negative, got %g", config.LearningRate)
	}
	if config.ClipRange < 0 || config.ClipRange >= 1 || !finite64(config.ClipRange) {
		return nil, fmt.Errorf("actor clip range must be finite in [0,1), got %g", config.ClipRange)
	}
	if config.EntropyCoefficient < 0 || !finite64(config.EntropyCoefficient) || config.SupervisedReferenceKLCoeff < 0 || !finite64(config.SupervisedReferenceKLCoeff) {
		return nil, errors.New("actor entropy and supervised-reference KL coefficients must be finite and non-negative")
	}
	if config.MaximumGradientNorm <= 0 || !finite64(config.MaximumGradientNorm) {
		return nil, fmt.Errorf("actor maximum gradient norm must be finite and positive, got %g", config.MaximumGradientNorm)
	}
	actor, err := policy.New(config.Policy)
	if err != nil {
		return nil, err
	}
	store := model.NewStore()
	store.SetParam(model.ParamInitialSeed, config.Seed)
	session := &ActorSession{Policy: actor, Store: store, config: config}
	var actorOptimizer optimizer.Interface
	if config.LearningRate == 0 {
		actorOptimizer = noUpdateOptimizer{}
	} else {
		actorOptimizer = supervised.NewAdamWithGlobalNormClip(config.LearningRate, config.MaximumGradientNorm)
	}
	session.Trainer = train.NewTrainer(
		backend,
		store,
		session.trainingForward,
		session.actorLoss,
		actorOptimizer,
		nil,
		nil,
	)
	session.Inference, err = model.NewExec(backend, store, session.inferenceGraph)
	if err != nil {
		session.Finalize()
		return nil, fmt.Errorf("create actor inference executor: %w", err)
	}
	return session, nil
}

func (s *ActorSession) trainingForward(scope *model.Scope, inputs []*graph.Node) (*graph.Node, *graph.Node) {
	if len(inputs) != 5 {
		panic(fmt.Sprintf("PPO actor needs five inputs, got %d", len(inputs)))
	}
	return s.Policy.Forward(scope, inputs[0], inputs[1], inputs[2], inputs[3]), inputs[4]
}

func (s *ActorSession) inferenceGraph(scope *model.Scope, candidateMask, candidateStats, turn, remainingActionMask *graph.Node) *graph.Node {
	scope.Store().SetTraining(candidateMask.Graph(), false)
	return s.Policy.Forward(scope, candidateMask, candidateStats, turn, remainingActionMask)
}

// actorLoss is standard clipped PPO plus entropy regularisation and exact
// categorical KL(reference || current), always under the stored action mask.
func (s *ActorSession) actorLoss(labels, predictions []*graph.Node) *graph.Node {
	if len(labels) != 4 || len(predictions) != 2 {
		panic(fmt.Sprintf("PPO actor loss needs four labels and two predictions, got %d/%d", len(labels), len(predictions)))
	}
	rawLogits := predictions[0]
	availability := predictions[1]
	actions, oldLogProbs, advantages, referenceLogits := labels[0], labels[1], labels[2], labels[3]
	legal := graph.GreaterThan(availability, graph.ScalarZero(availability.Graph(), availability.DType()))
	currentLogProbs := graph.MaskedLogSoftmax(rawLogits, legal, -1)
	referenceLogProbs := graph.MaskedLogSoftmax(referenceLogits, legal, -1)
	finiteFloor := graph.Scalar(rawLogits.Graph(), rawLogits.DType(), -1e30)
	safeCurrentLogProbs := graph.Where(legal, currentLogProbs, finiteFloor)
	safeReferenceLogProbs := graph.Where(legal, referenceLogProbs, finiteFloor)
	oneHotActions := graph.OneHot(actions, s.config.Policy.NumActions, rawLogits.DType())
	selectedLogProbs := graph.ReduceSum(graph.Mul(safeCurrentLogProbs, oneHotActions), -1)
	ratio := graph.Exp(graph.Sub(selectedLogProbs, oldLogProbs))
	unclipped := graph.Mul(ratio, advantages)
	clipped := graph.Mul(graph.ClipScalar(ratio, 1-s.config.ClipRange, 1+s.config.ClipRange), advantages)
	policyLoss := graph.Neg(graph.ReduceAllMean(graph.Min(unclipped, clipped)))

	currentProbabilities := graph.MaskedSoftmax(rawLogits, legal, -1)
	entropy := graph.Neg(graph.ReduceAllMean(graph.ReduceSum(graph.Mul(currentProbabilities, safeCurrentLogProbs), -1)))
	referenceProbabilities := graph.MaskedSoftmax(referenceLogits, legal, -1)
	referenceKL := graph.ReduceAllMean(graph.ReduceSum(
		graph.Mul(referenceProbabilities, graph.Sub(safeReferenceLogProbs, safeCurrentLogProbs)),
		-1,
	))
	return graph.Add(
		graph.Sub(policyLoss, graph.MulScalar(entropy, s.config.EntropyCoefficient)),
		graph.MulScalar(referenceKL, s.config.SupervisedReferenceKLCoeff),
	)
}

// Finalize releases actor executors and store resources.
func (s *ActorSession) Finalize() {
	if s == nil {
		return
	}
	if s.Inference != nil {
		s.Inference.Finalize()
		s.Inference = nil
	}
	if s.Trainer != nil {
		s.Trainer.ResetComputationGraphs()
		s.Trainer = nil
	}
	if s.Store != nil {
		s.Store.Finalize()
		s.Store = nil
	}
	s.Policy = nil
}

// NewCriticSession creates the separately checkpointed value estimator.
func NewCriticSession(config CriticConfig, backend compute.Backend) (*CriticSession, error) {
	if backend == nil {
		return nil, errors.New("critic backend must not be nil")
	}
	if config.Seed == 0 || config.LearningRate <= 0 || !finite64(config.LearningRate) || config.ValueLossCoefficient <= 0 || !finite64(config.ValueLossCoefficient) || config.MaximumGradientNorm <= 0 || !finite64(config.MaximumGradientNorm) {
		return nil, errors.New("critic seed, learning rate, value coefficient, and maximum gradient norm must be valid and positive")
	}
	if _, err := policy.New(config.Policy); err != nil {
		return nil, fmt.Errorf("critic policy dimensions: %w", err)
	}
	store := model.NewStore()
	store.SetParam(model.ParamInitialSeed, config.Seed)
	session := &CriticSession{Critic: Critic{config: config.Policy}, Store: store, config: config}
	session.Trainer = train.NewTrainer(
		backend,
		store,
		session.trainingForward,
		session.criticLoss,
		supervised.NewAdamWithGlobalNormClip(config.LearningRate, config.MaximumGradientNorm),
		nil,
		nil,
	)
	var err error
	session.Inference, err = model.NewExec(backend, store, session.inferenceGraph)
	if err != nil {
		session.Finalize()
		return nil, fmt.Errorf("create critic inference executor: %w", err)
	}
	return session, nil
}

func (s *CriticSession) trainingForward(scope *model.Scope, inputs []*graph.Node) *graph.Node {
	if len(inputs) != 4 {
		panic(fmt.Sprintf("critic needs the four encoded actor inputs, got %d inputs", len(inputs)))
	}
	return s.Critic.Forward(scope, inputs[0], inputs[1], inputs[2], inputs[3])
}

func (s *CriticSession) inferenceGraph(scope *model.Scope, candidateMask, stats, turn, remainingActionMask *graph.Node) *graph.Node {
	scope.Store().SetTraining(stats.Graph(), false)
	return s.Critic.Forward(scope, candidateMask, stats, turn, remainingActionMask)
}

func (s *CriticSession) criticLoss(labels, predictions []*graph.Node) *graph.Node {
	if len(labels) != 1 || len(predictions) != 1 {
		panic(fmt.Sprintf("critic loss needs one label and prediction, got %d/%d", len(labels), len(predictions)))
	}
	errorNode := graph.Sub(predictions[0], labels[0])
	return graph.MulScalar(graph.ReduceAllMean(graph.Square(errorNode)), s.config.ValueLossCoefficient)
}

// Finalize releases critic executors and store resources.
func (s *CriticSession) Finalize() {
	if s == nil {
		return
	}
	if s.Inference != nil {
		s.Inference.Finalize()
		s.Inference = nil
	}
	if s.Trainer != nil {
		s.Trainer.ResetComputationGraphs()
		s.Trainer = nil
	}
	if s.Store != nil {
		s.Store.Finalize()
		s.Store = nil
	}
}

// noUpdateOptimizer constructs the complete differentiable loss graph without
// mutating parameters. It exists solely for the required zero-learning-rate
// mechanical correctness run.
type noUpdateOptimizer struct{}

func (noUpdateOptimizer) UpdateGraph(_ *model.Scope, _ *graph.Graph, loss *graph.Node) {
	if !loss.Shape().IsScalar() {
		panic(fmt.Sprintf("no-update optimiser requires scalar loss, got %s", loss.Shape()))
	}
}
func (noUpdateOptimizer) Clear(_ *model.Scope) error { return nil }

// LoadBaselineActor clones only the proven `/wordle_policy` tensors from a
// supervised checkpoint. It intentionally imports no supervised Adam moments,
// diagnostics, cursor, or global-step state.
func LoadBaselineActor(checkpointDir string, targets ...*ActorSession) error {
	if strings.TrimSpace(checkpointDir) == "" || len(targets) == 0 {
		return errors.New("baseline checkpoint and at least one actor target are required")
	}
	temporary := model.NewStore()
	defer temporary.Finalize()
	if _, err := checkpoint.Load(temporary).Dir(checkpointDir).Immediate().Done(); err != nil {
		return fmt.Errorf("load supervised baseline checkpoint: %w", err)
	}
	actorVariables := make([]*model.Variable, 0)
	for variable := range temporary.IterVariables() {
		if strings.HasPrefix(variable.Path(), actorVariablePrefix) {
			actorVariables = append(actorVariables, variable)
		}
	}
	if len(actorVariables) == 0 {
		return errors.New("supervised checkpoint contains no /wordle_policy variables")
	}
	sort.Slice(actorVariables, func(i, j int) bool { return actorVariables[i].Path() < actorVariables[j].Path() })
	for targetIndex, target := range targets {
		if target == nil || target.Store == nil {
			return fmt.Errorf("actor target %d is nil or finalized", targetIndex)
		}
		for _, variable := range actorVariables {
			if target.Store.GetVariable(variable.Path()) != nil {
				return fmt.Errorf("actor target %d already contains %s", targetIndex, variable.Path())
			}
			clone, err := variable.CloneToStore(target.Store)
			if err != nil {
				return fmt.Errorf("clone baseline actor variable %s: %w", variable.Path(), err)
			}
			clone.SetTrainable(true)
		}
	}
	return nil
}

// LoadActorCheckpoint attaches a PPO actor checkpoint, including exact Adam
// and trainer/global-step state, to a fresh ActorSession.
func LoadActorCheckpoint(session *ActorSession, dir string) error {
	if session == nil || session.Store == nil {
		return errors.New("actor session is nil or finalized")
	}
	_, err := checkpoint.Load(session.Store).Dir(dir).Done()
	if err != nil {
		return fmt.Errorf("load PPO actor checkpoint: %w", err)
	}
	return nil
}

// LoadCriticCheckpoint attaches a complete critic checkpoint to a fresh
// CriticSession.
func LoadCriticCheckpoint(session *CriticSession, dir string) error {
	if session == nil || session.Store == nil {
		return errors.New("critic session is nil or finalized")
	}
	_, err := checkpoint.Load(session.Store).Dir(dir).Done()
	if err != nil {
		return fmt.Errorf("load PPO critic checkpoint: %w", err)
	}
	return nil
}

// SaveStoreCheckpoint saves every materialized model/optimizer variable and
// Store parameter to a fresh isolated checkpoint directory.
func SaveStoreCheckpoint(store *model.Store, dir string) error {
	if store == nil {
		return errors.New("cannot checkpoint nil Store")
	}
	handler, err := checkpoint.Build(store).Dir(dir).Keep(1).Done()
	if err != nil {
		return err
	}
	return handler.Save()
}

// ExportActorOnly writes only `/wordle_policy` tensors to a deployment
// checkpoint. It deliberately contains no critic or optimiser dependency.
func ExportActorOnly(actor *ActorSession, dir string) error {
	if actor == nil || actor.Store == nil {
		return errors.New("cannot export nil or finalized actor")
	}
	export := model.NewStore()
	defer export.Finalize()
	count := 0
	for variable := range actor.Store.IterVariables() {
		if !strings.HasPrefix(variable.Path(), actorVariablePrefix) {
			continue
		}
		clone, err := variable.CloneToStore(export)
		if err != nil {
			return fmt.Errorf("clone actor-only variable %s: %w", variable.Path(), err)
		}
		clone.SetTrainable(variable.Trainable)
		count++
	}
	if count == 0 {
		return errors.New("actor has no materialized policy variables to export")
	}
	return SaveStoreCheckpoint(export, dir)
}

// ActorParameterDeltaNorm returns the global L2 distance between two actor
// parameter sets with identical `/wordle_policy` paths.
func ActorParameterDeltaNorm(left, right *ActorSession) (float64, error) {
	if left == nil || right == nil || left.Store == nil || right.Store == nil {
		return 0, errors.New("actor delta requires two live sessions")
	}
	leftValues, err := variableValues(left.Store, actorVariablePrefix)
	if err != nil {
		return 0, err
	}
	rightValues, err := variableValues(right.Store, actorVariablePrefix)
	if err != nil {
		return 0, err
	}
	if len(leftValues) != len(rightValues) {
		return 0, fmt.Errorf("actor variable counts differ: %d/%d", len(leftValues), len(rightValues))
	}
	var scale, normalizedSquares float64
	for path, leftRow := range leftValues {
		rightRow, found := rightValues[path]
		if !found || len(leftRow) != len(rightRow) {
			return 0, fmt.Errorf("actor variable %s differs in shape or presence", path)
		}
		for index, leftValue := range leftRow {
			delta := math.Abs(float64(leftValue - rightRow[index]))
			if delta == 0 {
				continue
			}
			if scale < delta {
				if scale != 0 {
					ratio := scale / delta
					normalizedSquares *= ratio * ratio
				}
				scale = delta
			}
			ratio := delta / scale
			normalizedSquares += ratio * ratio
		}
	}
	return scale * math.Sqrt(normalizedSquares), nil
}

func variableValues(store *model.Store, prefix string) (map[string][]float32, error) {
	result := make(map[string][]float32)
	for variable := range store.IterVariables() {
		if !variable.Trainable || !strings.HasPrefix(variable.Path(), prefix) {
			continue
		}
		values, err := tensors.CopyFlatData[float32](variable.MustValue())
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", variable.Path(), err)
		}
		result[variable.Path()] = values
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("no trainable variables under %s", prefix)
	}
	return result, nil
}

// VariableChecksum hashes variable paths, shapes and FP32 values for a stable
// actor/critic identity and exact immutability assertions.
func VariableChecksum(store *model.Store, prefix string) (string, error) {
	if store == nil {
		return "", errors.New("cannot checksum nil Store")
	}
	variables := make([]*model.Variable, 0)
	for variable := range store.IterVariables() {
		if variable.Trainable && strings.HasPrefix(variable.Path(), prefix) {
			variables = append(variables, variable)
		}
	}
	if len(variables) == 0 {
		return "", fmt.Errorf("no trainable variables under %s", prefix)
	}
	sort.Slice(variables, func(i, j int) bool { return variables[i].Path() < variables[j].Path() })
	hasher := sha256.New()
	var bytes [8]byte
	for _, variable := range variables {
		_, _ = hasher.Write([]byte(variable.Path()))
		values, err := tensors.CopyFlatData[float32](variable.MustValue())
		if err != nil {
			return "", fmt.Errorf("read %s: %w", variable.Path(), err)
		}
		binary.LittleEndian.PutUint64(bytes[:], uint64(len(values)))
		_, _ = hasher.Write(bytes[:])
		for _, value := range values {
			if !finite64(float64(value)) {
				return "", fmt.Errorf("parameter %s is not finite", variable.Path())
			}
			var word [4]byte
			binary.LittleEndian.PutUint32(word[:], math.Float32bits(value))
			_, _ = hasher.Write(word[:])
		}
	}
	return fmt.Sprintf("%x", hasher.Sum(nil)), nil
}

// ParametersFinite scans all materialized trainable tensors. Optimizer
// gradient finiteness is reported separately by supervised diagnostics.
func ParametersFinite(store *model.Store) error {
	if store == nil {
		return errors.New("cannot inspect nil Store")
	}
	for variable := range store.IterVariables() {
		if !variable.Trainable {
			continue
		}
		values, err := tensors.CopyFlatData[float32](variable.MustValue())
		if err != nil {
			return fmt.Errorf("read parameter %s: %w", variable.Path(), err)
		}
		for index, value := range values {
			if !finite64(float64(value)) {
				return fmt.Errorf("parameter %s[%d] is not finite: %v", variable.Path(), index, value)
			}
		}
	}
	return nil
}

func finite64(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

var _ optimizer.Interface = noUpdateOptimizer{}
