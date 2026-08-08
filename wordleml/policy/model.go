// Package policy implements the GoMLX Wordle policy network.
package policy

import (
	"fmt"
	"strings"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	gomlxmodel "github.com/gomlx/gomlx/ml/model"
)

const (
	// CandidateStatsSize is the number of precomputed candidate-set statistics.
	CandidateStatsSize = 209

	numTurns                = 6
	candidateProjectionSize = 96
	statsProjectionSize     = 48
	turnEmbeddingSize       = 16
	trunkSize               = 160
	modelScopeName          = "wordle_policy"
)

// Config contains the vocabulary-dependent dimensions of the policy.
type Config struct {
	NumSolutions int
	NumActions   int
}

// Model is the fixed Wordle policy architecture for a configured vocabulary.
type Model struct {
	config Config
}

// New validates config and constructs a policy model.
func New(config Config) (*Model, error) {
	if config.NumSolutions <= 0 {
		return nil, fmt.Errorf("number of solutions must be positive, got %d", config.NumSolutions)
	}
	if config.NumActions <= 0 {
		return nil, fmt.Errorf("number of actions must be positive, got %d", config.NumActions)
	}
	return &Model{config: config}, nil
}

// MustNew is like New, but panics if config is invalid.
func MustNew(config Config) *Model {
	model, err := New(config)
	if err != nil {
		panic(err)
	}
	return model
}

// Config returns the vocabulary dimensions used to construct the model.
func (m *Model) Config() Config {
	return m.config
}

// Forward builds the policy graph and returns raw logits with shape [batch, NumActions].
//
// candidateMask, candidateStats, and remainingActionMask must be FP32. turn must
// be an integer tensor containing values from 0 through 5. remainingActionMask
// adds a learned candidate bonus; it is deliberately not a legality mask.
func (m *Model) Forward(
	scope *gomlxmodel.Scope,
	candidateMask, candidateStats, turn, remainingActionMask *graph.Node,
) *graph.Node {
	logits, _ := m.ForwardWithBeta(scope, candidateMask, candidateStats, turn, remainingActionMask)
	return logits
}

// ForwardWithBeta builds the policy graph and returns its raw action logits
// together with the learned per-example candidate bonus. Training and host-side
// diagnostics use the latter to inspect whether the candidate branch is being
// used; it does not alter the policy's action space or masking semantics.
func (m *Model) ForwardWithBeta(
	scope *gomlxmodel.Scope,
	candidateMask, candidateStats, turn, remainingActionMask *graph.Node,
) (logits, beta *graph.Node) {
	m.validateInputs(candidateMask, candidateStats, turn, remainingActionMask)

	scope = scope.In(modelScopeName)

	candidateCount := graph.InsertAxes(graph.ReduceSum(candidateMask, -1), -1)
	// Game evaluation normally supplies at least one surviving candidate. Keep
	// raw logits finite even for a defensive empty/terminal state: dividing an
	// all-zero mask by zero would otherwise poison every downstream diagnostic
	// with NaNs. This leaves every non-empty mask's mean pooling unchanged.
	normalizedCandidateMask := graph.Div(candidateMask, graph.Max(candidateCount, graph.ScalarOne(candidateMask.Graph(), dtypes.Float32)))
	candidateFeatures := activation.Relu(layers.DenseWithBias(
		scope.In("candidate_projection"),
		normalizedCandidateMask,
		candidateProjectionSize,
	))

	statsFeatures := activation.Relu(layers.DenseWithBias(
		scope.In("stats_projection"),
		candidateStats,
		statsProjectionSize,
	))

	turnFeatures := layers.Embedding(
		scope.In("turn_embedding"),
		graph.InsertAxes(turn, -1),
		dtypes.Float32,
		numTurns,
		turnEmbeddingSize,
	)

	h := graph.Concatenate([]*graph.Node{candidateFeatures, statsFeatures, turnFeatures}, -1)
	r := activation.Relu(layers.DenseWithBias(scope.In("residual_in"), h, trunkSize))
	r = layers.DenseWithBias(scope.In("residual_out"), r, trunkSize)
	h = activation.Relu(graph.Add(h, r))

	baseLogits := layers.DenseWithBias(scope.In("base_logits"), h, m.config.NumActions)
	beta = layers.DenseWithBias(scope.In("candidate_bonus"), h, 1)
	return graph.Add(baseLogits, graph.Mul(beta, remainingActionMask)), beta
}

func (m *Model) validateInputs(candidateMask, candidateStats, turn, remainingActionMask *graph.Node) {
	if err := candidateMask.Shape().Check(dtypes.Float32, -1, m.config.NumSolutions); err != nil {
		panic(fmt.Errorf("candidateMask: %w", err))
	}
	batchSize := candidateMask.Shape().Dimensions[0]
	if err := candidateStats.Shape().Check(dtypes.Float32, batchSize, CandidateStatsSize); err != nil {
		panic(fmt.Errorf("candidateStats: %w", err))
	}
	if !turn.DType().IsInt() {
		panic(fmt.Errorf("turn: dtype must be an integer, got %s", turn.DType()))
	}
	if err := turn.Shape().CheckDims(batchSize); err != nil {
		panic(fmt.Errorf("turn: %w", err))
	}
	if err := remainingActionMask.Shape().Check(dtypes.Float32, batchSize, m.config.NumActions); err != nil {
		panic(fmt.Errorf("remainingActionMask: %w", err))
	}
}

// TrainableParameterCount returns the number of trainable scalar parameters
// materialized below scope by a graph build.
func TrainableParameterCount(scope *gomlxmodel.Scope) int {
	total := 0
	for variable := range scope.IterVariables() {
		if variable.Trainable {
			total += variable.Shape().Size()
		}
	}
	return total
}

// TrainableParameterBytes returns the FP32 weight storage used by the trainable
// variables materialized below scope. It panics if a trainable variable is not FP32.
func TrainableParameterBytes(scope *gomlxmodel.Scope) int64 {
	var total int64
	for variable := range scope.IterVariables() {
		if !variable.Trainable {
			continue
		}
		if variable.DType() != dtypes.Float32 {
			panic(fmt.Errorf("trainable variable %s is %s, want float32", strings.TrimPrefix(variable.Path(), "/"), variable.DType()))
		}
		total += variable.Shape().ByteSize()
	}
	return total
}
