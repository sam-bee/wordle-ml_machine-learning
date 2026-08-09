package ppo

import (
	"fmt"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/policy"
)

const (
	criticScope          = "ppo_critic"
	criticStatsWidth     = 64
	criticTurnWidth      = 8
	criticHiddenWidth    = 64
	criticNumberOfTurns  = 6
	actorVariablePrefix  = "/wordle_policy/"
	criticVariablePrefix = "/ppo_critic/"
)

// Critic is deliberately separate from the proven actor checkpoint. It takes
// every tensor in the established actor encoding rather than a new board
// representation or any environment-side answer information.
type Critic struct{ config policy.Config }

// Forward returns one scalar value estimate per encoded state.
func (critic Critic) Forward(scope *model.Scope, candidateMask, candidateStats, turn, remainingActionMask *graph.Node) *graph.Node {
	if err := candidateMask.Shape().Check(dtypes.Float32, -1, critic.config.NumSolutions); err != nil {
		panic(fmt.Errorf("critic candidate mask: %w", err))
	}
	if err := candidateStats.Shape().Check(dtypes.Float32, -1, modelstate.CandidateStatsSize); err != nil {
		panic(fmt.Errorf("critic candidate statistics: %w", err))
	}
	if !turn.DType().IsInt() {
		panic(fmt.Errorf("critic turn dtype must be integer, got %s", turn.DType()))
	}
	batch := candidateStats.Shape().Dimensions[0]
	if err := turn.Shape().CheckDims(batch); err != nil {
		panic(fmt.Errorf("critic turn: %w", err))
	}
	if err := remainingActionMask.Shape().Check(dtypes.Float32, batch, critic.config.NumActions); err != nil {
		panic(fmt.Errorf("critic remaining-action mask: %w", err))
	}

	scope = scope.In(criticScope)
	statsFeatures := activation.Relu(layers.DenseWithBias(scope.In("stats_projection"), candidateStats, criticStatsWidth))
	turnFeatures := layers.Embedding(
		scope.In("turn_embedding"),
		graph.InsertAxes(turn, -1),
		dtypes.Float32,
		criticNumberOfTurns,
		criticTurnWidth,
	)
	// These two scalar summaries make the separate critic consume every tensor
	// in the established four-input actor encoding without introducing a second
	// state representation or a large duplicate mask projection.
	candidateFraction := graph.InsertAxes(graph.ReduceMean(candidateMask, -1), -1)
	remainingActionFraction := graph.InsertAxes(graph.ReduceMean(remainingActionMask, -1), -1)
	hidden := graph.Concatenate([]*graph.Node{statsFeatures, turnFeatures, candidateFraction, remainingActionFraction}, -1)
	hidden = activation.Relu(layers.DenseWithBias(scope.In("hidden"), hidden, criticHiddenWidth))
	value := layers.DenseWithBias(scope.In("value"), hidden, 1)
	return graph.Squeeze(value, -1)
}
