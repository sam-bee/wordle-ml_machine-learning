// Package proofgames owns the reusable, post-checkpoint complete-game proof
// path. It keeps game-engine evaluation out of proofrun and proofeval so both
// can call the same deterministic logic without an import cycle.
package proofgames

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/gomlx/gomlx/core/tensors"
	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
	"github.com/sam-bee/wordle-ml_machine-learning/supervised"
	"github.com/sam-bee/wordle-ml_machine-learning/tensorboard"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

// ScoreFunc returns finite raw logits for a production game position. The
// authoritative evaluator applies the legal/repeated-guess mask itself.
type ScoreFunc = gameeval.ScoreFunc
type Evaluation = gameeval.Evaluation
type Position = gameeval.Position

// Evaluate runs exactly the supplied prefix of the frozen validation solutions
// (10 or all 100), preserving its deterministic file order.
func Evaluate(ctx context.Context, vocab *vocabulary.Vocabulary, solutions []string, score ScoreFunc) (gameeval.Evaluation, error) {
	if vocab == nil || score == nil {
		return gameeval.Evaluation{}, errors.New("vocabulary and scorer are required")
	}
	if len(solutions) != 10 && len(solutions) != vocabulary.NumValidationSolutions {
		return gameeval.Evaluation{}, fmt.Errorf("fixed game evaluation needs 10 or 100 solutions, got %d", len(solutions))
	}
	if !slices.Equal(solutions, vocab.Validation()[:len(solutions)]) {
		return gameeval.Evaluation{}, errors.New("game solutions must be the fixed validation prefix")
	}
	evaluator, err := gameeval.New(gameeval.Config{Vocabulary: vocab, Score: score})
	if err != nil {
		return gameeval.Evaluation{}, err
	}
	evaluation, err := evaluator.Evaluate(ctx, solutions)
	if err != nil {
		return gameeval.Evaluation{}, err
	}
	if err := CheckTrajectories(evaluation); err != nil {
		return gameeval.Evaluation{}, err
	}
	return evaluation, nil
}

// EvaluateSession adapts a restored supervised session to the authoritative
// game evaluator. It returns only finite raw logits; legal/repeated-action
// selection deliberately remains inside gameeval.
func EvaluateSession(ctx context.Context, session *supervised.Session, vocab *vocabulary.Vocabulary, solutions []string) (Evaluation, error) {
	if session == nil {
		return Evaluation{}, errors.New("supervised session is required")
	}
	return Evaluate(ctx, vocab, solutions, func(_ context.Context, position Position) ([]float32, error) {
		rawTensor, maskedTensor, betaTensor, err := session.PredictDiagnostics(
			[][]float32{position.Inputs.CandidateMask}, [][]float32{position.Inputs.CandidateStats}, []int32{position.Inputs.Turn},
			[][]float32{position.Inputs.RemainingActionMask}, [][]float32{position.AvailableActionMask},
		)
		if err != nil {
			return nil, err
		}
		defer func() { _ = rawTensor.FinalizeAll(); _ = maskedTensor.FinalizeAll(); _ = betaTensor.FinalizeAll() }()
		raw, err := tensors.CopyFlatData[float32](rawTensor)
		if err != nil {
			return nil, err
		}
		for id, value := range raw {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, fmt.Errorf("raw game logit %d is non-finite", id)
			}
		}
		return raw, nil
	})
}

// EvaluateInitialGames10 runs the fixed first ten validation games through an
// independently loaded initial checkpoint. It is kept outside proofrun so the
// runner can call it for run-zero without importing proofeval.
func EvaluateInitialGames10(ctx context.Context, session *supervised.Session, vocab *vocabulary.Vocabulary) (Evaluation, error) {
	if vocab == nil {
		return Evaluation{}, errors.New("vocabulary is required")
	}
	return EvaluateSession(ctx, session, vocab, vocab.Validation()[:10])
}

// TensorBoardScalars returns the plan's deterministic game-summary event
// payload: solved fraction, mean guesses, failures, then guess counts 1..6.
func TensorBoardScalars(evaluation Evaluation) []tensorboard.Scalar {
	summary := evaluation.Summary
	scalars := make([]tensorboard.Scalar, 0, 3+len(summary.GuessCountDistribution))
	scalars = append(scalars,
		tensorboard.Scalar{Tag: "games/solved_fraction", Value: float32(summary.SolvedFraction)},
		tensorboard.Scalar{Tag: "games/mean_guesses", Value: float32(summary.MeanGuesses)},
		tensorboard.Scalar{Tag: "games/failures", Value: float32(summary.Failures)},
	)
	for index, count := range summary.GuessCountDistribution {
		scalars = append(scalars, tensorboard.Scalar{Tag: fmt.Sprintf("games/guess_count_%d", index+1), Value: float32(count)})
	}
	return scalars
}

// CheckTrajectories proves that accepted game turns contain no repeated guess.
// gameeval records InvalidSelections for a suppressed raw preference outside
// its allowed vocabulary; that is diagnostic data, not an accepted engine move.
// Every accepted guess has already been checked by the authoritative engine.
func CheckTrajectories(evaluation gameeval.Evaluation) error {
	for _, game := range evaluation.Games {
		seen := make(map[string]struct{}, len(game.Turns))
		for _, turn := range game.Turns {
			if _, duplicate := seen[turn.Guess]; duplicate {
				return fmt.Errorf("game %q repeated accepted guess %q", game.Solution, turn.Guess)
			}
			seen[turn.Guess] = struct{}{}
		}
	}
	return nil
}

// JSONL returns one complete JSON object and trailing newline per game, ready
// for the two atomic publications in a proof run.
func JSONL(games []gameeval.GameResult) ([]byte, error) {
	var buffer bytes.Buffer
	if err := gameeval.WriteJSONL(&buffer, games); err != nil {
		return nil, err
	}
	return buffer.Bytes(), nil
}
