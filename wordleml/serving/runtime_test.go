package serving

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

func TestRuntimePlayAcceptsOnlyValidationSolutions(t *testing.T) {
	vocab, err := vocabulary.Load("../../data")
	if err != nil {
		t.Fatal(err)
	}
	solution := vocab.Validation()[0]
	actionID, found := vocab.ActionID(solution)
	if !found {
		t.Fatalf("validation solution %q has no action ID", solution)
	}
	evaluator, err := gameeval.New(gameeval.Config{
		Vocabulary: vocab,
		Score: func(context.Context, gameeval.Position) ([]float32, error) {
			scores := make([]float32, vocabulary.NumActions)
			scores[actionID] = 1
			return scores, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	runtime := &Runtime{
		evaluator: evaluator,
		solutions: vocab.Validation(),
		allowed:   make(map[string]struct{}, vocabulary.NumValidationSolutions),
		gate:      make(chan struct{}, 1),
	}
	for _, word := range runtime.solutions {
		runtime.allowed[word] = struct{}{}
	}
	game, err := runtime.Play(context.Background(), strings.ToLower(solution))
	if err != nil {
		t.Fatal(err)
	}
	if !game.Solved || game.Guesses != 1 || game.Solution != solution {
		t.Fatalf("game = %+v", game)
	}
	if _, err := runtime.Play(context.Background(), vocab.Training()[0]); !errors.Is(err, ErrInvalidSolution) {
		t.Fatalf("training solution error = %v", err)
	}
	if _, err := runtime.Play(context.Background(), vocab.Test()[0]); !errors.Is(err, ErrInvalidSolution) {
		t.Fatalf("test solution error = %v", err)
	}
}
