package gameeval

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

func TestEvaluateUsesAuthoritativeFeedbackAndCandidateShortlist(t *testing.T) {
	v := testVocabulary(t)
	evaluator := newEvaluator(t, v,
		[]string{"GEESE", "EERIE", "CIGAR", "SPEED"},
		[]string{"GEESE", "CIGAR", "SPEED"},
		func(_ context.Context, position Position) ([]float32, error) {
			if position.Inputs.Turn != int32(position.Turn) {
				return nil, fmt.Errorf("encoded turn = %d, position turn = %d", position.Inputs.Turn, position.Turn)
			}
			if position.Turn == 0 {
				if got := len(position.CandidateSolutions); got != 3 {
					return nil, fmt.Errorf("opening candidates = %d, want 3", got)
				}
			} else {
				if got := position.CandidateSolutions; !reflect.DeepEqual(got, []string{"GEESE"}) {
					return nil, fmt.Errorf("post-feedback candidates = %v, want [GEESE]", got)
				}
				eerieID, _ := v.ActionID("EERIE")
				if position.AvailableActionMask[eerieID] != 0 {
					return nil, errors.New("prior game guess was not masked")
				}
			}
			// A scorer receives a private position copy. Its accidental mutation
			// must not change the evaluator's authoritative legality mask.
			for actionID := range position.AvailableActionMask {
				position.AvailableActionMask[actionID] = 0
			}
			if position.Turn == 0 {
				return scoresFor(v, "EERIE"), nil
			}
			return scoresFor(v, "GEESE"), nil
		},
	)

	result, err := evaluator.Evaluate(context.Background(), []string{"GEESE"})
	if err != nil {
		t.Fatal(err)
	}
	game := result.Games[0]
	if !game.Solved || game.Guesses != 2 {
		t.Fatalf("game solved=%t guesses=%d, want true/2", game.Solved, game.Guesses)
	}
	if got, want := game.Turns[0].Feedback, "YG--G"; got != want {
		t.Fatalf("first feedback = %q, want %q", got, want)
	}
	if got, want := game.Turns[0].ShortlistSizeBefore, 3; got != want {
		t.Fatalf("first shortlist before = %d, want %d", got, want)
	}
	if got, want := game.Turns[0].ShortlistSizeAfter, 1; got != want {
		t.Fatalf("first shortlist after = %d, want %d", got, want)
	}
	if got, want := game.Turns[1].ShortlistSizeBefore, 1; got != want {
		t.Fatalf("second shortlist before = %d, want %d", got, want)
	}
	if got := result.Summary.GuessCountDistribution; got != [6]int{0, 1, 0, 0, 0, 0} {
		t.Fatalf("guess distribution = %v, want one two-guess game", got)
	}
}

func TestEvaluateMasksInvalidAndRepeatedRawSelections(t *testing.T) {
	v := testVocabulary(t)
	evaluator := newEvaluator(t, v,
		[]string{"CIGAR", "REBUT"},
		[]string{"CIGAR", "REBUT"},
		func(_ context.Context, position Position) ([]float32, error) {
			if position.Turn == 0 {
				// SISSY belongs to the global action vocabulary but not this
				// game's valid-guess list. REBUT is the legal fallback.
				return rankedScores(v, "SISSY", "REBUT"), nil
			}
			// REBUT is now repeated and must be masked; CIGAR is selected.
			return rankedScores(v, "REBUT", "CIGAR"), nil
		},
	)

	result, err := evaluator.Evaluate(context.Background(), []string{"CIGAR"})
	if err != nil {
		t.Fatal(err)
	}
	game := result.Games[0]
	if got, want := []string{game.Turns[0].Guess, game.Turns[1].Guess}, []string{"REBUT", "CIGAR"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("accepted guesses = %v, want %v", got, want)
	}
	if game.InvalidSelections != 1 || game.RepeatedSelections != 1 || game.SuppressedRawTopSelections != 2 {
		t.Fatalf("selection counts = invalid=%d repeated=%d suppressed=%d, want 1/1/2", game.InvalidSelections, game.RepeatedSelections, game.SuppressedRawTopSelections)
	}
	if result.Summary.InvalidSelections != 1 || result.Summary.RepeatedSelections != 1 || result.Summary.SuppressedRawTopSelections != 2 {
		t.Fatalf("summary selection counts = %#v", result.Summary)
	}
}

func TestEvaluateRecordsFailureAndDeterministicJSON(t *testing.T) {
	v := testVocabulary(t)
	misses := []string{"REBUT", "SISSY", "HUMPH", "AWAKE", "BLUSH", "FOCAL"}
	evaluator := newEvaluator(t, v,
		append([]string{"CIGAR"}, misses...),
		[]string{"CIGAR", "REBUT", "SISSY"},
		func(_ context.Context, position Position) ([]float32, error) {
			return scoresFor(v, misses[position.Turn]), nil
		},
	)

	first, err := evaluator.Evaluate(context.Background(), []string{"CIGAR"})
	if err != nil {
		t.Fatal(err)
	}
	second, err := evaluator.Evaluate(context.Background(), []string{"CIGAR"})
	if err != nil {
		t.Fatal(err)
	}
	game := first.Games[0]
	if game.Solved || game.Guesses != 6 || game.Failure != "unsolved_after_six_guesses" {
		t.Fatalf("failure game = %#v", game)
	}
	if got, want := first.Summary.FailedSolutions, []string{"CIGAR"}; !reflect.DeepEqual(got, want) {
		t.Fatalf("failed solutions = %v, want %v", got, want)
	}
	if got := first.Summary.GuessCountDistribution; got != [6]int{0, 0, 0, 0, 0, 1} {
		t.Fatalf("guess distribution = %v, want one sixth-turn failure", got)
	}
	firstJSON, err := json.Marshal(first)
	if err != nil {
		t.Fatal(err)
	}
	secondJSON, err := json.Marshal(second)
	if err != nil {
		t.Fatal(err)
	}
	if string(firstJSON) != string(secondJSON) {
		t.Fatalf("evaluation JSON is not deterministic:\n%s\n%s", firstJSON, secondJSON)
	}
}

func TestEvaluateRejectsMalformedScores(t *testing.T) {
	v := testVocabulary(t)
	evaluator := newEvaluator(t, v, []string{"CIGAR"}, []string{"CIGAR"}, func(context.Context, Position) ([]float32, error) {
		return []float32{1}, nil
	})
	_, err := evaluator.Evaluate(context.Background(), []string{"CIGAR"})
	if !errors.Is(err, ErrInvalidScores) {
		t.Fatalf("short scores error = %v, want ErrInvalidScores", err)
	}

	evaluator = newEvaluator(t, v, []string{"CIGAR"}, []string{"CIGAR"}, func(context.Context, Position) ([]float32, error) {
		scores := scoresFor(v, "CIGAR")
		scores[0] = float32(math.NaN())
		return scores, nil
	})
	_, err = evaluator.Evaluate(context.Background(), []string{"CIGAR"})
	if !errors.Is(err, ErrInvalidScores) {
		t.Fatalf("non-finite scores error = %v, want ErrInvalidScores", err)
	}
}

func TestSelectActionUsesStableTiesAndLegalFallbacks(t *testing.T) {
	scores := make([]float32, vocabulary.NumActions)
	available := make([]float32, vocabulary.NumActions)
	for index := range scores {
		scores[index] = -100
	}
	scores[19], scores[7] = 5, 5
	available[19], available[7] = 1, 1
	raw, selected, err := selectAction(scores, available)
	if err != nil {
		t.Fatalf("select tied actions: %v", err)
	}
	if raw != 7 || selected != 7 {
		t.Fatalf("tied actions selected raw=%d selected=%d, want 7/7", raw, selected)
	}

	scores[13], scores[9] = 10, 8
	available[13], available[9] = 0, 1
	raw, selected, err = selectAction(scores, available)
	if err != nil {
		t.Fatalf("select legal fallback: %v", err)
	}
	if raw != 13 || selected != 9 {
		t.Fatalf("fallback selected raw=%d selected=%d, want 13/9", raw, selected)
	}
}

func TestSelectActionRejectsNoLegalAction(t *testing.T) {
	scores := make([]float32, vocabulary.NumActions)
	available := make([]float32, vocabulary.NumActions)
	if _, _, err := selectAction(scores, available); !errors.Is(err, ErrNoLegalAction) {
		t.Fatalf("select with no legal actions error = %v, want ErrNoLegalAction", err)
	}
}

func TestNewRejectsMismatchedEvaluationPopulation(t *testing.T) {
	v := testVocabulary(t)
	_, err := New(Config{
		Vocabulary:         v,
		Score:              func(context.Context, Position) ([]float32, error) { return scoresFor(v, "CIGAR"), nil },
		ValidGuesses:       []string{"REBUT"},
		CandidateSolutions: []string{"CIGAR"},
	})
	if err == nil {
		t.Fatal("New succeeded with candidate that is not a valid guess")
	}
}

func newEvaluator(t *testing.T, v *vocabulary.Vocabulary, valid, candidates []string, score ScoreFunc) *Evaluator {
	t.Helper()
	evaluator, err := New(Config{
		Vocabulary:         v,
		Score:              score,
		ValidGuesses:       valid,
		CandidateSolutions: candidates,
	})
	if err != nil {
		t.Fatal(err)
	}
	return evaluator
}

func scoresFor(v *vocabulary.Vocabulary, top string) []float32 {
	return rankedScores(v, top)
}

func rankedScores(v *vocabulary.Vocabulary, ranked ...string) []float32 {
	scores := make([]float32, vocabulary.NumActions)
	for index := range scores {
		scores[index] = -100
	}
	for index, word := range ranked {
		actionID, found := v.ActionID(word)
		if !found {
			panic("test word missing from action vocabulary: " + word)
		}
		scores[actionID] = float32(len(ranked) - index)
	}
	return scores
}

func testVocabulary(t *testing.T) *vocabulary.Vocabulary {
	t.Helper()
	dataDir := os.Getenv("WORDLEML_DATA_DIR")
	if dataDir == "" {
		dataDir = filepath.Join("..", "..", "data")
	}
	v, err := vocabulary.Load(dataDir)
	if err != nil {
		t.Fatal(err)
	}
	return v
}
