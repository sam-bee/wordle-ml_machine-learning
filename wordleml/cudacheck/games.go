package cudacheck

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"reflect"

	"github.com/sam-bee/wordle-ml_machine-learning/cudamodel"
	"github.com/sam-bee/wordle-ml_machine-learning/cudaref"
	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

// GamesReport records exact trajectory parity for the frozen 100-game
// validation population.
type GamesReport struct {
	GamesCompared   int              `json:"games_compared"`
	ExpectedSummary gameeval.Summary `json:"expected_summary"`
	ActualSummary   gameeval.Summary `json:"actual_summary"`
	Exact           bool             `json:"exact"`
	FirstDivergence *GameDivergence  `json:"first_divergence,omitempty"`
	Error           string           `json:"error,omitempty"`
}

// GameDivergence identifies the first non-identical game field.
type GameDivergence struct {
	Solution string `json:"solution"`
	Turn     int    `json:"turn"`
	Field    string `json:"field"`
	Expected any    `json:"expected"`
	Actual   any    `json:"actual"`
}

// LoadGoldenGames reads the standalone exported GoMLX game reference without
// importing any checkpoint or GoMLX code.
func LoadGoldenGames(dir string) (cudaref.Games, error) {
	path := filepath.Join(dir, cudaref.GoldenGamesFilename)
	contents, err := os.ReadFile(path)
	if err != nil {
		return cudaref.Games{}, fmt.Errorf("read %s: %w", cudaref.GoldenGamesFilename, err)
	}
	decoder := json.NewDecoder(bytes.NewReader(contents))
	decoder.DisallowUnknownFields()
	var games cudaref.Games
	if err := decoder.Decode(&games); err != nil {
		return cudaref.Games{}, fmt.Errorf("decode %s: %w", cudaref.GoldenGamesFilename, err)
	}
	if err := decoder.Decode(&struct{}{}); !errors.Is(err, io.EOF) {
		return cudaref.Games{}, fmt.Errorf("decode %s: trailing JSON", cudaref.GoldenGamesFilename)
	}
	if games.RunID == "" || games.Checkpoint == "" || games.CheckpointUpdate < 0 {
		return cudaref.Games{}, fmt.Errorf("golden games identity is incomplete")
	}
	if len(games.Evaluation.Games) != vocabulary.NumValidationSolutions {
		return cudaref.Games{}, fmt.Errorf("golden games contain %d games, want %d", len(games.Evaluation.Games), vocabulary.NumValidationSolutions)
	}
	return games, nil
}

// VerifyGames evaluates the fixed validation population through CUDA and checks
// every complete trajectory, including raw selection diagnostics and summary.
func VerifyGames(ctx context.Context, manifest cudamodel.Manifest, vocab *vocabulary.Vocabulary, reference cudaref.Games, backend Scorer) GamesReport {
	report := GamesReport{ExpectedSummary: reference.Evaluation.Summary}
	if ctx == nil || vocab == nil || backend == nil {
		report.Error = "verification context, vocabulary, and CUDA backend are required"
		return report
	}
	if reference.RunID != manifest.RunID || reference.Checkpoint != manifest.Checkpoint || reference.CheckpointUpdate != int64(manifest.CheckpointUpdate) {
		report.Error = "golden-game model identity differs from portable model manifest"
		return report
	}
	if len(reference.Evaluation.Games) != vocabulary.NumValidationSolutions {
		report.Error = fmt.Sprintf("golden games contain %d games, want %d", len(reference.Evaluation.Games), vocabulary.NumValidationSolutions)
		return report
	}
	expectedSolutions := vocab.Validation()
	for index, game := range reference.Evaluation.Games {
		if game.Solution != expectedSolutions[index] {
			report.Error = fmt.Sprintf("golden game %d solution %q differs from validation solution %q", index, game.Solution, expectedSolutions[index])
			return report
		}
	}
	evaluator, err := gameeval.New(gameeval.Config{
		Vocabulary: vocab,
		Score: func(ctx context.Context, position gameeval.Position) ([]float32, error) {
			return backend.Score(ctx, position.Inputs)
		},
	})
	if err != nil {
		report.Error = fmt.Sprintf("create CUDA game evaluator: %v", err)
		return report
	}
	actual, err := evaluator.Evaluate(ctx, expectedSolutions)
	if err != nil {
		report.Error = fmt.Sprintf("evaluate CUDA validation games: %v", err)
		return report
	}
	report.ActualSummary = actual.Summary
	report.GamesCompared = len(actual.Games)
	report.FirstDivergence = compareEvaluation(reference.Evaluation, actual)
	report.Exact = report.FirstDivergence == nil
	return report
}

func compareEvaluation(expected, actual gameeval.Evaluation) *GameDivergence {
	if len(expected.Games) != len(actual.Games) {
		return &GameDivergence{Field: "games.length", Expected: len(expected.Games), Actual: len(actual.Games)}
	}
	for gameIndex := range expected.Games {
		if divergence := compareGame(expected.Games[gameIndex], actual.Games[gameIndex]); divergence != nil {
			return divergence
		}
	}
	if !reflect.DeepEqual(expected.Summary, actual.Summary) {
		return &GameDivergence{Field: "summary", Expected: expected.Summary, Actual: actual.Summary}
	}
	return nil
}

func compareGame(expected, actual gameeval.GameResult) *GameDivergence {
	base := func(field string, want, got any) *GameDivergence {
		if reflect.DeepEqual(want, got) {
			return nil
		}
		return &GameDivergence{Solution: expected.Solution, Field: field, Expected: want, Actual: got}
	}
	for _, check := range []struct {
		field string
		want  any
		got   any
	}{
		{"solution", expected.Solution, actual.Solution},
		{"solved", expected.Solved, actual.Solved},
		{"guesses", expected.Guesses, actual.Guesses},
		{"failure", expected.Failure, actual.Failure},
		{"invalid_selections", expected.InvalidSelections, actual.InvalidSelections},
		{"suppressed_raw_top_selections", expected.SuppressedRawTopSelections, actual.SuppressedRawTopSelections},
		{"repeated_selections", expected.RepeatedSelections, actual.RepeatedSelections},
		{"turns.length", len(expected.Turns), len(actual.Turns)},
	} {
		if divergence := base(check.field, check.want, check.got); divergence != nil {
			return divergence
		}
	}
	for index, expectedTurn := range expected.Turns {
		actualTurn := actual.Turns[index]
		for _, check := range []struct {
			field string
			want  any
			got   any
		}{
			{"turn", expectedTurn.Turn, actualTurn.Turn},
			{"raw_top_action_id", expectedTurn.RawTopActionID, actualTurn.RawTopActionID},
			{"raw_top_guess", expectedTurn.RawTopGuess, actualTurn.RawTopGuess},
			{"guess", expectedTurn.Guess, actualTurn.Guess},
			{"feedback", expectedTurn.Feedback, actualTurn.Feedback},
			{"shortlist_size_before", expectedTurn.ShortlistSizeBefore, actualTurn.ShortlistSizeBefore},
			{"shortlist_size_after", expectedTurn.ShortlistSizeAfter, actualTurn.ShortlistSizeAfter},
		} {
			if !reflect.DeepEqual(check.want, check.got) {
				return &GameDivergence{Solution: expected.Solution, Turn: index + 1, Field: check.field, Expected: check.want, Actual: check.got}
			}
		}
	}
	return nil
}
