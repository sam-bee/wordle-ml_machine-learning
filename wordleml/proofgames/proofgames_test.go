package proofgames

import (
	"strings"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
)

func TestJSONLAndRepeatedAcceptedGuessCheck(t *testing.T) {
	contents, err := JSONL([]gameeval.GameResult{{Solution: "ABCDE", Guesses: 1}})
	if err != nil || !strings.Contains(string(contents), `"solution":"ABCDE"`) {
		t.Fatalf("JSONL = %q, %v", contents, err)
	}
	err = CheckTrajectories(gameeval.Evaluation{Games: []gameeval.GameResult{{Solution: "ABCDE", Turns: []gameeval.TurnResult{{Guess: "FIRST"}, {Guess: "FIRST"}}}}})
	if err == nil {
		t.Fatal("repeated accepted guess was accepted")
	}
	if err := CheckTrajectories(gameeval.Evaluation{Summary: gameeval.Summary{InvalidSelections: 1}}); err != nil {
		t.Fatalf("suppressed raw invalid preference is not an accepted move: %v", err)
	}
}

func TestTensorBoardScalarsHaveFixedOrderedGameTags(t *testing.T) {
	evaluation := Evaluation{Summary: gameeval.Summary{SolvedFraction: .75, MeanGuesses: 3.5, Failures: 2, GuessCountDistribution: [6]int{1, 2, 3, 4, 5, 6}}}
	scalars := TensorBoardScalars(evaluation)
	wantTags := []string{"games/solved_fraction", "games/mean_guesses", "games/failures", "games/guess_count_1", "games/guess_count_2", "games/guess_count_3", "games/guess_count_4", "games/guess_count_5", "games/guess_count_6"}
	if len(scalars) != len(wantTags) {
		t.Fatalf("scalar count = %d, want %d", len(scalars), len(wantTags))
	}
	for index, want := range wantTags {
		if scalars[index].Tag != want {
			t.Errorf("scalar %d tag = %q, want %q", index, scalars[index].Tag, want)
		}
	}
	if scalars[0].Value != .75 || scalars[8].Value != 6 {
		t.Fatalf("unexpected scalar values: %+v", scalars)
	}
}
