package rl

import (
	"errors"
	"reflect"
	"testing"

	"github.com/sam-bee/wordle-ml_game-engine/game"
)

func TestEnvironmentPermitsLegalNonCandidateProbe(t *testing.T) {
	v := testVocabulary(t)
	env, err := NewEnvironment(v, "CIGAR", 11)
	if err != nil {
		t.Fatal(err)
	}
	probeID := nonSolutionActionID(t, v)
	observation, err := env.Observation()
	if err != nil {
		t.Fatal(err)
	}
	if observation.AvailableActionMask[probeID] != 1 {
		t.Fatalf("non-candidate probe action %d was masked", probeID)
	}
	for actionID, available := range observation.AvailableActionMask {
		if available != 1 {
			t.Fatalf("opening action %d availability = %v, want 1", actionID, available)
		}
	}
	transition, err := env.Step(probeID)
	if err != nil {
		t.Fatalf("legal non-candidate probe was rejected: %v", err)
	}
	if transition.Terminal || transition.Solved || transition.Reward != -0.05 {
		t.Fatalf("probe transition = %#v, want non-terminal -0.05", transition)
	}
}

func TestEnvironmentMasksAndRejectsPreviouslyAcceptedGuess(t *testing.T) {
	v := testVocabulary(t)
	env, err := NewEnvironment(v, "CIGAR", 12)
	if err != nil {
		t.Fatal(err)
	}
	probeID := nonSolutionActionID(t, v)
	if _, err := env.Step(probeID); err != nil {
		t.Fatal(err)
	}
	observation, err := env.Observation()
	if err != nil {
		t.Fatal(err)
	}
	if observation.AvailableActionMask[probeID] != 0 {
		t.Fatalf("accepted action %d remained available", probeID)
	}
	for actionID, available := range observation.AvailableActionMask {
		want := float32(1)
		if actionID == probeID {
			want = 0
		}
		if available != want {
			t.Fatalf("action %d availability = %v, want %v after one accepted guess", actionID, available, want)
		}
	}
	if _, err := env.Step(probeID); !errors.Is(err, ErrActionUnavailable) {
		t.Fatalf("repeated action error = %v, want ErrActionUnavailable", err)
	}
}

func TestEnvironmentOpeningObservationDoesNotExposeHiddenAnswer(t *testing.T) {
	v := testVocabulary(t)
	cigar, err := NewEnvironment(v, "CIGAR", 101)
	if err != nil {
		t.Fatal(err)
	}
	rebut, err := NewEnvironment(v, "REBUT", 202)
	if err != nil {
		t.Fatal(err)
	}
	openingCigar, err := cigar.Observation()
	if err != nil {
		t.Fatal(err)
	}
	openingRebut, err := rebut.Observation()
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(openingCigar, openingRebut) {
		t.Fatal("opening actor observations differed by hidden answer")
	}
	if cigar.Metadata().SolutionID == rebut.Metadata().SolutionID {
		t.Fatal("test setup did not create distinct environment-side solution identifiers")
	}
}

func TestEnvironmentSixTurnFailureUsesExactRewards(t *testing.T) {
	v := testVocabulary(t)
	env, err := NewEnvironment(v, "CIGAR", 13)
	if err != nil {
		t.Fatal(err)
	}
	misses := distinctNonSolutionActionIDs(t, v, game.MaxTurns)
	for turn, actionID := range misses {
		transition, err := env.Step(actionID)
		if err != nil {
			t.Fatalf("turn %d: %v", turn, err)
		}
		if transition.Turn != turn {
			t.Fatalf("transition turn = %d, want %d", transition.Turn, turn)
		}
		if got, want := transition.Metadata.EpisodeID, int64(13); got != want {
			t.Fatalf("episode ID = %d, want %d", got, want)
		}
		if turn < game.MaxTurns-1 {
			if transition.Terminal || transition.Reward != -0.05 {
				t.Fatalf("turn %d transition = %#v, want non-terminal -0.05", turn, transition)
			}
			continue
		}
		if !transition.Terminal || transition.Solved || transition.Reward != -1.00 {
			t.Fatalf("sixth-turn failure = %#v, want terminal unsolved -1.00", transition)
		}
	}
	if !env.Finished() {
		t.Fatal("environment not finished after sixth accepted guess")
	}
	if _, err := env.Observation(); !errors.Is(err, game.ErrFinished) {
		t.Fatalf("terminal observation error = %v, want game.ErrFinished", err)
	}
	if _, err := env.Step(nonSolutionActionID(t, v)); !errors.Is(err, game.ErrFinished) {
		t.Fatalf("post-terminal step error = %v, want game.ErrFinished", err)
	}
}

func TestEnvironmentSolveRewardsOneOnAnyTurn(t *testing.T) {
	v := testVocabulary(t)
	solutionID, found := v.ActionID("CIGAR")
	if !found {
		t.Fatal("CIGAR missing from action vocabulary")
	}
	env, err := NewEnvironment(v, "CIGAR", 14)
	if err != nil {
		t.Fatal(err)
	}
	transition, err := env.Step(solutionID)
	if err != nil {
		t.Fatal(err)
	}
	if !transition.Terminal || !transition.Solved || transition.Reward != 1.00 {
		t.Fatalf("solve transition = %#v, want terminal solved +1", transition)
	}
}

func nonSolutionActionID(t *testing.T, v interface {
	Actions() []string
	ActionID(string) (int, bool)
	SolutionID(string) (int, bool)
}) int {
	t.Helper()
	for _, word := range v.Actions() {
		if _, solution := v.SolutionID(word); solution {
			continue
		}
		actionID, found := v.ActionID(word)
		if !found {
			t.Fatalf("action word %q was not addressable", word)
		}
		return actionID
	}
	t.Fatal("action vocabulary has no non-solution probe word")
	return 0
}

func distinctNonSolutionActionIDs(t *testing.T, v interface {
	Actions() []string
	ActionID(string) (int, bool)
	SolutionID(string) (int, bool)
}, count int) []int {
	t.Helper()
	ids := make([]int, 0, count)
	for _, word := range v.Actions() {
		if _, solution := v.SolutionID(word); solution {
			continue
		}
		actionID, _ := v.ActionID(word)
		ids = append(ids, actionID)
		if len(ids) == count {
			return ids
		}
	}
	t.Fatalf("only found %d non-solution actions, need %d", len(ids), count)
	return nil
}
