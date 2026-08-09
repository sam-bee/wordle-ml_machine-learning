package evaluation

import (
	"math"
	"reflect"
	"testing"
)

func TestSummarizeReportsDeploymentMetricsAndTelemetry(t *testing.T) {
	evaluation := Evaluation{
		Games: []GameResult{
			game("A", true, 1, "RAISE", 0.23, 0, 0),
			game("B", true, 3, "RAISE", 0.23, 0, 0),
			game("C", false, 6, "RAISE", 0.23, 1, 2),
		},
		Diagnostics: stableDiagnostics(),
	}
	summary, err := Summarize(evaluation)
	if err != nil {
		t.Fatal(err)
	}
	if summary.Games != 3 || summary.SolvedCount != 2 || summary.FailureCount != 1 {
		t.Fatalf("game counts = %+v", summary)
	}
	if summary.SolveRate != 2.0/3.0 || summary.MeanGuessesSolved != 2 || summary.FailureCountedMeanGuesses != 11.0/3.0 {
		t.Fatalf("means = %+v", summary)
	}
	if want := [MaxTurns]int{1, 0, 1, 0, 0, 1}; summary.GuessCountHistogram != want {
		t.Fatalf("histogram = %v, want %v", summary.GuessCountHistogram, want)
	}
	if summary.OpeningGuess != "RAISE" || summary.OpeningActionProbability != 0.23 {
		t.Fatalf("opening = %q/%g", summary.OpeningGuess, summary.OpeningActionProbability)
	}
	if summary.AcceptedIllegalActionCount != 1 || summary.AcceptedRepeatedActionCount != 2 {
		t.Fatalf("accepted action counters = %+v", summary)
	}
	if summary.MeanTurnPolicyEntropy != 2.8 {
		t.Fatalf("mean per-turn entropy = %g, want 2.8", summary.MeanTurnPolicyEntropy)
	}
}

func TestSummarizeRejectsMismatchedOpeningAndMalformedGames(t *testing.T) {
	base := Evaluation{Games: []GameResult{game("A", true, 1, "RAISE", 0.2, 0, 0), game("B", true, 2, "SLATE", 0.2, 0, 0)}, Diagnostics: stableDiagnostics()}
	if _, err := Summarize(base); err == nil {
		t.Fatal("mismatched greedy opening was accepted")
	}

	bad := game("A", false, 5, "RAISE", 0.2, 0, 0)
	if _, err := Summarize(Evaluation{Games: []GameResult{bad}, Diagnostics: stableDiagnostics()}); err == nil {
		t.Fatal("failure before sixth accepted guess was accepted")
	}

	bad = game("A", true, 1, "RAISE", 0.2, 0, 0)
	bad.Turns[0].ActionProbability = math.NaN()
	if _, err := Summarize(Evaluation{Games: []GameResult{bad}, Diagnostics: stableDiagnostics()}); err == nil {
		t.Fatal("non-finite turn statistic was accepted")
	}
}

func TestComparePairsByIDAndCalculatesChangesAndBootstrapDeterministically(t *testing.T) {
	baseline := Evaluation{
		Games: []GameResult{
			game("C", false, 6, "RAISE", 0.2, 0, 0), // 7 -> 2: newly solved
			game("A", true, 2, "RAISE", 0.2, 0, 0),  // 2 -> 1: improved
			game("B", true, 1, "RAISE", 0.2, 0, 0),  // 1 -> 3: worsened
			game("D", true, 3, "RAISE", 0.2, 0, 0),  // 3 -> 7: newly failed
			game("E", true, 6, "RAISE", 0.2, 0, 0),  // 6 -> 7: solved count decreases
		},
		Diagnostics: stableDiagnostics(),
	}
	candidate := Evaluation{
		Games: []GameResult{
			game("E", false, 6, "RAISE", 0.2, 0, 0),
			game("D", false, 6, "RAISE", 0.2, 0, 0),
			game("B", true, 3, "RAISE", 0.2, 0, 0),
			game("C", true, 2, "RAISE", 0.2, 0, 0),
			game("A", true, 1, "RAISE", 0.2, 0, 0),
		},
		Diagnostics: stableDiagnostics(),
	}
	options := DefaultCompareOptions(47)
	options.BootstrapSamples = 101
	first, err := Compare(baseline, candidate, options)
	if err != nil {
		t.Fatal(err)
	}
	second, err := Compare(baseline, candidate, options)
	if err != nil {
		t.Fatal(err)
	}
	if first.Improved != 2 || first.Worsened != 3 || first.Unchanged != 0 {
		t.Fatalf("paired counts = %+v", first)
	}
	if !reflect.DeepEqual(first.NewlySolvedIDs, []string{"C"}) || !reflect.DeepEqual(first.NewlyFailedIDs, []string{"D", "E"}) {
		t.Fatalf("new solved/failed = %v/%v", first.NewlySolvedIDs, first.NewlyFailedIDs)
	}
	if first.PairedMeanDifference != 0.2 { // (-1 + 2 - 5 + 4 + 1) / 5
		t.Fatalf("paired mean difference = %g, want 0.2", first.PairedMeanDifference)
	}
	if !reflect.DeepEqual(first.PairedBootstrap95, second.PairedBootstrap95) {
		t.Fatalf("bootstrap changed for same data/seed: %+v then %+v", first.PairedBootstrap95, second.PairedBootstrap95)
	}
	if first.Acceptance.SolvedCountNonDecreasing {
		t.Fatal("candidate with one newly failed game passed solved-count gate")
	}
	if first.Classification != Rejected {
		t.Fatalf("classification = %q, want rejected", first.Classification)
	}
}

func TestCompareClassificationAndAcceptanceGates(t *testing.T) {
	baseline := Evaluation{Games: []GameResult{
		game("A", true, 4, "RAISE", 0.2, 0, 0), game("B", true, 4, "RAISE", 0.2, 0, 0),
		game("C", true, 4, "RAISE", 0.2, 0, 0), game("D", true, 4, "RAISE", 0.2, 0, 0),
	}, Diagnostics: stableDiagnostics()}
	candidate := Evaluation{Games: []GameResult{
		game("D", true, 3, "RAISE", 0.2, 0, 0), game("C", true, 3, "RAISE", 0.2, 0, 0),
		game("B", true, 3, "RAISE", 0.2, 0, 0), game("A", true, 3, "RAISE", 0.2, 0, 0),
	}, Diagnostics: stableDiagnostics()}
	options := DefaultCompareOptions(9)
	options.BootstrapSamples = 200
	comparison, err := Compare(baseline, candidate, options)
	if err != nil {
		t.Fatal(err)
	}
	if !comparison.Acceptance.Accepted || comparison.Classification != ConvincinglyImproved {
		t.Fatalf("uniform strict improvement = acceptance %+v classification %q", comparison.Acceptance, comparison.Classification)
	}

	candidate.Diagnostics.NumericallyStable = false
	comparison, err = Compare(baseline, candidate, options)
	if err != nil {
		t.Fatal(err)
	}
	if comparison.Acceptance.Accepted || comparison.Classification != Rejected || !contains(comparison.Acceptance.Reasons, "candidate reported numerical instability") {
		t.Fatalf("numeric failure gate = %+v / %q", comparison.Acceptance, comparison.Classification)
	}

	candidate.Diagnostics = stableDiagnostics()
	candidate.Games[0].AcceptedRepeatedActions = 1
	comparison, err = Compare(baseline, candidate, options)
	if err != nil {
		t.Fatal(err)
	}
	if comparison.Acceptance.NoAcceptedIllegalOrRepeated || comparison.Classification != Rejected {
		t.Fatalf("accepted repeated action gate = %+v / %q", comparison.Acceptance, comparison.Classification)
	}
}

func TestCompareIsInconclusiveWithoutStrictDeploymentImprovement(t *testing.T) {
	baseline := Evaluation{Games: []GameResult{game("A", true, 3, "RAISE", 0.2, 0, 0)}, Diagnostics: stableDiagnostics()}
	candidate := Evaluation{Games: []GameResult{game("A", true, 3, "RAISE", 0.2, 0, 0)}, Diagnostics: stableDiagnostics()}
	comparison, err := Compare(baseline, candidate, DefaultCompareOptions(3))
	if err != nil {
		t.Fatal(err)
	}
	if comparison.Acceptance.Accepted || comparison.Classification != Inconclusive {
		t.Fatalf("equal candidate = acceptance %+v classification %q", comparison.Acceptance, comparison.Classification)
	}
}

func TestCompareClassifiesUncertainPointImprovementAsPromising(t *testing.T) {
	baseline := Evaluation{Games: []GameResult{
		game("A", true, 4, "RAISE", 0.2, 0, 0), game("B", true, 4, "RAISE", 0.2, 0, 0),
		game("C", true, 4, "RAISE", 0.2, 0, 0), game("D", true, 4, "RAISE", 0.2, 0, 0),
	}, Diagnostics: stableDiagnostics()}
	candidate := Evaluation{Games: []GameResult{
		game("A", true, 1, "RAISE", 0.2, 0, 0), game("B", true, 6, "RAISE", 0.2, 0, 0),
		game("C", true, 4, "RAISE", 0.2, 0, 0), game("D", true, 4, "RAISE", 0.2, 0, 0),
	}, Diagnostics: stableDiagnostics()}
	options := DefaultCompareOptions(99)
	options.BootstrapSamples = 1_000
	comparison, err := Compare(baseline, candidate, options)
	if err != nil {
		t.Fatal(err)
	}
	if !comparison.Acceptance.Accepted || comparison.PairedBootstrap95.Lower >= 0 || comparison.PairedBootstrap95.Upper < 0 || comparison.Classification != Promising {
		t.Fatalf("uncertain improvement = acceptance %+v interval %+v classification %q", comparison.Acceptance, comparison.PairedBootstrap95, comparison.Classification)
	}
}

func TestPairedBootstrap95IsSeededAndHandlesDegenerateInput(t *testing.T) {
	differences := []float64{-1, 0, 1}
	first := PairedBootstrap95(differences, 11, 100)
	second := PairedBootstrap95(differences, 11, 100)
	if first != second || first.Lower > first.Upper {
		t.Fatalf("deterministic interval = %+v then %+v", first, second)
	}
	degenerate := PairedBootstrap95(nil, 11, 0)
	if degenerate.Level != 0.95 || degenerate.Samples != 0 || degenerate.Lower != 0 || degenerate.Upper != 0 {
		t.Fatalf("degenerate interval = %+v", degenerate)
	}
}

func game(id string, solved bool, guesses int, opening string, probability float64, illegal, repeated int) GameResult {
	turns := make([]TurnRecord, guesses)
	for index := range turns {
		turns[index] = TurnRecord{
			Turn: index + 1, Guess: opening, ActionID: index, ActionProbability: probability,
			PolicyEntropy: float64(index + 1), Terminal: index == len(turns)-1,
		}
	}
	return GameResult{
		SolutionID: id, Solved: solved, Guesses: guesses, Turns: turns,
		AcceptedIllegalActions: illegal, AcceptedRepeatedActions: repeated,
	}
}

func stableDiagnostics() Diagnostics {
	return Diagnostics{
		PolicyEntropy: 1, ApproxOldPolicyKL: 0.005, SupervisedReferenceKL: 0.01,
		ClipFraction: 0.1, CriticExplainedVar: 0.3, NumericallyStable: true,
	}
}

func contains(values []string, wanted string) bool {
	for _, value := range values {
		if value == wanted {
			return true
		}
	}
	return false
}
