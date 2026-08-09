package cudacheck

import (
	"context"
	"encoding/json"
	"math"
	"strings"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/cudainfer"
	"github.com/sam-bee/wordle-ml_machine-learning/cudaref"
	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

func TestCompareLogitsUsesStableTopKAndTolerance(t *testing.T) {
	expected := make([]float32, vocabulary.NumActions)
	actual := make([]float32, vocabulary.NumActions)
	expected[5], expected[7], expected[9], expected[11], expected[13] = 10, 10, 9, 8, 7
	actual[5], actual[7], actual[9], actual[11], actual[13] = 10, 10, 9, 8, 7
	actual[13] += 0.01
	comparison, err := CompareLogits(expected, actual, 0.001)
	if err != nil {
		t.Fatal(err)
	}
	if comparison.Top1Agreement != true || comparison.Top5SetAgreement != true {
		t.Fatalf("top-k agreement = %+v", comparison)
	}
	if comparison.WorstActionID != 13 || comparison.WithinTolerance {
		t.Fatalf("tolerance comparison = %+v", comparison)
	}
}

func TestSelectActionKeepsRawAndLegalDecisionsSeparate(t *testing.T) {
	logits := make([]float32, vocabulary.NumActions)
	available := make([]float32, vocabulary.NumActions)
	logits[2], logits[9], logits[10] = 5, 4, 4
	available[9], available[10] = 1, 1
	selection, err := SelectAction(logits, available)
	if err != nil {
		t.Fatal(err)
	}
	if selection.RawTopActionID != 2 || selection.SelectedActionID != 9 || selection.TopTwoMargin != 1 {
		t.Fatalf("selection = %+v", selection)
	}
}

func TestCompareLogitsRejectsNonFiniteValues(t *testing.T) {
	expected := make([]float32, vocabulary.NumActions)
	actual := make([]float32, vocabulary.NumActions)
	actual[3] = float32(math.NaN())
	if _, err := CompareLogits(expected, actual, DefaultTolerance); err == nil {
		t.Fatal("non-finite CUDA logit was accepted")
	}
}

func TestCompareEvaluationFindsRawTopTrajectoryDifference(t *testing.T) {
	expected := gameeval.Evaluation{Games: []gameeval.GameResult{{
		Solution: "ADEPT", Solved: true, Guesses: 1,
		Turns: []gameeval.TurnResult{{Turn: 1, RawTopActionID: 7, RawTopGuess: "ARISE", Guess: "ADEPT", Feedback: "GGGGG", ShortlistSizeBefore: 2309, ShortlistSizeAfter: 1}},
	}}}
	actual := gameeval.Evaluation{Games: []gameeval.GameResult{{
		Solution: "ADEPT", Solved: true, Guesses: 1,
		Turns: []gameeval.TurnResult{{Turn: 1, RawTopActionID: 8, RawTopGuess: "ARISE", Guess: "ADEPT", Feedback: "GGGGG", ShortlistSizeBefore: 2309, ShortlistSizeAfter: 1}},
	}}}
	divergence := compareEvaluation(expected, actual)
	if divergence == nil || divergence.Field != "raw_top_action_id" || divergence.Turn != 1 {
		t.Fatalf("divergence = %+v", divergence)
	}
}

func TestCompareEvaluationFindsTurnNumberDifference(t *testing.T) {
	expected := gameeval.Evaluation{Games: []gameeval.GameResult{{
		Solution: "ADEPT", Turns: []gameeval.TurnResult{{Turn: 1, RawTopActionID: 7, RawTopGuess: "ARISE", Guess: "ADEPT", Feedback: "GGGGG", ShortlistSizeBefore: 2309, ShortlistSizeAfter: 1}},
	}}}
	actual := gameeval.Evaluation{Games: []gameeval.GameResult{{
		Solution: "ADEPT", Turns: []gameeval.TurnResult{{Turn: 2, RawTopActionID: 7, RawTopGuess: "ARISE", Guess: "ADEPT", Feedback: "GGGGG", ShortlistSizeBefore: 2309, ShortlistSizeAfter: 1}},
	}}}
	divergence := compareEvaluation(expected, actual)
	if divergence == nil || divergence.Field != "turn" || divergence.Turn != 1 {
		t.Fatalf("divergence = %+v", divergence)
	}
}

func TestBenchmarkVerifiesEveryCallAndSummarizes(t *testing.T) {
	vector := validVector()
	scorer := &countingScorer{logits: vector.RawLogits}
	report, err := Benchmark(context.Background(), BenchmarkOptions{
		Backend: scorer, Vector: vector, Tolerance: DefaultTolerance, Warmup: 2, Iterations: 3,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !report.Passed || report.WarmCall.Count != 3 || scorer.calls != 6 {
		t.Fatalf("benchmark report = %+v, calls = %d", report, scorer.calls)
	}
}

func TestRawTopMismatchIsAnExactVerificationFailure(t *testing.T) {
	vector := validVector()
	vector.RawLogits[7] = 2
	vector.RawTopActionID = 7
	vector.AvailableActionMask[7] = 0
	actual := append([]float32(nil), vector.RawLogits...)
	actual[5] = 3
	report := VerificationReport{Tolerance: DefaultTolerance}
	report.recordActionDivergence(
		vector,
		Selection{RawTopActionID: 7, SelectedActionID: 3, TopTwoMargin: 1},
		Selection{RawTopActionID: 5, SelectedActionID: 3, TopTwoMargin: 2},
		vector.RawLogits,
		actual,
		"reference_vs_cuda",
	)
	if len(report.Failures) != 1 || !strings.Contains(report.Failures[0], "raw top action differs") {
		t.Fatalf("raw mismatch was not an exact failure: %+v", report)
	}
	if report.FirstDivergence == nil || report.FirstDivergence.ExpectedRawTopLogit != 2 || report.FirstDivergence.ActualRawTopLogit != 3 || report.FirstDivergence.ExpectedTopTwoMargin != 1 || report.FirstDivergence.ActualTopTwoMargin != 2 {
		t.Fatalf("raw mismatch diagnostics = %+v", report.FirstDivergence)
	}
}

func TestBenchmarkRejectsRawTopMismatchDespiteMatchingLegalAction(t *testing.T) {
	vector := validVector()
	vector.RawLogits[5] = 2
	vector.RawTopActionID = 5
	vector.AvailableActionMask[5] = 0
	actual := append([]float32(nil), vector.RawLogits...)
	actual[4] = 3
	report, err := Benchmark(context.Background(), BenchmarkOptions{
		Backend: &countingScorer{logits: actual}, Vector: vector, Tolerance: 10, Iterations: 1,
	})
	if err != nil {
		t.Fatal(err)
	}
	if report.Passed || !strings.Contains(report.Failure, "raw top action") {
		t.Fatalf("benchmark accepted raw mismatch: %+v", report)
	}
}

func TestBenchmarkReportIncludesModelAndDeviceIdentity(t *testing.T) {
	contents, err := json.Marshal(BenchmarkReport{Identity: Identity{
		RunID: "seed-replication", Checkpoint: "best", Device: cudainfer.Info{DeviceName: "NVIDIA GeForce RTX 5070 Ti"},
	}})
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{"\"identity\"", "\"run_id\":\"seed-replication\"", "\"device_name\":\"NVIDIA GeForce RTX 5070 Ti\""} {
		if !strings.Contains(string(contents), want) {
			t.Fatalf("benchmark JSON does not include %s: %s", want, contents)
		}
	}
}

func validVector() cudaref.Vector {
	inputs := modelstate.Inputs{
		CandidateMask:       make([]float32, vocabulary.NumSolutions),
		CandidateStats:      make([]float32, modelstate.CandidateStatsSize),
		RemainingActionMask: make([]float32, vocabulary.NumActions),
	}
	inputs.CandidateMask[0] = 1
	logits := make([]float32, vocabulary.NumActions)
	logits[3] = 1
	available := make([]float32, vocabulary.NumActions)
	available[3] = 1
	return cudaref.Vector{
		ID: "opening", Inputs: inputs, AvailableActionMask: available, RawLogits: logits,
		RawTopActionID: 3, SelectedActionID: 3, TopTwoMargin: 1,
		Provenance: cudaref.Provenance{CandidateCount: 1},
	}
}

type countingScorer struct {
	logits []float32
	calls  int
}

func (scorer *countingScorer) Score(_ context.Context, _ modelstate.Inputs) ([]float32, error) {
	scorer.calls++
	return append([]float32(nil), scorer.logits...), nil
}
