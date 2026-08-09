package cudafinal

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/sam-bee/wordle-ml_machine-learning/cudainfer"
	"github.com/sam-bee/wordle-ml_machine-learning/cudamodel"
	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
)

func TestCompleteIsAggregateOnlyAndUTC(t *testing.T) {
	started := startedReport(t)
	summary := gameeval.Summary{
		Games: 100, Solved: 99, SolvedFraction: .99, MeanGuesses: 2.04,
		GuessCountDistribution: [6]int{0, 99, 0, 0, 0, 1}, Failures: 1,
		FailedSolutions: []string{"SECRET"}, InvalidSelections: 2, SuppressedRawTopSelections: 3, RepeatedSelections: 4,
	}
	report, err := Complete(started, time.Date(2026, 8, 10, 10, 0, 0, 0, time.FixedZone("other", 3600)), strings.Repeat("b", 64), summary)
	if err != nil {
		t.Fatalf("Complete: %v", err)
	}
	if report.Status != StatusComplete || report.TimestampUTC.Location() != time.UTC || report.Aggregate == nil || report.Aggregate.Games != 100 || report.Aggregate.Failures != 1 {
		t.Fatalf("report = %+v", report)
	}
	contents, err := json.Marshal(report)
	if err != nil {
		t.Fatal(err)
	}
	for _, forbidden := range []string{"SECRET", "failed_solutions"} {
		if strings.Contains(string(contents), forbidden) {
			t.Fatalf("sanitized report contains %q: %s", forbidden, contents)
		}
	}
}

func TestClaimBlocksEveryLaterAttemptAndReplacePublishesFailure(t *testing.T) {
	path := filepath.Join(t.TempDir(), "reports", "final.json")
	started := startedReport(t)
	if err := Claim(path, started); err != nil {
		t.Fatalf("Claim: %v", err)
	}
	if err := Claim(path, started); err == nil {
		t.Fatal("second final-test claim was accepted")
	}
	failed, err := Failed(started, time.Now(), FailureGameplay)
	if err != nil {
		t.Fatal(err)
	}
	if err := Replace(path, failed); err != nil {
		t.Fatalf("Replace: %v", err)
	}
	contents, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(contents), `"status": "failed"`) || strings.Contains(string(contents), "SECRET") {
		t.Fatalf("failed report = %s", contents)
	}
}

func TestCompleteRejectsInconsistentSummary(t *testing.T) {
	_, err := Complete(startedReport(t), time.Now(), strings.Repeat("c", 64), gameeval.Summary{
		Games: 100, Solved: 99, SolvedFraction: .99, MeanGuesses: 2, GuessCountDistribution: [6]int{100}, Failures: 0,
	})
	if err == nil {
		t.Fatal("Complete accepted an inconsistent summary")
	}
}

func TestFailedRejectsUnsafeFailureText(t *testing.T) {
	_, err := Failed(startedReport(t), time.Now(), FailureCode("SECRET"))
	if err == nil {
		t.Fatalf("Failed error = %v, want unsafe failure rejection", err)
	}
}

func startedReport(t *testing.T) Report {
	t.Helper()
	manifest := cudamodel.Manifest{
		Format: cudamodel.Format, RunID: "seed-replication", Checkpoint: "best", CheckpointUpdate: 2600,
		TrainingCommit: strings.Repeat("a", 40), WeightsSHA256: strings.Repeat("b", 64), ParameterCount: cudamodel.ParameterCount,
	}
	report, err := NewStarted(time.Now(), strings.Repeat("c", 40), manifest, cudainfer.Info{DeviceName: "RTX", ComputeCapability: "12.0"})
	if err != nil {
		t.Fatalf("NewStarted: %v", err)
	}
	return report
}
