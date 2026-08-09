package productionreport

import (
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
)

func TestValidateAndWriteSeedReplicationIncludesPairedEvidence(t *testing.T) {
	runsDir := filepath.Join(t.TempDir(), "runs")
	original := createFixtureRun(t, runsDir, "production-20260809-005026Z", "production", productionUpdates, 97)
	replication := createFixtureRun(t, runsDir, "seed-replication-20260809-010203Z", "seed-replication", productionUpdates, 98)
	writePairedTop1Fixture(t, original, replication, 1400, 100, 100, 900)

	options := SeedReplicationOptions{RunsDir: runsDir, OriginalRunID: original.layout.ID, ReplicationRunID: replication.layout.ID}
	report, err := ValidateSeedReplication(options)
	if err != nil {
		t.Fatal(err)
	}
	if report.Original.Seed != 20260808 || report.Replication.Seed != 20260809 || len(report.Games) != 100 {
		t.Fatalf("replication report identity = %+v", report)
	}
	if !reflect.DeepEqual(report.Failures.Both, []string{"S0098", "S0099"}) || !reflect.DeepEqual(report.Failures.OriginalOnly, []string{"S0097"}) || len(report.Failures.ReplicationOnly) != 0 {
		t.Fatalf("failure comparison = %+v", report.Failures)
	}
	if report.PairedTop1.BothTeacherTop1 != 1400 || report.PairedTop1.OriginalOnly != 100 || report.PairedTop1.ReplicationOnly != 100 || report.PairedTop1.NeitherTeacherTop1 != 900 {
		t.Fatalf("paired top-1 = %+v", report.PairedTop1)
	}
	output := filepath.Join(t.TempDir(), "docs", "ml", "seed-replication-report.md")
	options.OutputPath = output
	written, err := WriteSeedReplication(options)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(written, report) {
		t.Fatalf("written report differs from validated report")
	}
	contents, err := os.ReadFile(output)
	if err != nil {
		t.Fatal(err)
	}
	for _, want := range []string{
		"<!-- seedreplicationreport: complete -->", "not a comprehensive statistical study", "20260808", "20260809",
		"Final validation loss", "Both selected teacher top-1", "Original only", "Replication only", "Neither",
		"S0097", "| S0000 | solved | solved |", "best-paired-top1.json", "final-test split remained unopened",
	} {
		if !strings.Contains(string(contents), want) {
			t.Errorf("report lacks %q:\n%s", want, contents)
		}
	}
}

func TestSeedReplicationRejectsInvalidPairingAndProductionReportPath(t *testing.T) {
	runsDir := filepath.Join(t.TempDir(), "runs")
	original := createFixtureRun(t, runsDir, "production-001", "production", productionUpdates, 97)
	replication := createFixtureRun(t, runsDir, "seed-replication-001", "seed-replication", productionUpdates, 98)
	writePairedTop1Fixture(t, original, replication, 1400, 100, 100, 899)
	options := SeedReplicationOptions{RunsDir: runsDir, OriginalRunID: original.layout.ID, ReplicationRunID: replication.layout.ID}
	if _, err := ValidateSeedReplication(options); err == nil || !strings.Contains(err.Error(), "2,500 validation states") {
		t.Fatalf("invalid paired counts error = %v", err)
	}
	options.OutputPath = filepath.Join(t.TempDir(), "production-training-report.md")
	if _, err := WriteSeedReplication(options); err == nil || !strings.Contains(err.Error(), "must not overwrite") {
		t.Fatalf("production report overwrite error = %v", err)
	}
}

func writePairedTop1Fixture(t *testing.T, original, replication fixtureRun, both, originalOnly, replicationOnly, neither int) {
	t.Helper()
	writeJSON(t, filepath.Join(replication.layout.EvaluationsDir, pairedTop1ArtifactName), pairedTop1{
		OriginalRunID: original.layout.ID, ReplicationRunID: replication.layout.ID,
		OriginalSeed: original.config.Seed, ReplicationSeed: replication.config.Seed,
		OriginalBestUpdate: original.final.BestValidationStep, ReplicationBestUpdate: replication.final.BestValidationStep,
		ValidationSplitHash: original.evaluation.ValidationSplitHash, Examples: 2500,
		BothTeacherTop1: both, OriginalOnly: originalOnly, ReplicationOnly: replicationOnly, NeitherTeacherTop1: neither,
		OriginalSelectionHash: strings.Repeat("a", 64), ReplicationHash: strings.Repeat("b", 64),
	})
}
