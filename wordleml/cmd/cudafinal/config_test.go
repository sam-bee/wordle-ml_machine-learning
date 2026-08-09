//go:build cuda_cgo

package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/cudamodel"
)

func TestParseConfigRequiresExplicitFinalTestConfirmation(t *testing.T) {
	_, err := parseConfig([]string{"-model-dir=model"}, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "confirm-final-test") {
		t.Fatalf("parseConfig error = %v, want confirmation rejection", err)
	}
}

func TestAuthorizedModelAndFinalTestPins(t *testing.T) {
	manifest := cudamodel.Manifest{
		RunID: authorizedRunID, Checkpoint: authorizedCheckpoint, CheckpointUpdate: authorizedCheckpointUpdate,
		TrainingCommit: authorizedTrainingCommit, WeightsSHA256: authorizedWeightsSHA256,
	}
	if err := validateAuthorizedModel(manifest); err != nil {
		t.Fatalf("validateAuthorizedModel: %v", err)
	}
	manifest.WeightsSHA256 = strings.Repeat("0", 64)
	if err := validateAuthorizedModel(manifest); err == nil {
		t.Fatal("validateAuthorizedModel accepted wrong weights")
	}
	if !authorizedFinalTest(authorizedFinalTestSHA256, 100) || authorizedFinalTest(authorizedFinalTestSHA256, 99) || authorizedFinalTest(strings.Repeat("0", 64), 100) {
		t.Fatal("final-test authorization pin accepted/rejected the wrong identity")
	}
}

func TestParseConfigAcceptsOnlyTheFixedReportContract(t *testing.T) {
	configuration, err := parseConfig([]string{"-confirm-final-test", "-model-dir=model"}, &bytes.Buffer{})
	if err != nil {
		t.Fatalf("parseConfig: %v", err)
	}
	if configuration.modelDir != "model" {
		t.Fatalf("model dir = %q, want model", configuration.modelDir)
	}
}

func TestParseConfigRejectsReportAndCommitOverrides(t *testing.T) {
	_, err := parseConfig([]string{
		"-confirm-final-test", "-model-dir=model", "-report=/tmp/another-claim", "-evaluator-commit=" + strings.Repeat("a", 40),
	}, &bytes.Buffer{})
	if err == nil || !strings.Contains(err.Error(), "flag provided but not defined") {
		t.Fatalf("parseConfig error = %v, want unsupported override rejection", err)
	}
}

func TestEmbeddedEvaluatorCommitIsRequired(t *testing.T) {
	prior := evaluatorCommit
	t.Cleanup(func() { evaluatorCommit = prior })
	evaluatorCommit = ""
	if _, err := embeddedEvaluatorCommit(); err == nil {
		t.Fatal("embeddedEvaluatorCommit accepted an empty direct-build value")
	}
	evaluatorCommit = strings.Repeat("a", 40)
	if got, err := embeddedEvaluatorCommit(); err != nil || got != evaluatorCommit {
		t.Fatalf("embeddedEvaluatorCommit = %q, %v", got, err)
	}
}
