package proofrun

import (
	"math"
	"strings"
	"testing"

	"github.com/sam-bee/wordle-ml_machine-learning/tensorboard"
)

func TestVerifyTensorBoardEventsForFixedStages(t *testing.T) {
	for _, stage := range []Stage{Overfit, Mini, Full} {
		t.Run(string(stage), func(t *testing.T) {
			eventsDir := t.TempDir()
			config, err := ConfigFor(stage)
			if err != nil {
				t.Fatal(err)
			}
			if stage == Mini {
				writeProofEventSegment(t, eventsDir, config, 10, 500, 0)
				writeProofEventSegment(t, eventsDir, config, 510, config.TargetUpdates, 600)
			} else {
				writeProofEventSegment(t, eventsDir, config, 10, config.TargetUpdates, 0)
			}

			proof, err := VerifyTensorBoardEvents(eventsDir, stage)
			if err != nil {
				t.Fatalf("VerifyTensorBoardEvents() error = %v", err)
			}
			if got, want := proof.TrainingSteps[0], int64(10); got != want {
				t.Errorf("first training step = %d, want %d", got, want)
			}
			if got, want := proof.TrainingSteps[len(proof.TrainingSteps)-1], config.TargetUpdates; got != want {
				t.Errorf("last training step = %d, want %d", got, want)
			}
			if got, want := len(proof.ValidationSteps), int(config.TargetUpdates/config.ValidationEvery)+1; got != want {
				t.Errorf("validation cadence entries = %d, want %d", got, want)
			}
			if stage == Full {
				if proof.TrainingLossTrend == nil || proof.TrainingLossTrend.FinalFiveMean >= proof.TrainingLossTrend.FirstFiveMean || proof.TrainingLossTrend.LeastSquaresSlope >= 0 {
					t.Errorf("full loss trend = %+v, want downward evidence", proof.TrainingLossTrend)
				}
			} else if proof.TrainingLossTrend != nil {
				t.Errorf("%s loss trend = %+v, want no full-stage evidence", stage, proof.TrainingLossTrend)
			}
		})
	}
}

func TestVerifyMiniTensorBoardEventsProvesContinuousResumedRun(t *testing.T) {
	eventsDir := t.TempDir()
	config, err := ConfigFor(Mini)
	if err != nil {
		t.Fatal(err)
	}
	writeProofEventSegment(t, eventsDir, config, 10, 500, 0)
	writeProofEventSegment(t, eventsDir, config, 510, 1000, 600)

	proof, err := VerifyMiniTensorBoardEvents(eventsDir)
	if err != nil {
		t.Fatalf("VerifyMiniTensorBoardEvents() error = %v", err)
	}
	if got, want := proof.TrainingSteps[49], int64(500); got != want {
		t.Errorf("checkpoint training step = %d, want %d", got, want)
	}
	if got, want := proof.TrainingSteps[50], int64(510); got != want {
		t.Errorf("first resumed training step = %d, want %d", got, want)
	}
	for _, tag := range miniValidationHistogramTags {
		if got, want := len(proof.HistogramStepsByTag[tag]), 11; got != want {
			t.Errorf("%s histogram cadence entries = %d, want %d", tag, got, want)
		}
	}
}

func TestVerifyTensorBoardEventsRejectsMissingRequiredTag(t *testing.T) {
	eventsDir := t.TempDir()
	config, err := ConfigFor(Overfit)
	if err != nil {
		t.Fatal(err)
	}
	writeProofEventSegment(t, eventsDir, config, 10, config.TargetUpdates, 0)
	inspection, err := tensorboard.InspectDir(eventsDir)
	if err != nil {
		t.Fatal(err)
	}
	inspection.Scalars = withoutScalarTag(inspection.Scalars, "optimizer/parameter_norm")
	_, err = verifyTensorBoardInspection(inspection, Overfit)
	if err == nil || !strings.Contains(err.Error(), "optimizer/parameter_norm") {
		t.Fatalf("verifyTensorBoardInspection() error = %v, want required-tag rejection", err)
	}
}

func TestVerifyMiniTensorBoardEventsRejectsTrainingReset(t *testing.T) {
	eventsDir := t.TempDir()
	config, err := ConfigFor(Mini)
	if err != nil {
		t.Fatal(err)
	}
	writeProofEventSegment(t, eventsDir, config, 10, 500, 0)
	writeProofEventSegment(t, eventsDir, config, 510, 1000, 600)
	writer, err := tensorboard.New(eventsDir)
	if err != nil {
		t.Fatal(err)
	}
	if err := writer.WriteScalars(0, tensorboard.Scalar{Tag: "train/loss", Value: 9}); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}

	_, err = VerifyMiniTensorBoardEvents(eventsDir)
	if err == nil || !strings.Contains(err.Error(), "train/loss") {
		t.Fatalf("VerifyMiniTensorBoardEvents() error = %v, want reset rejection", err)
	}
}

func TestVerifyFullTensorBoardEventsRejectsNonFiniteOrNonDecreasingLoss(t *testing.T) {
	config, err := ConfigFor(Full)
	if err != nil {
		t.Fatal(err)
	}
	inspection := completeInspection(config, 1)
	for index := range inspection.Scalars {
		if inspection.Scalars[index].Tag == "train/loss" {
			inspection.Scalars[index].Value = float32(index)
		}
	}
	_, err = verifyTensorBoardInspection(inspection, Full)
	if err == nil || !strings.Contains(err.Error(), "final five-point mean") {
		t.Fatalf("verifyTensorBoardInspection() error = %v, want non-decreasing-loss rejection", err)
	}
	for index := range inspection.Scalars {
		if inspection.Scalars[index].Tag == "train/loss" {
			inspection.Scalars[index].Value = float32(math.NaN())
			break
		}
	}
	_, err = verifyTensorBoardInspection(inspection, Full)
	if err == nil || !strings.Contains(err.Error(), "non-finite") {
		t.Fatalf("verifyTensorBoardInspection() error = %v, want non-finite-loss rejection", err)
	}
}

func TestVerifyGameTensorBoardEvents(t *testing.T) {
	eventsDir := t.TempDir()
	writer, err := tensorboard.New(eventsDir)
	if err != nil {
		t.Fatal(err)
	}
	scalars := make([]tensorboard.Scalar, 0, len(gameScalarTags))
	for _, tag := range gameScalarTags {
		scalars = append(scalars, tensorboard.Scalar{Tag: tag, Value: 1})
	}
	if err := writer.WriteScalars(0, scalars...); err != nil {
		t.Fatal(err)
	}
	if err := writer.Close(); err != nil {
		t.Fatal(err)
	}
	if err := VerifyGameTensorBoardEvents(eventsDir, 0); err != nil {
		t.Fatalf("VerifyGameTensorBoardEvents() error = %v", err)
	}
}

func writeProofEventSegment(t *testing.T, eventsDir string, config Config, trainFirst, trainLast, validationFirst int64) {
	t.Helper()
	writer, err := tensorboard.New(eventsDir)
	if err != nil {
		t.Fatal(err)
	}
	defer func() {
		if err := writer.Close(); err != nil {
			t.Errorf("Close() error = %v", err)
		}
	}()
	for step := trainFirst; step <= trainLast; step += config.ScalarEvery {
		if err := writer.WriteScalars(step, taggedScalars(trainingTags(), float32(config.TargetUpdates-step+1))...); err != nil {
			t.Fatal(err)
		}
	}
	for step := validationFirst; step <= trainLast; step += config.ValidationEvery {
		if err := writer.WriteScalars(step, taggedScalars(validationTags(), float32(config.TargetUpdates-step+1))...); err != nil {
			t.Fatal(err)
		}
		histograms := make([]tensorboard.Histogram, 0, len(miniValidationHistogramTags))
		for _, tag := range miniValidationHistogramTags {
			histograms = append(histograms, tensorboard.Histogram{Tag: tag, Values: []float64{float64(step)}})
		}
		if err := writer.WriteHistograms(step, histograms...); err != nil {
			t.Fatal(err)
		}
	}
}

func completeInspection(config Config, loss float32) tensorboard.Inspection {
	inspection := tensorboard.Inspection{}
	for _, step := range expectedSteps(config.ScalarEvery, config.ScalarEvery, config.TargetUpdates) {
		for _, tag := range trainingTags() {
			value := float32(config.TargetUpdates - step + 1)
			if tag == "train/loss" {
				value = loss
			}
			inspection.Scalars = append(inspection.Scalars, tensorboard.ScalarRecord{Tag: tag, Step: step, Value: value})
		}
	}
	for _, step := range expectedSteps(0, config.ValidationEvery, config.TargetUpdates) {
		for _, tag := range validationTags() {
			inspection.Scalars = append(inspection.Scalars, tensorboard.ScalarRecord{Tag: tag, Step: step, Value: 1})
		}
		for _, tag := range miniValidationHistogramTags {
			inspection.Histograms = append(inspection.Histograms, tensorboard.HistogramRecord{Tag: tag, Step: step})
		}
	}
	return inspection
}

func trainingTags() []string {
	tags := append([]string{}, trainingScalarTags...)
	return append(tags, openingTrainingAndValidationTags...)
}

func validationTags() []string {
	tags := append([]string{}, validationScalarTags...)
	return append(tags, openingTrainingAndValidationTags...)
}

func taggedScalars(tags []string, loss float32) []tensorboard.Scalar {
	scalars := make([]tensorboard.Scalar, 0, len(tags))
	for _, tag := range tags {
		value := float32(1)
		if tag == "train/loss" {
			value = loss
		}
		scalars = append(scalars, tensorboard.Scalar{Tag: tag, Value: value})
	}
	return scalars
}

func withoutScalarTag(records []tensorboard.ScalarRecord, tag string) []tensorboard.ScalarRecord {
	filtered := make([]tensorboard.ScalarRecord, 0, len(records))
	for _, record := range records {
		if record.Tag != tag {
			filtered = append(filtered, record)
		}
	}
	return filtered
}
