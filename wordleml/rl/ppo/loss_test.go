package ppo

import (
	"math"
	"testing"
)

func TestProbabilityRatioIsOneBeforeUpdate(t *testing.T) {
	ratio, err := ProbabilityRatio(-1.234, -1.234)
	if err != nil {
		t.Fatal(err)
	}
	if ratio != 1 {
		t.Fatalf("ratio = %v, want 1", ratio)
	}
	kl, err := ApproximateOldPolicyKL([]float64{-1.234, -0.2}, []float64{-1.234, -0.2})
	if err != nil {
		t.Fatal(err)
	}
	if kl != 0 {
		t.Fatalf("KL = %v, want 0", kl)
	}
}

func TestClippedSurrogateFollowsAdvantageDirection(t *testing.T) {
	positive, _, err := ClippedSurrogateObjective(math.Log(1.05), 0, 1, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if positive <= 1 {
		t.Fatalf("positive advantage objective = %v, want greater than objective at ratio one", positive)
	}

	negative, _, err := ClippedSurrogateObjective(math.Log(1.05), 0, -1, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if negative >= -1 {
		t.Fatalf("negative advantage objective = %v, want less than objective at ratio one", negative)
	}
}

func TestClippedSurrogateActivatesForExaggeratedRatio(t *testing.T) {
	objective, diagnostics, err := ClippedSurrogateObjective(math.Log(1.5), 0, 2, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if !diagnostics.Clipped {
		t.Fatal("expected ratio to be clipped")
	}
	if !almostEqual(diagnostics.Ratio, 1.5) || !almostEqual(diagnostics.ClippedRatio, 1.1) {
		t.Fatalf("diagnostics = %#v, want ratio 1.5 and clipped ratio 1.1", diagnostics)
	}
	if !almostEqual(objective, 2.2) {
		t.Fatalf("objective = %v, want 2.2", objective)
	}
}

func TestClipFraction(t *testing.T) {
	fraction, err := ClipFraction(
		[]float64{0, math.Log(1.05), math.Log(1.3)},
		[]float64{0, 0, 0},
		0.1,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !almostEqual(fraction, 1.0/3.0) {
		t.Fatalf("clip fraction = %v, want 1/3", fraction)
	}
}

func TestLossValidationRejectsNonFiniteValues(t *testing.T) {
	if _, err := ProbabilityRatio(math.Inf(1), 0); err == nil {
		t.Fatal("ProbabilityRatio accepted infinity")
	}
	if _, _, err := ClippedSurrogateObjective(0, 0, math.NaN(), 0.1); err == nil {
		t.Fatal("ClippedSurrogateObjective accepted NaN advantage")
	}
	if _, err := ApproximateOldPolicyKL([]float64{0}, []float64{math.NaN()}); err == nil {
		t.Fatal("ApproximateOldPolicyKL accepted NaN")
	}
}
