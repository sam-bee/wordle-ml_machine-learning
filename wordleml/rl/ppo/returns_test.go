package ppo

import (
	"math"
	"testing"
)

func TestRewardForTransition(t *testing.T) {
	tests := []struct {
		name             string
		solved, terminal bool
		want             float64
	}{
		{name: "non-terminal", want: NonTerminalReward},
		{name: "solve", solved: true, terminal: true, want: SolveReward},
		{name: "failure", terminal: true, want: FailureReward},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := RewardForTransition(test.solved, test.terminal); got != test.want {
				t.Fatalf("RewardForTransition(%v, %v) = %v, want %v", test.solved, test.terminal, got, test.want)
			}
		})
	}
}

func TestDiscountedReturnsWordleOutcomes(t *testing.T) {
	for solveTurn := 1; solveTurn <= 6; solveTurn++ {
		t.Run("solve turn "+string(rune('0'+solveTurn)), func(t *testing.T) {
			rewards := make([]float64, solveTurn)
			for i := 0; i < solveTurn-1; i++ {
				rewards[i] = NonTerminalReward
			}
			rewards[solveTurn-1] = SolveReward

			returns, err := DiscountedReturns(rewards, 1)
			if err != nil {
				t.Fatal(err)
			}
			wantInitial := SolveReward + float64(solveTurn-1)*NonTerminalReward
			if !almostEqual(returns[0], wantInitial) {
				t.Fatalf("initial return = %.12g, want %.12g", returns[0], wantInitial)
			}
			if !almostEqual(returns[solveTurn-1], SolveReward) {
				t.Fatalf("terminal solve return = %.12g, want %.12g", returns[solveTurn-1], SolveReward)
			}
		})
	}

	t.Run("sixth turn failure", func(t *testing.T) {
		rewards := []float64{NonTerminalReward, NonTerminalReward, NonTerminalReward, NonTerminalReward, NonTerminalReward, FailureReward}
		returns, err := DiscountedReturns(rewards, 1)
		if err != nil {
			t.Fatal(err)
		}
		if want := FailureReward + 5*NonTerminalReward; !almostEqual(returns[0], want) {
			t.Fatalf("initial failure return = %.12g, want %.12g", returns[0], want)
		}
		if !almostEqual(returns[5], FailureReward) {
			t.Fatalf("terminal failure return = %.12g, want %.12g", returns[5], FailureReward)
		}
	})
}

func TestDiscountedReturnsUsesGamma(t *testing.T) {
	got, err := DiscountedReturns([]float64{1, 2, 3}, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	want := []float64{2.75, 3.5, 3}
	assertFloatSlice(t, got, want)
}

func TestGeneralizedAdvantagesHandComputed(t *testing.T) {
	// delta_1 = 2 - 0.5 = 1.5 (terminal),
	// delta_0 = 1 + 0.9*0.5 - 0.2 = 1.25,
	// A_0 = 1.25 + 0.9*0.8*1.5 = 2.33.
	got, err := GeneralizedAdvantages(
		[]float64{1, 2},
		[]float64{0.2, 0.5, 0},
		[]bool{false, true},
		0.9,
		0.8,
	)
	if err != nil {
		t.Fatal(err)
	}
	assertFloatSlice(t, got, []float64{2.33, 1.5})
}

func TestGeneralizedAdvantagesTerminalDoesNotBootstrapNextEpisode(t *testing.T) {
	got, err := GeneralizedAdvantages(
		[]float64{SolveReward, NonTerminalReward},
		[]float64{0.25, 1234, 0},
		[]bool{true, true},
		1,
		0.95,
	)
	if err != nil {
		t.Fatal(err)
	}
	// The huge value is the opening value of a following episode and must be
	// ignored after a terminal transition.
	assertFloatSlice(t, got, []float64{0.75, NonTerminalReward - 1234})
}

func TestNormalizeAdvantagesIsFiniteForConstantBatch(t *testing.T) {
	normalized, mean, stddev, err := NormalizeAdvantages([]float64{2, 2, 2})
	if err != nil {
		t.Fatal(err)
	}
	if mean != 2 || stddev != 0 {
		t.Fatalf("statistics = mean %v stddev %v, want 2, 0", mean, stddev)
	}
	assertFloatSlice(t, normalized, []float64{0, 0, 0})
	if err := CheckFinite("normalized", normalized); err != nil {
		t.Fatalf("normalized advantages should be finite: %v", err)
	}
}

func TestFiniteValidation(t *testing.T) {
	if _, err := DiscountedReturns([]float64{math.NaN()}, 1); err == nil {
		t.Fatal("DiscountedReturns accepted NaN reward")
	}
	if _, err := GeneralizedAdvantages([]float64{1}, []float64{0, math.Inf(1)}, []bool{true}, 1, 0.95); err == nil {
		t.Fatal("GeneralizedAdvantages accepted infinite value")
	}
	if _, _, _, err := NormalizeAdvantages([]float64{1, math.Inf(-1)}); err == nil {
		t.Fatal("NormalizeAdvantages accepted infinite advantage")
	}
}

func assertFloatSlice(t *testing.T, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length = %d, want %d", len(got), len(want))
	}
	for i := range want {
		if !almostEqual(got[i], want[i]) {
			t.Fatalf("value[%d] = %.12g, want %.12g", i, got[i], want[i])
		}
	}
}

func almostEqual(got, want float64) bool {
	return math.Abs(got-want) < 1e-10
}
