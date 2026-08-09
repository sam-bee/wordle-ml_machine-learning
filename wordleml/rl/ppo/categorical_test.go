package ppo

import (
	"math"
	"math/rand"
	"testing"
)

func TestCategoricalMasksActionsExactly(t *testing.T) {
	distribution, err := NewCategorical([]float32{1, 1000, 2}, []float32{1, 0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if distribution.Probability(1) != 0 || !math.IsInf(distribution.LogProbability(1), -1) {
		t.Fatalf("masked action has probability/logp %v/%v", distribution.Probability(1), distribution.LogProbability(1))
	}
	if got, err := distribution.Greedy(); err != nil || got != 2 {
		t.Fatalf("greedy = %d, %v; want action 2", got, err)
	}
	rng := rand.New(rand.NewSource(7))
	for range 10_000 {
		action, err := distribution.Sample(rng)
		if err != nil {
			t.Fatal(err)
		}
		if action == 1 {
			t.Fatal("masked action was sampled")
		}
	}
}

func TestCategoricalSamplingIsSeedReproducible(t *testing.T) {
	distribution, err := NewCategorical([]float32{0, 0, 0}, []float32{1, 1, 1})
	if err != nil {
		t.Fatal(err)
	}
	a := rand.New(rand.NewSource(99))
	b := rand.New(rand.NewSource(99))
	for range 100 {
		left, _ := distribution.Sample(a)
		right, _ := distribution.Sample(b)
		if left != right {
			t.Fatalf("same seed diverged: %d != %d", left, right)
		}
	}
}

func TestReferenceKL(t *testing.T) {
	reference, _ := NewCategorical([]float32{0, 0}, []float32{1, 1})
	same, _ := NewCategorical([]float32{10, 10}, []float32{1, 1})
	if got, err := ReferenceKL(reference, same); err != nil || got > 1e-12 {
		t.Fatalf("identical distributions KL = %v, %v", got, err)
	}
	different, _ := NewCategorical([]float32{2, -2}, []float32{1, 1})
	if got, err := ReferenceKL(reference, different); err != nil || got <= 0 {
		t.Fatalf("different distributions KL = %v, %v", got, err)
	}
}
