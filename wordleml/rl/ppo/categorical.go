package ppo

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
)

// Categorical is a host-side masked categorical distribution. Logits are
// never modified in place, and unavailable actions receive exactly zero
// probability. The collector uses this type for both deterministic greedy
// deployment and reproducible stochastic on-policy action selection.
type Categorical struct {
	probabilities []float64
	logProbs      []float64
	entropy       float64
}

// NewCategorical constructs a numerically stable categorical distribution
// from one policy-logit row and the exact action-availability mask stored with
// the transition.
func NewCategorical(logits, availability []float32) (*Categorical, error) {
	if len(logits) == 0 || len(logits) != len(availability) {
		return nil, fmt.Errorf("logit/availability lengths are %d/%d", len(logits), len(availability))
	}
	maximum := math.Inf(-1)
	legal := 0
	for action, logit32 := range logits {
		logit := float64(logit32)
		if math.IsNaN(logit) || math.IsInf(logit, 0) {
			return nil, fmt.Errorf("logit %d is not finite: %v", action, logit)
		}
		switch availability[action] {
		case 0:
		case 1:
			legal++
			maximum = math.Max(maximum, logit)
		default:
			return nil, fmt.Errorf("availability %d is %v, want exactly zero or one", action, availability[action])
		}
	}
	if legal == 0 {
		return nil, errors.New("categorical has no available action")
	}

	d := &Categorical{
		probabilities: make([]float64, len(logits)),
		logProbs:      make([]float64, len(logits)),
	}
	for action := range d.logProbs {
		d.logProbs[action] = math.Inf(-1)
	}
	var normalizer float64
	for action, logit32 := range logits {
		if availability[action] == 0 {
			continue
		}
		weight := math.Exp(float64(logit32) - maximum)
		d.probabilities[action] = weight
		normalizer += weight
	}
	if normalizer <= 0 || math.IsNaN(normalizer) || math.IsInf(normalizer, 0) {
		return nil, fmt.Errorf("categorical normalizer is not finite and positive: %v", normalizer)
	}
	logNormalizer := maximum + math.Log(normalizer)
	for action, logit32 := range logits {
		if availability[action] == 0 {
			continue
		}
		probability := d.probabilities[action] / normalizer
		logProbability := float64(logit32) - logNormalizer
		d.probabilities[action] = probability
		d.logProbs[action] = logProbability
		d.entropy -= probability * logProbability
	}
	if math.IsNaN(d.entropy) || math.IsInf(d.entropy, 0) || d.entropy < 0 {
		return nil, fmt.Errorf("categorical entropy is invalid: %v", d.entropy)
	}
	return d, nil
}

// Probability returns the exact probability of action, including zero for an
// unavailable action.
func (d *Categorical) Probability(action int) float64 {
	if d == nil || action < 0 || action >= len(d.probabilities) {
		return 0
	}
	return d.probabilities[action]
}

// LogProbability returns the selected-action log probability. Unavailable or
// out-of-range actions return negative infinity and must never be stored by a
// valid collector.
func (d *Categorical) LogProbability(action int) float64 {
	if d == nil || action < 0 || action >= len(d.logProbs) {
		return math.Inf(-1)
	}
	return d.logProbs[action]
}

// Entropy returns the categorical entropy in nats.
func (d *Categorical) Entropy() float64 {
	if d == nil {
		return 0
	}
	return d.entropy
}

// Greedy returns the lowest-ID action among exact maximum logits/probabilities.
func (d *Categorical) Greedy() (int, error) {
	if d == nil || len(d.probabilities) == 0 {
		return 0, errors.New("categorical is nil or empty")
	}
	best := -1
	bestProbability := math.Inf(-1)
	for action, probability := range d.probabilities {
		if probability > bestProbability && probability > 0 {
			best = action
			bestProbability = probability
		}
	}
	if best < 0 {
		return 0, errors.New("categorical has no selectable action")
	}
	return best, nil
}

// Sample draws one action from the distribution using only rng. Supplying a
// locally seeded generator makes rollout reproduction independent of global
// process randomness.
func (d *Categorical) Sample(rng *rand.Rand) (int, error) {
	if d == nil || len(d.probabilities) == 0 {
		return 0, errors.New("categorical is nil or empty")
	}
	if rng == nil {
		return 0, errors.New("categorical sampler requires a local RNG")
	}
	target := rng.Float64()
	var cumulative float64
	last := -1
	for action, probability := range d.probabilities {
		if probability == 0 {
			continue
		}
		last = action
		cumulative += probability
		if target < cumulative {
			return action, nil
		}
	}
	// Floating-point summation may end a few ulps below one. The final legal
	// action owns that residual interval.
	if last >= 0 {
		return last, nil
	}
	return 0, errors.New("categorical has no selectable action")
}

// ReferenceKL returns KL(reference || current) using two distributions built
// with the same exact action mask. This is the experiment's conservative
// cumulative-drift measurement and actor penalty.
func ReferenceKL(reference, current *Categorical) (float64, error) {
	if reference == nil || current == nil || len(reference.probabilities) != len(current.probabilities) {
		return 0, errors.New("reference/current categorical dimensions differ")
	}
	var kl float64
	for action, probability := range reference.probabilities {
		if probability == 0 {
			continue
		}
		currentProbability := current.probabilities[action]
		if currentProbability <= 0 {
			return 0, fmt.Errorf("current policy assigns zero probability to reference action %d", action)
		}
		kl += probability * (reference.logProbs[action] - current.logProbs[action])
	}
	if kl < 0 && kl > -1e-12 {
		kl = 0
	}
	if math.IsNaN(kl) || math.IsInf(kl, 0) || kl < 0 {
		return 0, fmt.Errorf("reference KL is invalid: %v", kl)
	}
	return kl, nil
}
