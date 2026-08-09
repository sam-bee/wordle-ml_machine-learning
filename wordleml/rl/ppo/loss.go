package ppo

import (
	"fmt"
	"math"
)

// RatioDiagnostics are the scalar quantities derived from one sampled action
// when the candidate policy is compared with the frozen rollout policy.
type RatioDiagnostics struct {
	Ratio        float64
	ClippedRatio float64
	Clipped      bool
}

// ProbabilityRatio returns pi_theta(a|s) / pi_old(a|s), calculated from
// masked log-probabilities. Both values must have been computed with exactly
// the action mask retained in the rollout transition.
func ProbabilityRatio(currentLogProbability, oldLogProbability float64) (float64, error) {
	if !isFinite(currentLogProbability) || !isFinite(oldLogProbability) {
		return 0, fmt.Errorf("log-probabilities must be finite: current=%v old=%v", currentLogProbability, oldLogProbability)
	}
	ratio := math.Exp(currentLogProbability - oldLogProbability)
	if !isFinite(ratio) {
		return 0, fmt.Errorf("probability ratio is not finite")
	}
	return ratio, nil
}

// DiagnoseRatio calculates the raw and clipped policy ratio for one action.
// clipRange is the PPO epsilon and must be in [0, 1).
func DiagnoseRatio(currentLogProbability, oldLogProbability, clipRange float64) (RatioDiagnostics, error) {
	if err := validateClipRange(clipRange); err != nil {
		return RatioDiagnostics{}, err
	}
	ratio, err := ProbabilityRatio(currentLogProbability, oldLogProbability)
	if err != nil {
		return RatioDiagnostics{}, err
	}
	minimum, maximum := 1-clipRange, 1+clipRange
	clipped := math.Min(math.Max(ratio, minimum), maximum)
	return RatioDiagnostics{
		Ratio:        ratio,
		ClippedRatio: clipped,
		Clipped:      ratio < minimum || ratio > maximum,
	}, nil
}

// ClippedSurrogateObjective is the single-sample PPO objective to maximize:
//
//	min(ratio * advantage, clip(ratio, 1-epsilon, 1+epsilon) * advantage)
//
// A positive advantage therefore rewards increasing the selected-action
// probability; a negative advantage rewards decreasing it. Optimisers which
// minimise losses should minimise the negative of this value.
func ClippedSurrogateObjective(currentLogProbability, oldLogProbability, advantage, clipRange float64) (float64, RatioDiagnostics, error) {
	if !isFinite(advantage) {
		return 0, RatioDiagnostics{}, fmt.Errorf("advantage must be finite, got %v", advantage)
	}
	diagnostics, err := DiagnoseRatio(currentLogProbability, oldLogProbability, clipRange)
	if err != nil {
		return 0, RatioDiagnostics{}, err
	}
	objective := math.Min(diagnostics.Ratio*advantage, diagnostics.ClippedRatio*advantage)
	if !isFinite(objective) {
		return 0, RatioDiagnostics{}, fmt.Errorf("clipped surrogate objective is not finite")
	}
	return objective, diagnostics, nil
}

// PPOActorLoss returns the scalar contribution for a loss-minimising
// optimiser: the negative clipped surrogate objective.
func PPOActorLoss(currentLogProbability, oldLogProbability, advantage, clipRange float64) (float64, RatioDiagnostics, error) {
	objective, diagnostics, err := ClippedSurrogateObjective(currentLogProbability, oldLogProbability, advantage, clipRange)
	if err != nil {
		return 0, RatioDiagnostics{}, err
	}
	return -objective, diagnostics, nil
}

// ApproximateOldPolicyKL returns the non-negative k3 sampled KL estimator
// mean(exp(log_ratio) - 1 - log_ratio). It is suitable for PPO epoch early
// stopping and is exactly zero when the candidate and rollout policies match.
func ApproximateOldPolicyKL(currentLogProbabilities, oldLogProbabilities []float64) (float64, error) {
	if len(currentLogProbabilities) == 0 {
		return 0, fmt.Errorf("cannot calculate KL for an empty batch")
	}
	if len(currentLogProbabilities) != len(oldLogProbabilities) {
		return 0, fmt.Errorf("current log-probability length %d does not match old length %d", len(currentLogProbabilities), len(oldLogProbabilities))
	}
	var total float64
	for i := range currentLogProbabilities {
		ratio, err := ProbabilityRatio(currentLogProbabilities[i], oldLogProbabilities[i])
		if err != nil {
			return 0, fmt.Errorf("KL sample %d: %w", i, err)
		}
		logRatio := currentLogProbabilities[i] - oldLogProbabilities[i]
		term := ratio - 1 - logRatio
		if !isFinite(term) {
			return 0, fmt.Errorf("KL sample %d is not finite", i)
		}
		// The expression is mathematically non-negative. Clamp only a
		// cancellation-sized negative result so metric consumers may safely
		// treat this diagnostic as a KL-like non-negative quantity.
		if term < 0 {
			term = 0
		}
		total += term
	}
	kl := total / float64(len(currentLogProbabilities))
	if !isFinite(kl) {
		return 0, fmt.Errorf("old-policy KL is not finite")
	}
	return kl, nil
}

// ClipFraction reports the fraction of sampled actions whose raw ratio lies
// outside PPO's clipping interval.
func ClipFraction(currentLogProbabilities, oldLogProbabilities []float64, clipRange float64) (float64, error) {
	if len(currentLogProbabilities) == 0 {
		return 0, fmt.Errorf("cannot calculate clip fraction for an empty batch")
	}
	if len(currentLogProbabilities) != len(oldLogProbabilities) {
		return 0, fmt.Errorf("current log-probability length %d does not match old length %d", len(currentLogProbabilities), len(oldLogProbabilities))
	}
	if err := validateClipRange(clipRange); err != nil {
		return 0, err
	}
	var clipped int
	for i := range currentLogProbabilities {
		diagnostics, err := DiagnoseRatio(currentLogProbabilities[i], oldLogProbabilities[i], clipRange)
		if err != nil {
			return 0, fmt.Errorf("clip fraction sample %d: %w", i, err)
		}
		if diagnostics.Clipped {
			clipped++
		}
	}
	return float64(clipped) / float64(len(currentLogProbabilities)), nil
}

func validateClipRange(clipRange float64) error {
	if !isFinite(clipRange) || clipRange < 0 || clipRange >= 1 {
		return fmt.Errorf("clip range must be finite and in [0, 1), got %v", clipRange)
	}
	return nil
}
