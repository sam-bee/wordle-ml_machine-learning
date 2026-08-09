// Package ppo contains the small, model-independent mathematical pieces of
// the PPO experiment. Model execution, action masking and checkpointing belong
// to the surrounding RL packages; keeping the calculations here makes their
// invariants straightforward to test on the host.
package ppo

import (
	"fmt"
	"math"
)

// Wordle's fixed six-turn game has this completed-game objective. A non-final
// accepted guess is mildly costly; the terminal transition receives either the
// solve or failure reward, never the non-terminal reward.
const (
	NonTerminalReward = -0.05
	SolveReward       = 1.0
	FailureReward     = -1.0
)

// RewardForTransition returns the unshaped PPO reward for an authoritative
// game-engine transition. A solve is necessarily terminal in Wordle. The
// terminal argument makes the failure case explicit and is retained so a
// collector cannot accidentally give a sixth-turn failure a step reward.
func RewardForTransition(solved, terminal bool) float64 {
	if solved {
		return SolveReward
	}
	if terminal {
		return FailureReward
	}
	return NonTerminalReward
}

// DiscountedReturns calculates G_t = r_t + gamma * G_(t+1). It is intended
// for complete trajectories, so the return beyond the final reward is zero.
func DiscountedReturns(rewards []float64, gamma float64) ([]float64, error) {
	if err := validateDiscount(gamma, "gamma"); err != nil {
		return nil, err
	}
	if err := CheckFinite("rewards", rewards); err != nil {
		return nil, err
	}

	returns := make([]float64, len(rewards))
	var running float64
	for i := len(rewards) - 1; i >= 0; i-- {
		running = rewards[i] + gamma*running
		if !isFinite(running) {
			return nil, fmt.Errorf("discounted return at index %d is not finite: %v", i, running)
		}
		returns[i] = running
	}
	return returns, nil
}

// GeneralizedAdvantages calculates the standard GAE(lambda) estimates:
//
//	delta_t = r_t + gamma * (1-done_t) * V_(t+1) - V_t
//	A_t     = delta_t + gamma * lambda * (1-done_t) * A_(t+1)
//
// Values contains one value for every transition plus the bootstrap value
// after the final transition. A complete Wordle game has a terminal final
// transition and therefore uses zero as that final bootstrap value. Supplying
// it explicitly also makes this helper correct for a deliberately truncated
// rollout.
func GeneralizedAdvantages(rewards, values []float64, terminals []bool, gamma, lambda float64) ([]float64, error) {
	if len(rewards) == 0 {
		return nil, fmt.Errorf("GAE requires at least one transition")
	}
	if len(values) != len(rewards)+1 {
		return nil, fmt.Errorf("GAE values length %d, want rewards length + 1 (%d)", len(values), len(rewards)+1)
	}
	if len(terminals) != len(rewards) {
		return nil, fmt.Errorf("GAE terminals length %d, want rewards length %d", len(terminals), len(rewards))
	}
	if err := validateDiscount(gamma, "gamma"); err != nil {
		return nil, err
	}
	if err := validateDiscount(lambda, "lambda"); err != nil {
		return nil, err
	}
	if err := CheckFinite("rewards", rewards); err != nil {
		return nil, err
	}
	if err := CheckFinite("values", values); err != nil {
		return nil, err
	}

	advantages := make([]float64, len(rewards))
	var nextAdvantage float64
	for i := len(rewards) - 1; i >= 0; i-- {
		continuation := 1.0
		if terminals[i] {
			continuation = 0
		}
		delta := rewards[i] + gamma*continuation*values[i+1] - values[i]
		nextAdvantage = delta + gamma*lambda*continuation*nextAdvantage
		if !isFinite(nextAdvantage) {
			return nil, fmt.Errorf("GAE advantage at index %d is not finite: %v", i, nextAdvantage)
		}
		advantages[i] = nextAdvantage
	}
	return advantages, nil
}

// AdvantageNormalizationEpsilon prevents a constant advantage batch from
// producing a division by zero. It is intentionally small relative to normal
// advantage magnitudes and is part of the reproducible PPO configuration.
const AdvantageNormalizationEpsilon = 1e-8

// NormalizeAdvantages returns population-standard-deviation normalized
// advantages and the unnormalized batch mean and standard deviation. Constant
// batches become finite zeros, which correctly provide no actor update.
func NormalizeAdvantages(advantages []float64) (normalized []float64, mean, stddev float64, err error) {
	if len(advantages) == 0 {
		return nil, 0, 0, fmt.Errorf("cannot normalize an empty advantage batch")
	}
	if err := CheckFinite("advantages", advantages); err != nil {
		return nil, 0, 0, err
	}

	for _, advantage := range advantages {
		mean += advantage
	}
	mean /= float64(len(advantages))
	for _, advantage := range advantages {
		delta := advantage - mean
		stddev += delta * delta
	}
	stddev = math.Sqrt(stddev / float64(len(advantages)))
	if !isFinite(mean) || !isFinite(stddev) {
		return nil, 0, 0, fmt.Errorf("advantage statistics are not finite: mean=%v stddev=%v", mean, stddev)
	}

	normalized = make([]float64, len(advantages))
	denominator := stddev + AdvantageNormalizationEpsilon
	for i, advantage := range advantages {
		normalized[i] = (advantage - mean) / denominator
		if !isFinite(normalized[i]) {
			return nil, 0, 0, fmt.Errorf("normalized advantage at index %d is not finite: %v", i, normalized[i])
		}
	}
	return normalized, mean, stddev, nil
}

// CheckFinite reports the first NaN or infinity in values. It is deliberately
// shared by the loss helpers so callers can use the same guard for rewards,
// values, returns, advantages, losses, gradients and flattened parameters.
func CheckFinite(name string, values []float64) error {
	for i, value := range values {
		if !isFinite(value) {
			return fmt.Errorf("%s[%d] is not finite: %v", name, i, value)
		}
	}
	return nil
}

func validateDiscount(value float64, name string) error {
	if !isFinite(value) || value < 0 || value > 1 {
		return fmt.Errorf("%s must be finite and in [0, 1], got %v", name, value)
	}
	return nil
}

func isFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}
