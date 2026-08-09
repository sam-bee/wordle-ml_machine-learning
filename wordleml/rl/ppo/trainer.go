package ppo

import (
	"errors"
	"fmt"
	"math"
	"math/rand"

	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/supervised"
)

// UpdateConfig controls the bounded optimisation of one fresh rollout.
type UpdateConfig struct {
	Epochs        int
	MinibatchSize int
	TargetOldKL   float64
	ClipRange     float64
	Seed          int64
}

// PolicyDiagnostics are measured over the whole frozen rollout under its
// stored masks, never merely from the most recent minibatch.
type PolicyDiagnostics struct {
	PolicyLoss                  float64 `json:"policy_loss"`
	Entropy                     float64 `json:"entropy"`
	ApproxOldPolicyKL           float64 `json:"approx_old_policy_kl"`
	SupervisedReferenceKL       float64 `json:"supervised_reference_kl"`
	ClipFraction                float64 `json:"clip_fraction"`
	MeanRatio                   float64 `json:"mean_ratio"`
	MaximumAbsoluteRatioFromOne float64 `json:"maximum_absolute_ratio_from_one"`
}

// UpdateMetrics retain the candidate's PPO/value loss and all safety gates.
type UpdateMetrics struct {
	EpochsCompleted         int               `json:"epochs_completed"`
	StoppedEarlyForKL       bool              `json:"stopped_early_for_kl"`
	ActorMinibatchLossMean  float64           `json:"actor_minibatch_loss_mean"`
	CriticMinibatchLossMean float64           `json:"critic_minibatch_loss_mean"`
	Policy                  PolicyDiagnostics `json:"policy"`
	Critic                  ValueStatistics   `json:"critic"`
	ActorGradientNorm       float64           `json:"actor_gradient_norm"`
	CriticGradientNorm      float64           `json:"critic_gradient_norm"`
	GradientsFinite         bool              `json:"gradients_finite"`
	ParametersFinite        bool              `json:"parameters_finite"`
	NumericallyStable       bool              `json:"numerically_stable"`
}

// DiagnosePolicy recomputes action probabilities under the current actor,
// old-policy statistics, and permanent supervised reference using precisely
// each transition's stored action mask.
func DiagnosePolicy(current, reference *ActorSession, transitions []TrajectoryTransition, batchSize int, clipRange float64) (PolicyDiagnostics, error) {
	if current == nil || reference == nil || len(transitions) == 0 {
		return PolicyDiagnostics{}, errors.New("policy diagnostics require current/reference actors and transitions")
	}
	if batchSize <= 0 {
		batchSize = 256
	}
	currentLogProbs := make([]float64, len(transitions))
	oldLogProbs := make([]float64, len(transitions))
	var policyLoss, entropy, referenceKL, ratioSum, maximumRatioDelta float64
	for start := 0; start < len(transitions); start += batchSize {
		end := min(start+batchSize, len(transitions))
		inputs := make([]modelstate.Inputs, end-start)
		for offset, transition := range transitions[start:end] {
			if len(transition.ModelInputs.CandidateMask) == 0 || len(transition.Availability) == 0 {
				return PolicyDiagnostics{}, fmt.Errorf("transition %d has no retained actor observation", start+offset)
			}
			inputs[offset] = transition.ModelInputs
		}
		currentLogits, err := current.Predict(inputs)
		if err != nil {
			return PolicyDiagnostics{}, err
		}
		referenceLogits, err := reference.Predict(inputs)
		if err != nil {
			return PolicyDiagnostics{}, err
		}
		for offset, transition := range transitions[start:end] {
			index := start + offset
			currentDistribution, err := NewCategorical(currentLogits[offset], transition.Availability)
			if err != nil {
				return PolicyDiagnostics{}, err
			}
			referenceDistribution, err := NewCategorical(referenceLogits[offset], transition.Availability)
			if err != nil {
				return PolicyDiagnostics{}, err
			}
			action := int(transition.ActionID)
			currentLogProbs[index] = currentDistribution.LogProbability(action)
			oldLogProbs[index] = transition.OldLogProb
			loss, ratio, err := PPOActorLoss(currentLogProbs[index], oldLogProbs[index], transition.Advantage, clipRange)
			if err != nil {
				return PolicyDiagnostics{}, err
			}
			policyLoss += loss
			entropy += currentDistribution.Entropy()
			kl, err := ReferenceKL(referenceDistribution, currentDistribution)
			if err != nil {
				return PolicyDiagnostics{}, err
			}
			referenceKL += kl
			ratioSum += ratio.Ratio
			maximumRatioDelta = math.Max(maximumRatioDelta, math.Abs(ratio.Ratio-1))
		}
	}
	oldKL, err := ApproximateOldPolicyKL(currentLogProbs, oldLogProbs)
	if err != nil {
		return PolicyDiagnostics{}, err
	}
	clipFraction, err := ClipFraction(currentLogProbs, oldLogProbs, clipRange)
	if err != nil {
		return PolicyDiagnostics{}, err
	}
	size := float64(len(transitions))
	result := PolicyDiagnostics{
		PolicyLoss: policyLoss / size, Entropy: entropy / size, ApproxOldPolicyKL: oldKL,
		SupervisedReferenceKL: referenceKL / size, ClipFraction: clipFraction,
		MeanRatio: ratioSum / size, MaximumAbsoluteRatioFromOne: maximumRatioDelta,
	}
	if err := CheckFinite("policy diagnostics", []float64{result.PolicyLoss, result.Entropy, result.ApproxOldPolicyKL, result.SupervisedReferenceKL, result.ClipFraction, result.MeanRatio, result.MaximumAbsoluteRatioFromOne}); err != nil {
		return PolicyDiagnostics{}, err
	}
	return result, nil
}

// UpdateCandidate performs standard clipped PPO actor updates and value
// regression for one newly collected batch. It stops complete PPO epochs as
// soon as the whole-rollout old-policy KL exceeds the configured target.
func UpdateCandidate(candidate *ActorSession, critic *CriticSession, supervisedReference *ActorSession, rollout Rollout, config UpdateConfig) (UpdateMetrics, error) {
	if candidate == nil || critic == nil || supervisedReference == nil {
		return UpdateMetrics{}, errors.New("candidate actor, critic, and supervised reference are required")
	}
	if len(rollout.Transitions) == 0 || config.Epochs <= 0 || config.MinibatchSize <= 0 || config.Seed == 0 || config.TargetOldKL <= 0 || !finite64(config.TargetOldKL) {
		return UpdateMetrics{}, errors.New("PPO update needs transitions and valid bounded configuration")
	}
	if config.ClipRange < 0 || config.ClipRange >= 1 {
		return UpdateMetrics{}, errors.New("PPO update clip range must be in [0,1)")
	}
	indices := make([]int, len(rollout.Transitions))
	for index := range indices {
		indices[index] = index
	}
	rng := rand.New(rand.NewSource(config.Seed))
	var actorLossSum, criticLossSum float64
	var updates int
	metrics := UpdateMetrics{}
	for epoch := 0; epoch < config.Epochs; epoch++ {
		rng.Shuffle(len(indices), func(i, j int) { indices[i], indices[j] = indices[j], indices[i] })
		for start := 0; start < len(indices); start += config.MinibatchSize {
			end := min(start+config.MinibatchSize, len(indices))
			selected := indices[start:end]
			actorBatch, criticBatch, err := materializeUpdateBatches(rollout.Transitions, selected, supervisedReference)
			if err != nil {
				return UpdateMetrics{}, err
			}
			actorLoss, err := candidate.TrainStep(actorBatch)
			if err != nil {
				return UpdateMetrics{}, fmt.Errorf("actor minibatch: %w", err)
			}
			criticLoss, err := critic.TrainStep(criticBatch)
			if err != nil {
				return UpdateMetrics{}, fmt.Errorf("critic minibatch: %w", err)
			}
			actorLossSum += actorLoss
			criticLossSum += criticLoss
			updates++
		}
		metrics.EpochsCompleted = epoch + 1
		diagnostics, err := DiagnosePolicy(candidate, supervisedReference, rollout.Transitions, config.MinibatchSize, config.ClipRange)
		if err != nil {
			return UpdateMetrics{}, err
		}
		metrics.Policy = diagnostics
		if diagnostics.ApproxOldPolicyKL > config.TargetOldKL {
			metrics.StoppedEarlyForKL = true
			break
		}
	}
	if updates == 0 {
		return UpdateMetrics{}, errors.New("PPO update performed no minibatches")
	}
	metrics.ActorMinibatchLossMean = actorLossSum / float64(updates)
	metrics.CriticMinibatchLossMean = criticLossSum / float64(updates)
	criticStats, err := criticStatisticsForTransitions(critic, rollout.Transitions, config.MinibatchSize)
	if err != nil {
		return UpdateMetrics{}, err
	}
	metrics.Critic = criticStats
	actorDiagnostics, err := supervised.ReadTrainingDiagnostics(candidate.Store)
	if err != nil {
		return UpdateMetrics{}, fmt.Errorf("actor optimizer diagnostics: %w", err)
	}
	criticDiagnostics, err := supervised.ReadTrainingDiagnostics(critic.Store)
	if err != nil {
		return UpdateMetrics{}, fmt.Errorf("critic optimizer diagnostics: %w", err)
	}
	metrics.ActorGradientNorm = float64(actorDiagnostics.PreclipGlobalGradientNorm)
	metrics.CriticGradientNorm = float64(criticDiagnostics.PreclipGlobalGradientNorm)
	metrics.GradientsFinite = actorDiagnostics.GradientsFinite && criticDiagnostics.GradientsFinite
	if err := ParametersFinite(candidate.Store); err != nil {
		return UpdateMetrics{}, err
	}
	if err := ParametersFinite(critic.Store); err != nil {
		return UpdateMetrics{}, err
	}
	metrics.ParametersFinite = actorDiagnostics.ParametersFinite && criticDiagnostics.ParametersFinite
	metrics.NumericallyStable = metrics.GradientsFinite && metrics.ParametersFinite
	if err := CheckFinite("PPO update metrics", []float64{metrics.ActorMinibatchLossMean, metrics.CriticMinibatchLossMean, metrics.ActorGradientNorm, metrics.CriticGradientNorm, metrics.Critic.ValueLoss, metrics.Critic.ExplainedVariance}); err != nil {
		return UpdateMetrics{}, err
	}
	return metrics, nil
}

func materializeUpdateBatches(transitions []TrajectoryTransition, indices []int, reference *ActorSession) (ActorBatch, CriticBatch, error) {
	inputs := make([]modelstate.Inputs, len(indices))
	actorBatch := ActorBatch{
		Inputs: inputs, Availability: make([][]float32, len(indices)), Actions: make([]int32, len(indices)),
		OldLogProbs: make([]float32, len(indices)), Advantages: make([]float32, len(indices)),
	}
	criticBatch := CriticBatch{Inputs: make([]modelstate.Inputs, len(indices)), Returns: make([]float32, len(indices))}
	for offset, index := range indices {
		if index < 0 || index >= len(transitions) {
			return ActorBatch{}, CriticBatch{}, fmt.Errorf("minibatch transition index %d is out of range", index)
		}
		transition := transitions[index]
		actorBatch.Inputs[offset] = transition.ModelInputs
		actorBatch.Availability[offset] = transition.Availability
		actorBatch.Actions[offset] = transition.ActionID
		actorBatch.OldLogProbs[offset] = float32(transition.OldLogProb)
		actorBatch.Advantages[offset] = float32(transition.Advantage)
		criticBatch.Inputs[offset] = transition.ModelInputs
		criticBatch.Returns[offset] = float32(transition.Return)
	}
	referenceLogits, err := reference.Predict(actorBatch.Inputs)
	if err != nil {
		return ActorBatch{}, CriticBatch{}, err
	}
	actorBatch.ReferenceLogits = referenceLogits
	return actorBatch, criticBatch, nil
}

func criticStatisticsForTransitions(critic *CriticSession, transitions []TrajectoryTransition, batchSize int) (ValueStatistics, error) {
	predictions := make([]float64, len(transitions))
	returns := make([]float64, len(transitions))
	turns := make([]int, len(transitions))
	solved := make([]bool, len(transitions))
	for start := 0; start < len(transitions); start += batchSize {
		end := min(start+batchSize, len(transitions))
		inputs := make([]modelstate.Inputs, end-start)
		for offset, transition := range transitions[start:end] {
			inputs[offset] = transition.ModelInputs
		}
		values, err := critic.Predict(inputs)
		if err != nil {
			return ValueStatistics{}, err
		}
		for offset, value := range values {
			index := start + offset
			predictions[index] = float64(value)
			returns[index] = transitions[index].Return
			turns[index] = transitions[index].Turn
			solved[index] = transitions[index].Solved
		}
	}
	return CalculateValueStatistics(predictions, returns, turns, solved)
}
