package ppo

import (
	"errors"
	"fmt"
	"math/rand"

	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/supervised"
)

// WarmupConfig is the bounded critic-only precondition for actor updates.
type WarmupConfig struct {
	Epochs                   int     `json:"epochs"`
	MinibatchSize            int     `json:"minibatch_size"`
	EvaluationEpisodeModulo  int     `json:"evaluation_episode_modulo"`
	MinimumExplainedVariance float64 `json:"minimum_explained_variance"`
	MinimumImprovement       float64 `json:"minimum_explained_variance_improvement"`
	Seed                     int64   `json:"seed"`
}

// WarmupEpoch records held-out value evidence after each critic-only epoch.
type WarmupEpoch struct {
	Epoch        int             `json:"epoch"`
	TrainingLoss float64         `json:"training_minibatch_loss_mean"`
	Evaluation   ValueStatistics `json:"evaluation"`
}

// WarmupReport determines whether the actor is permitted to unfreeze.
type WarmupReport struct {
	Games                        int             `json:"games"`
	Transitions                  int             `json:"transitions"`
	TrainingSamples              int             `json:"training_samples"`
	EvaluationSamples            int             `json:"evaluation_samples"`
	Initial                      ValueStatistics `json:"initial"`
	Final                        ValueStatistics `json:"final"`
	History                      []WarmupEpoch   `json:"history"`
	ExplainedVarianceImprovement float64         `json:"explained_variance_improvement"`
	GradientNorm                 float64         `json:"gradient_norm"`
	GradientsFinite              bool            `json:"gradients_finite"`
	ParametersFinite             bool            `json:"parameters_finite"`
	NumericallyStable            bool            `json:"numerically_stable"`
	Passed                       bool            `json:"passed"`
	FailureReason                string          `json:"failure_reason,omitempty"`
}

// WarmUpCritic trains only the separate critic on realised complete-game
// returns. Episodes selected by modulo are never used in critic updates and
// provide deterministic held-out warm-up evidence.
func WarmUpCritic(critic *CriticSession, rollout Rollout, config WarmupConfig) (WarmupReport, error) {
	if critic == nil || len(rollout.Transitions) == 0 || config.Epochs <= 0 || config.MinibatchSize <= 0 || config.EvaluationEpisodeModulo < 2 || config.Seed == 0 {
		return WarmupReport{}, errors.New("critic warm-up requires a critic, rollout, and valid bounded configuration")
	}
	if config.MinimumExplainedVariance < 0 || config.MinimumImprovement < 0 || !finite64(config.MinimumExplainedVariance) || !finite64(config.MinimumImprovement) {
		return WarmupReport{}, errors.New("critic warm-up explained-variance gates must be finite and non-negative")
	}
	report := WarmupReport{Games: len(rollout.Episodes), Transitions: len(rollout.Transitions)}
	trainIndices := make([]int, 0, len(rollout.Transitions))
	evaluationTransitions := make([]TrajectoryTransition, 0, len(rollout.Transitions)/config.EvaluationEpisodeModulo)
	for index, transition := range rollout.Transitions {
		if transition.EpisodeID%int64(config.EvaluationEpisodeModulo) == 0 {
			evaluationTransitions = append(evaluationTransitions, transition)
		} else {
			trainIndices = append(trainIndices, index)
		}
	}
	report.TrainingSamples = len(trainIndices)
	report.EvaluationSamples = len(evaluationTransitions)
	if len(trainIndices) == 0 || len(evaluationTransitions) == 0 {
		return WarmupReport{}, errors.New("critic warm-up split produced an empty train or evaluation population")
	}
	initial, err := criticStatisticsForTransitions(critic, evaluationTransitions, config.MinibatchSize)
	if err != nil {
		return WarmupReport{}, err
	}
	report.Initial = initial
	rng := rand.New(rand.NewSource(config.Seed))
	for epoch := 1; epoch <= config.Epochs; epoch++ {
		rng.Shuffle(len(trainIndices), func(i, j int) { trainIndices[i], trainIndices[j] = trainIndices[j], trainIndices[i] })
		var lossSum float64
		var updates int
		for start := 0; start < len(trainIndices); start += config.MinibatchSize {
			end := min(start+config.MinibatchSize, len(trainIndices))
			batch := CriticBatch{
				Inputs:  make([]modelstate.Inputs, end-start),
				Returns: make([]float32, end-start),
			}
			for offset, transitionIndex := range trainIndices[start:end] {
				transition := rollout.Transitions[transitionIndex]
				batch.Inputs[offset] = transition.ModelInputs
				batch.Returns[offset] = float32(transition.Return)
			}
			loss, err := critic.TrainStep(batch)
			if err != nil {
				return WarmupReport{}, fmt.Errorf("critic warm-up epoch %d: %w", epoch, err)
			}
			lossSum += loss
			updates++
		}
		evaluation, err := criticStatisticsForTransitions(critic, evaluationTransitions, config.MinibatchSize)
		if err != nil {
			return WarmupReport{}, err
		}
		report.History = append(report.History, WarmupEpoch{Epoch: epoch, TrainingLoss: lossSum / float64(updates), Evaluation: evaluation})
	}
	report.Final = report.History[len(report.History)-1].Evaluation
	report.ExplainedVarianceImprovement = report.Final.ExplainedVariance - report.Initial.ExplainedVariance
	diagnostics, err := supervised.ReadTrainingDiagnostics(critic.Store)
	if err != nil {
		return WarmupReport{}, err
	}
	report.GradientNorm = float64(diagnostics.PreclipGlobalGradientNorm)
	report.GradientsFinite = diagnostics.GradientsFinite
	report.ParametersFinite = diagnostics.ParametersFinite
	report.NumericallyStable = report.GradientsFinite && report.ParametersFinite
	if err := ParametersFinite(critic.Store); err != nil {
		report.NumericallyStable = false
		report.FailureReason = err.Error()
		return report, nil
	}
	switch {
	case !report.NumericallyStable:
		report.FailureReason = "critic loss, gradients, or parameters were not numerically stable"
	case report.Final.ExplainedVariance < config.MinimumExplainedVariance:
		report.FailureReason = fmt.Sprintf("held-out explained variance %.6f is below required %.6f", report.Final.ExplainedVariance, config.MinimumExplainedVariance)
	case report.ExplainedVarianceImprovement < config.MinimumImprovement:
		report.FailureReason = fmt.Sprintf("held-out explained variance improvement %.6f is below required %.6f", report.ExplainedVarianceImprovement, config.MinimumImprovement)
	default:
		report.Passed = true
	}
	return report, nil
}
