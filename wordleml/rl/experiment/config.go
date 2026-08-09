// Package experiment orchestrates the single bounded PPO option. It is not a
// generic reinforcement-learning framework: Wordle environment and PPO logic
// stay explicit and inspectable.
package experiment

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
)

// Config is the complete reproducible configuration for the first PPO pilot.
type Config struct {
	SchemaVersion int    `json:"schema_version"`
	Algorithm     string `json:"algorithm"`
	RunID         string `json:"run_id"`
	Seed          int64  `json:"seed"`

	Gamma                        float64 `json:"gamma"`
	GAELambda                    float64 `json:"gae_lambda"`
	ClipRange                    float64 `json:"clip_range"`
	PPOEpochs                    int     `json:"ppo_epochs_per_rollout"`
	MinibatchSize                int     `json:"minibatch_size"`
	ActorLearningRate            float64 `json:"actor_learning_rate"`
	CriticLearningRate           float64 `json:"critic_learning_rate"`
	ValueLossCoefficient         float64 `json:"value_loss_coefficient"`
	EntropyCoefficient           float64 `json:"entropy_coefficient"`
	MaximumGradientNorm          float64 `json:"maximum_gradient_norm"`
	TargetOldPolicyKL            float64 `json:"target_old_policy_kl"`
	SupervisedReferenceKLCoeff   float64 `json:"supervised_reference_kl_coefficient"`
	MaximumSupervisedReferenceKL float64 `json:"maximum_supervised_reference_kl"`
	MinimumPolicyEntropy         float64 `json:"minimum_policy_entropy"`

	WarmupGames                    int     `json:"critic_warmup_games"`
	WarmupEpochs                   int     `json:"critic_warmup_epochs"`
	WarmupEvaluationEpisodeModulo  int     `json:"critic_warmup_evaluation_episode_modulo"`
	WarmupMinimumExplainedVariance float64 `json:"critic_warmup_minimum_explained_variance"`
	WarmupMinimumEVImprovement     float64 `json:"critic_warmup_minimum_explained_variance_improvement"`

	RolloutGames          int `json:"rollout_games"`
	ParallelGames         int `json:"parallel_games"`
	PilotSolutions        int `json:"pilot_solutions"`
	GamesPerPilotSolution int `json:"games_per_pilot_solution"`
	PilotIterations       int `json:"pilot_iterations"`
	BootstrapSamples      int `json:"paired_bootstrap_samples"`

	SplitManifest string `json:"split_manifest"`
	BaseCommit    string `json:"base_commit"`
	Branch        string `json:"branch"`
}

// LoadConfig decodes one strict JSON configuration.
func LoadConfig(path string) (Config, error) {
	file, err := os.Open(path)
	if err != nil {
		return Config{}, fmt.Errorf("open PPO config: %w", err)
	}
	defer file.Close()
	decoder := json.NewDecoder(file)
	decoder.DisallowUnknownFields()
	var config Config
	if err := decoder.Decode(&config); err != nil {
		return Config{}, fmt.Errorf("decode PPO config: %w", err)
	}
	if err := ensureEOF(decoder); err != nil {
		return Config{}, err
	}
	if err := config.Validate(); err != nil {
		return Config{}, err
	}
	return config, nil
}

func ensureEOF(decoder *json.Decoder) error {
	var extra any
	if err := decoder.Decode(&extra); err == io.EOF {
		return nil
	} else if err != nil {
		return fmt.Errorf("decode trailing PPO config data: %w", err)
	}
	return errors.New("PPO config contains more than one JSON value")
}

// Validate enforces the deliberately bounded initial experiment.
func (config Config) Validate() error {
	if config.SchemaVersion != 1 || config.Algorithm != "ppo" || config.RunID == "" || config.Seed == 0 {
		return errors.New("PPO config needs schema version 1, algorithm ppo, run ID, and non-zero seed")
	}
	for name, value := range map[string]float64{
		"gamma": config.Gamma, "GAE lambda": config.GAELambda, "clip range": config.ClipRange,
		"actor learning rate": config.ActorLearningRate, "critic learning rate": config.CriticLearningRate,
		"value coefficient": config.ValueLossCoefficient, "entropy coefficient": config.EntropyCoefficient,
		"maximum gradient norm": config.MaximumGradientNorm, "target old-policy KL": config.TargetOldPolicyKL,
		"supervised-reference KL coefficient": config.SupervisedReferenceKLCoeff,
		"maximum supervised-reference KL":     config.MaximumSupervisedReferenceKL,
		"minimum policy entropy":              config.MinimumPolicyEntropy,
		"warm-up minimum explained variance":  config.WarmupMinimumExplainedVariance,
		"warm-up minimum EV improvement":      config.WarmupMinimumEVImprovement,
	} {
		if math.IsNaN(value) || math.IsInf(value, 0) || value < 0 {
			return fmt.Errorf("%s must be finite and non-negative, got %v", name, value)
		}
	}
	if config.Gamma > 1 || config.GAELambda > 1 || config.ClipRange >= 1 || config.ActorLearningRate == 0 || config.CriticLearningRate == 0 || config.ValueLossCoefficient == 0 || config.MaximumGradientNorm == 0 || config.TargetOldPolicyKL == 0 {
		return errors.New("PPO discounts/clip or positive optimiser settings are outside their valid ranges")
	}
	if config.PPOEpochs < 1 || config.PPOEpochs > 4 || config.MinibatchSize <= 0 || config.RolloutGames < 1024 || config.RolloutGames > 2048 || config.WarmupGames < 2000 || config.WarmupEpochs <= 0 || config.WarmupEvaluationEpisodeModulo < 2 || config.ParallelGames <= 0 {
		return errors.New("PPO rollout/warm-up/minibatch bounds do not describe the conservative pilot")
	}
	if config.PilotSolutions < 256 || config.PilotSolutions > 512 || config.GamesPerPilotSolution < 2 || config.RolloutGames != config.PilotSolutions*config.GamesPerPilotSolution || config.PilotIterations < 1 || config.PilotIterations > 3 || config.BootstrapSamples <= 0 {
		return errors.New("PPO pilot must use 256..512 solutions, at least two games each, 1..3 iterations, and matching rollout size")
	}
	if config.SplitManifest == "" || config.BaseCommit == "" || config.Branch != "experiment/ppo-rl" {
		return errors.New("PPO split manifest, base commit, and experiment/ppo-rl branch identity are required")
	}
	return nil
}
