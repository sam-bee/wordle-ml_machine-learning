package experiment

import (
	"github.com/sam-bee/wordle-ml_machine-learning/rl/evaluation"
	"github.com/sam-bee/wordle-ml_machine-learning/rl/ppo"
	"github.com/sam-bee/wordle-ml_machine-learning/tensorboard"
)

func writeBaselineTelemetry(writer *tensorboard.Writer, gameEvaluation evaluation.Evaluation, summary evaluation.Summary, critic ppo.ValueStatistics, warmup ppo.WarmupReport) error {
	openingAction := 0
	if len(gameEvaluation.Games) > 0 && len(gameEvaluation.Games[0].Turns) > 0 {
		openingAction = gameEvaluation.Games[0].Turns[0].ActionID
	}
	return writer.WriteScalars(0,
		tensorboard.Scalar{Tag: "ppo/episode_return_mean", Value: 0},
		tensorboard.Scalar{Tag: "ppo/solve_rate", Value: float32(summary.SolveRate)},
		tensorboard.Scalar{Tag: "ppo/mean_guesses_solved", Value: float32(summary.MeanGuessesSolved)},
		tensorboard.Scalar{Tag: "ppo/failure_counted_mean_guesses", Value: float32(summary.FailureCountedMeanGuesses)},
		tensorboard.Scalar{Tag: "ppo/policy_loss", Value: 0},
		tensorboard.Scalar{Tag: "ppo/value_loss", Value: float32(critic.ValueLoss)},
		tensorboard.Scalar{Tag: "ppo/entropy", Value: float32(summary.Diagnostics.PolicyEntropy)},
		tensorboard.Scalar{Tag: "ppo/explained_variance", Value: float32(critic.ExplainedVariance)},
		tensorboard.Scalar{Tag: "ppo/approx_old_policy_kl", Value: 0},
		tensorboard.Scalar{Tag: "ppo/supervised_reference_kl", Value: 0},
		tensorboard.Scalar{Tag: "ppo/clip_fraction", Value: 0},
		tensorboard.Scalar{Tag: "ppo/advantage_mean", Value: 0},
		tensorboard.Scalar{Tag: "ppo/advantage_std", Value: 0},
		tensorboard.Scalar{Tag: "ppo/return_mean", Value: 0},
		tensorboard.Scalar{Tag: "ppo/step_reward_mean", Value: 0},
		tensorboard.Scalar{Tag: "ppo/gradient_norm", Value: float32(warmup.GradientNorm)},
		tensorboard.Scalar{Tag: "ppo/rollout_games", Value: float32(warmup.Games)},
		tensorboard.Scalar{Tag: "ppo/rollout_steps", Value: float32(warmup.Transitions)},
		tensorboard.Scalar{Tag: "ppo/illegal_or_repeated_action_count", Value: 0},
		tensorboard.Scalar{Tag: "ppo/actor_parameter_delta_norm", Value: 0},
		tensorboard.Scalar{Tag: "ppo/opening_action", Value: float32(openingAction)},
		tensorboard.Scalar{Tag: "ppo/opening_action_probability", Value: float32(summary.OpeningActionProbability)},
	)
}

func writeIterationTelemetry(writer *tensorboard.Writer, step int64, rollout ppo.RolloutMetrics, update ppo.UpdateMetrics, gameEvaluation evaluation.Evaluation, summary evaluation.Summary, actorDelta float64) error {
	openingAction := 0
	if len(gameEvaluation.Games) > 0 && len(gameEvaluation.Games[0].Turns) > 0 {
		openingAction = gameEvaluation.Games[0].Turns[0].ActionID
	}
	return writer.WriteScalars(step,
		tensorboard.Scalar{Tag: "ppo/episode_return_mean", Value: float32(rollout.EpisodeReturnMean)},
		tensorboard.Scalar{Tag: "ppo/solve_rate", Value: float32(summary.SolveRate)},
		tensorboard.Scalar{Tag: "ppo/mean_guesses_solved", Value: float32(summary.MeanGuessesSolved)},
		tensorboard.Scalar{Tag: "ppo/failure_counted_mean_guesses", Value: float32(summary.FailureCountedMeanGuesses)},
		tensorboard.Scalar{Tag: "ppo/policy_loss", Value: float32(update.Policy.PolicyLoss)},
		tensorboard.Scalar{Tag: "ppo/value_loss", Value: float32(update.Critic.ValueLoss)},
		tensorboard.Scalar{Tag: "ppo/entropy", Value: float32(update.Policy.Entropy)},
		tensorboard.Scalar{Tag: "ppo/explained_variance", Value: float32(update.Critic.ExplainedVariance)},
		tensorboard.Scalar{Tag: "ppo/approx_old_policy_kl", Value: float32(update.Policy.ApproxOldPolicyKL)},
		tensorboard.Scalar{Tag: "ppo/supervised_reference_kl", Value: float32(update.Policy.SupervisedReferenceKL)},
		tensorboard.Scalar{Tag: "ppo/clip_fraction", Value: float32(update.Policy.ClipFraction)},
		tensorboard.Scalar{Tag: "ppo/advantage_mean", Value: float32(rollout.AdvantageMean)},
		tensorboard.Scalar{Tag: "ppo/advantage_std", Value: float32(rollout.AdvantageStd)},
		tensorboard.Scalar{Tag: "ppo/return_mean", Value: float32(rollout.ReturnMean)},
		tensorboard.Scalar{Tag: "ppo/step_reward_mean", Value: float32(rollout.StepRewardMean)},
		tensorboard.Scalar{Tag: "ppo/gradient_norm", Value: float32(update.ActorGradientNorm)},
		tensorboard.Scalar{Tag: "ppo/rollout_games", Value: float32(rollout.RolloutGames)},
		tensorboard.Scalar{Tag: "ppo/rollout_steps", Value: float32(rollout.RolloutSteps)},
		tensorboard.Scalar{Tag: "ppo/illegal_or_repeated_action_count", Value: float32(rollout.IllegalOrRepeatedActionCount)},
		tensorboard.Scalar{Tag: "ppo/actor_parameter_delta_norm", Value: float32(actorDelta)},
		tensorboard.Scalar{Tag: "ppo/opening_action", Value: float32(openingAction)},
		tensorboard.Scalar{Tag: "ppo/opening_action_probability", Value: float32(summary.OpeningActionProbability)},
	)
}
