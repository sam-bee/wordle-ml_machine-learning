package experiment

import (
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/sam-bee/wordle-ml_machine-learning/policy"
	rlenv "github.com/sam-bee/wordle-ml_machine-learning/rl"
	rlcheckpoints "github.com/sam-bee/wordle-ml_machine-learning/rl/checkpoints"
	"github.com/sam-bee/wordle-ml_machine-learning/rl/evaluation"
	"github.com/sam-bee/wordle-ml_machine-learning/rl/ppo"
	"github.com/sam-bee/wordle-ml_machine-learning/tensorboard"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

const runStateParameter = "ppo_run_state"

// Options identify the read-only baseline and fresh generated artifact root.
type Options struct {
	ConfigPath           string
	DataDir              string
	SupervisedCheckpoint string
	RunDir               string
}

// CheckpointIdentity pins the exact generation loaded from the immutable
// supervised checkpoint directory.
type CheckpointIdentity struct {
	Directory    string `json:"directory"`
	JSONFile     string `json:"json_file"`
	JSONSHA256   string `json:"json_sha256"`
	BinaryFile   string `json:"binary_file"`
	BinarySHA256 string `json:"binary_sha256"`
	BestUpdate   int    `json:"best_update"`
	SourceRun    string `json:"source_run"`
	SourceCommit string `json:"source_commit"`
}

// CandidateReport is the complete record for one PPO iteration.
type CandidateReport struct {
	Iteration                int                   `json:"iteration"`
	RolloutSeed              int64                 `json:"rollout_seed"`
	PreUpdate                ppo.PolicyDiagnostics `json:"pre_update_policy_diagnostics"`
	Rollout                  ppo.RolloutMetrics    `json:"rollout"`
	Update                   ppo.UpdateMetrics     `json:"update"`
	Greedy                   evaluation.Evaluation `json:"greedy_evaluation"`
	Critic                   ppo.ValueStatistics   `json:"critic_evaluation"`
	AgainstPrevious          evaluation.Comparison `json:"against_previous_accepted"`
	AgainstSupervised        evaluation.Comparison `json:"against_original_supervised"`
	ActorDeltaFromPrevious   float64               `json:"actor_parameter_delta_from_previous"`
	ActorDeltaFromSupervised float64               `json:"actor_parameter_delta_from_supervised"`
	Accepted                 bool                  `json:"accepted"`
	Status                   string                `json:"status"`
	Checkpoint               string                `json:"checkpoint"`
}

// Result is the generated machine-readable experiment report.
type Result struct {
	SchemaVersion              int                   `json:"schema_version"`
	Branch                     string                `json:"branch"`
	BaseCommit                 string                `json:"base_commit"`
	StartedAt                  time.Time             `json:"started_at"`
	CompletedAt                time.Time             `json:"completed_at"`
	Config                     Config                `json:"config"`
	Commands                   []string              `json:"commands"`
	SupervisedBaseline         CheckpointIdentity    `json:"supervised_baseline"`
	Split                      rlenv.Manifest        `json:"rl_split_manifest"`
	CriticArchitecture         string                `json:"critic_architecture"`
	CriticWarmup               ppo.WarmupReport      `json:"critic_warmup"`
	ActorImmutableDuringWarmup bool                  `json:"actor_immutable_during_warmup"`
	BaselineEvaluation         evaluation.Evaluation `json:"baseline_development_evaluation"`
	BaselineSummary            evaluation.Summary    `json:"baseline_development_summary"`
	Iterations                 []CandidateReport     `json:"iterations"`
	AcceptedIteration          int                   `json:"accepted_iteration"`
	BestPPOCheckpoint          string                `json:"best_ppo_checkpoint,omitempty"`
	SecondSeedReplicationRun   bool                  `json:"second_seed_replication_run"`
	Conclusion                 string                `json:"conclusion"`
	ConclusionReason           string                `json:"conclusion_reason"`
	GeneratedRunDir            string                `json:"generated_run_dir"`
	FinalReportPath            string                `json:"final_report_path"`
}

type iterationArtifact struct {
	Iteration         int                    `json:"iteration"`
	Status            string                 `json:"status"`
	Rollout           *ppo.RolloutMetrics    `json:"rollout,omitempty"`
	Update            *ppo.UpdateMetrics     `json:"update,omitempty"`
	Greedy            evaluation.Evaluation  `json:"greedy"`
	AgainstPrevious   *evaluation.Comparison `json:"against_previous,omitempty"`
	AgainstSupervised *evaluation.Comparison `json:"against_supervised,omitempty"`
}

// Run executes one bounded pilot and never reads the final-test vocabulary.
func Run(options Options, output io.Writer) (result Result, runErr error) {
	started := time.Now().UTC()
	config, err := LoadConfig(options.ConfigPath)
	if err != nil {
		return Result{}, err
	}
	if strings.TrimSpace(options.DataDir) == "" || strings.TrimSpace(options.SupervisedCheckpoint) == "" || strings.TrimSpace(options.RunDir) == "" {
		return Result{}, errors.New("data directory, supervised checkpoint, and run directory are required")
	}
	configPath, err := filepath.Abs(options.ConfigPath)
	if err != nil {
		return Result{}, fmt.Errorf("resolve PPO config path: %w", err)
	}
	dataDir, err := filepath.Abs(options.DataDir)
	if err != nil {
		return Result{}, fmt.Errorf("resolve PPO data directory: %w", err)
	}
	supervisedCheckpoint, err := filepath.Abs(options.SupervisedCheckpoint)
	if err != nil {
		return Result{}, fmt.Errorf("resolve supervised checkpoint: %w", err)
	}
	repositoryRoot, err := repositoryRootForConfig(configPath)
	if err != nil {
		return Result{}, err
	}
	if err := verifyGitProvenance(repositoryRoot, config); err != nil {
		return Result{}, err
	}
	runDir, err := validateGeneratedRunDirectory(repositoryRoot, options.RunDir, supervisedCheckpoint)
	if err != nil {
		return Result{}, err
	}
	if err := os.Mkdir(runDir, 0o755); err != nil {
		if errors.Is(err, os.ErrExist) {
			return Result{}, fmt.Errorf("PPO run directory %q already exists", runDir)
		}
		return Result{}, err
	}
	finalReportPath := filepath.Join(runDir, "experiment-result.json")
	result = Result{
		SchemaVersion: 1, Branch: config.Branch, BaseCommit: config.BaseCommit, StartedAt: started, Config: config,
		CriticArchitecture: "separate checkpointed critic: actor candidate-mask fraction, CandidateStats[209] -> Dense(64)+ReLU, Turn Embedding(6,8), action-mask fraction, concatenate -> Dense(64)+ReLU -> scalar; no hidden-answer input",
		AcceptedIteration:  -1, Conclusion: "rejected", GeneratedRunDir: runDir, FinalReportPath: finalReportPath,
		Commands: []string{fmt.Sprintf("go run ./cmd/rl-train --algorithm=ppo --config=%s --data-dir=%s --supervised-checkpoint=%s --run-dir=%s", configPath, dataDir, supervisedCheckpoint, runDir)},
	}
	defer func() {
		result.CompletedAt = time.Now().UTC()
		if writeErr := writeJSONAtomic(finalReportPath, result); writeErr != nil && runErr == nil {
			runErr = fmt.Errorf("write final experiment result: %w", writeErr)
		}
	}()

	vocab, err := vocabulary.LoadWithoutFinalTest(dataDir)
	if err != nil {
		return result, fmt.Errorf("load sealed-test-safe vocabulary: %w", err)
	}
	manifestPath := config.SplitManifest
	if !filepath.IsAbs(manifestPath) {
		manifestPath = filepath.Join(dataDir, manifestPath)
	}
	split, splitManifest, err := rlenv.LoadPPOV1Split(manifestPath, vocab)
	if err != nil {
		return result, err
	}
	result.Split = splitManifest
	baselineIdentity, err := identifyBaseline(supervisedCheckpoint)
	if err != nil {
		return result, err
	}
	result.SupervisedBaseline = baselineIdentity
	if baselineIdentity.SourceRun != "production-20260809-005026Z" || baselineIdentity.BestUpdate != 2200 || baselineIdentity.SourceCommit != "e3ed6ad7e0c58547e1932beb632c8ba750f1523b" {
		return result, fmt.Errorf("supervised checkpoint is %s update %d from %s, want selected production-20260809-005026Z update 2200 from e3ed6ad", baselineIdentity.SourceRun, baselineIdentity.BestUpdate, baselineIdentity.SourceCommit)
	}

	layout, err := rlcheckpoints.Create(filepath.Join(runDir, "checkpoints"), config.RunID)
	if err != nil {
		return result, err
	}
	if err := layout.WriteConfig(config); err != nil {
		return result, err
	}
	metadata := map[string]any{
		"schema_version": 1, "branch": config.Branch, "base_commit": config.BaseCommit,
		"supervised_baseline": baselineIdentity, "rl_split_manifest": splitManifest,
		"validation_untouched": true, "final_test_sealed_and_unopened": true,
	}
	if err := layout.WriteMetadata(metadata); err != nil {
		return result, err
	}
	if err := layout.WriteSupervisedBaselineMetadata(baselineIdentity); err != nil {
		return result, err
	}
	events, err := tensorboard.New(layout.EventsDir)
	if err != nil {
		return result, err
	}
	defer events.Close()

	fmt.Fprintf(output, "PPO run %s: loading immutable supervised actor\n", config.RunID)
	backend, err := compute.NewWithConfig("xla:cuda")
	if err != nil {
		return result, fmt.Errorf("create xla:cuda backend: %w", err)
	}
	defer backend.Finalize()
	actorConfig := ppo.ActorConfig{
		Policy:       policy.Config{NumSolutions: vocabulary.NumSolutions, NumActions: vocabulary.NumActions},
		LearningRate: config.ActorLearningRate, ClipRange: config.ClipRange, EntropyCoefficient: config.EntropyCoefficient,
		SupervisedReferenceKLCoeff: config.SupervisedReferenceKLCoeff, MaximumGradientNorm: config.MaximumGradientNorm, Seed: config.Seed,
	}
	referenceConfig := actorConfig
	referenceConfig.LearningRate = 0
	reference, err := ppo.NewActorSession(referenceConfig, backend)
	if err != nil {
		return result, err
	}
	defer reference.Finalize()
	acceptedActor, err := ppo.NewActorSession(referenceConfig, backend)
	if err != nil {
		return result, err
	}
	defer func() {
		if acceptedActor != nil {
			acceptedActor.Finalize()
		}
	}()
	if err := ppo.LoadBaselineActor(supervisedCheckpoint, reference, acceptedActor); err != nil {
		return result, err
	}
	actorBeforeWarmup, err := ppo.VariableChecksum(acceptedActor.Store, "/wordle_policy/")
	if err != nil {
		return result, err
	}
	criticConfig := ppo.CriticConfig{Policy: actorConfig.Policy, LearningRate: config.CriticLearningRate, ValueLossCoefficient: config.ValueLossCoefficient, MaximumGradientNorm: config.MaximumGradientNorm, Seed: config.Seed + 1}
	acceptedCritic, err := ppo.NewCriticSession(criticConfig, backend)
	if err != nil {
		return result, err
	}
	defer func() {
		if acceptedCritic != nil {
			acceptedCritic.Finalize()
		}
	}()

	fmt.Fprintf(output, "collecting %d stochastic games for critic-only warm-up\n", config.WarmupGames)
	warmupRollout, err := ppo.CollectRollout(vocab, acceptedActor, acceptedCritic, nil, ppo.CollectorConfig{
		AnswerPool: split.Rollout, Games: config.WarmupGames, ParallelGames: config.ParallelGames,
		Seed: config.Seed + 100, Gamma: config.Gamma, GAELambda: config.GAELambda,
	})
	if err != nil {
		return result, fmt.Errorf("critic warm-up rollout: %w", err)
	}
	warmup, err := ppo.WarmUpCritic(acceptedCritic, warmupRollout, ppo.WarmupConfig{
		Epochs: config.WarmupEpochs, MinibatchSize: config.MinibatchSize,
		EvaluationEpisodeModulo:  config.WarmupEvaluationEpisodeModulo,
		MinimumExplainedVariance: config.WarmupMinimumExplainedVariance, MinimumImprovement: config.WarmupMinimumEVImprovement,
		Seed: config.Seed + 200,
	})
	if err != nil {
		return result, fmt.Errorf("critic warm-up: %w", err)
	}
	result.CriticWarmup = warmup
	if err := writeJSONAtomic(filepath.Join(runDir, "critic-warmup.json"), warmup); err != nil {
		return result, err
	}
	actorAfterWarmup, err := ppo.VariableChecksum(acceptedActor.Store, "/wordle_policy/")
	if err != nil {
		return result, err
	}
	result.ActorImmutableDuringWarmup = actorBeforeWarmup == actorAfterWarmup
	warmupRollout = ppo.Rollout{} // trajectories are intentionally discarded.
	if !result.ActorImmutableDuringWarmup {
		result.ConclusionReason = "critic-only warm-up changed the frozen actor"
		return result, errors.New(result.ConclusionReason)
	}
	if !warmup.Passed {
		result.ConclusionReason = "critic warm-up gate failed: " + warmup.FailureReason
		fmt.Fprintln(output, result.ConclusionReason)
		return result, nil
	}
	fmt.Fprintf(output, "critic warm-up passed: held-out explained variance %.4f (initial %.4f)\n", warmup.Final.ExplainedVariance, warmup.Initial.ExplainedVariance)

	fmt.Fprintf(output, "evaluating supervised baseline greedily on all %d PPO-development solutions\n", len(split.Development))
	baselineEvaluation, baselineCritic, err := ppo.EvaluateGreedy(vocab, split.Development, acceptedActor, acceptedActor, reference, acceptedCritic, config.Gamma, config.ClipRange)
	if err != nil {
		return result, err
	}
	baselineSummary, err := evaluation.Summarize(baselineEvaluation)
	if err != nil {
		return result, err
	}
	result.BaselineEvaluation = baselineEvaluation
	result.BaselineSummary = baselineSummary
	iterationZero, err := layout.CreateIteration(0)
	if err != nil {
		return result, err
	}
	initialArtifact := iterationArtifact{Iteration: 0, Status: "initial_accepted_supervised_actor_with_warmed_critic", Greedy: baselineEvaluation}
	if err := saveIteration(iterationZero, acceptedActor, acceptedCritic, config.Seed+100, initialArtifact); err != nil {
		return result, err
	}
	if err := publishBestPayload(layout, iterationZero); err != nil {
		return result, err
	}
	if _, err := layout.Promote(0); err != nil {
		return result, err
	}
	result.AcceptedIteration = 0
	if err := writeBaselineTelemetry(events, baselineEvaluation, baselineSummary, baselineCritic, warmup); err != nil {
		return result, err
	}

	acceptedEvaluation := baselineEvaluation
	acceptedEntry := iterationZero
	pilotPool := append([]string(nil), split.Rollout[:config.PilotSolutions]...)
	compareOptions := evaluation.DefaultCompareOptions(config.Seed + 900)
	compareOptions.BootstrapSamples = config.BootstrapSamples
	compareOptions.MaxOldPolicyKL = config.TargetOldPolicyKL
	compareOptions.MaxSupervisedReferenceKL = config.MaximumSupervisedReferenceKL
	compareOptions.MinimumPolicyEntropy = config.MinimumPolicyEntropy
	acceptedPPO := false
	for iteration := 1; iteration <= config.PilotIterations; iteration++ {
		rolloutSeed := config.Seed + int64(iteration)*1000
		fmt.Fprintf(output, "iteration %d: collecting %d fresh stochastic games\n", iteration, config.RolloutGames)
		rollout, err := ppo.CollectRollout(vocab, acceptedActor, acceptedCritic, reference, ppo.CollectorConfig{
			AnswerPool: pilotPool, Games: config.RolloutGames, Balanced: true, ParallelGames: config.ParallelGames,
			Seed: rolloutSeed, Gamma: config.Gamma, GAELambda: config.GAELambda,
		})
		if err != nil {
			return result, err
		}
		preUpdate, err := ppo.DiagnosePolicy(acceptedActor, reference, rollout.Transitions, config.MinibatchSize, config.ClipRange)
		if err != nil {
			return result, err
		}
		if preUpdate.MaximumAbsoluteRatioFromOne > 1e-5 {
			return result, fmt.Errorf("stored PPO ratios differ from one before update: max delta %.8g", preUpdate.MaximumAbsoluteRatioFromOne)
		}
		candidateActor, err := ppo.NewActorSession(actorConfig, backend)
		if err != nil {
			return result, err
		}
		candidateCritic, err := ppo.NewCriticSession(criticConfig, backend)
		if err != nil {
			candidateActor.Finalize()
			return result, err
		}
		if err := ppo.LoadActorCheckpoint(candidateActor, acceptedEntry.ActorDir); err != nil {
			candidateActor.Finalize()
			candidateCritic.Finalize()
			return result, err
		}
		if err := ppo.LoadCriticCheckpoint(candidateCritic, acceptedEntry.CriticDir); err != nil {
			candidateActor.Finalize()
			candidateCritic.Finalize()
			return result, err
		}
		update, err := ppo.UpdateCandidate(candidateActor, candidateCritic, reference, rollout, ppo.UpdateConfig{
			Epochs: config.PPOEpochs, MinibatchSize: config.MinibatchSize, TargetOldKL: config.TargetOldPolicyKL,
			ClipRange: config.ClipRange, Seed: rolloutSeed + 1,
		})
		if err != nil {
			candidateActor.Finalize()
			candidateCritic.Finalize()
			return result, err
		}
		actorDeltaPrevious, err := ppo.ActorParameterDeltaNorm(candidateActor, acceptedActor)
		if err != nil {
			return result, err
		}
		actorDeltaSupervised, err := ppo.ActorParameterDeltaNorm(candidateActor, reference)
		if err != nil {
			return result, err
		}
		candidateEvaluation, candidateCriticStats, err := ppo.EvaluateGreedy(vocab, split.Development, candidateActor, acceptedActor, reference, candidateCritic, config.Gamma, config.ClipRange)
		if err != nil {
			candidateActor.Finalize()
			candidateCritic.Finalize()
			return result, err
		}
		againstPrevious, err := evaluation.Compare(acceptedEvaluation, candidateEvaluation, compareOptions)
		if err != nil {
			return result, err
		}
		againstSupervised, err := evaluation.Compare(baselineEvaluation, candidateEvaluation, compareOptions)
		if err != nil {
			return result, err
		}
		updateGatesPassed := update.NumericallyStable &&
			update.Policy.ApproxOldPolicyKL <= config.TargetOldPolicyKL &&
			update.Policy.SupervisedReferenceKL <= config.MaximumSupervisedReferenceKL &&
			update.Policy.Entropy >= config.MinimumPolicyEntropy
		accepted := againstPrevious.Acceptance.Accepted && againstSupervised.Acceptance.Accepted && updateGatesPassed
		status := "rejected"
		if accepted {
			status = "accepted"
		}
		entry, err := layout.CreateIteration(iteration)
		if err != nil {
			return result, err
		}
		artifact := iterationArtifact{Iteration: iteration, Status: status, Rollout: &rollout.Metrics, Update: &update, Greedy: candidateEvaluation, AgainstPrevious: &againstPrevious, AgainstSupervised: &againstSupervised}
		if err := saveIteration(entry, candidateActor, candidateCritic, rolloutSeed, artifact); err != nil {
			return result, err
		}
		report := CandidateReport{
			Iteration: iteration, RolloutSeed: rolloutSeed, PreUpdate: preUpdate, Rollout: rollout.Metrics, Update: update,
			Greedy: candidateEvaluation, Critic: candidateCriticStats, AgainstPrevious: againstPrevious, AgainstSupervised: againstSupervised,
			ActorDeltaFromPrevious: actorDeltaPrevious, ActorDeltaFromSupervised: actorDeltaSupervised,
			Accepted: accepted, Status: status, Checkpoint: entry.Dir,
		}
		result.Iterations = append(result.Iterations, report)
		candidateSummary, _ := evaluation.Summarize(candidateEvaluation)
		if err := writeIterationTelemetry(events, int64(iteration), rollout.Metrics, update, candidateEvaluation, candidateSummary, actorDeltaSupervised); err != nil {
			return result, err
		}
		fmt.Fprintf(output, "iteration %d: %s; solved %d/%d, failure-counted mean %.4f, paired diff %+.4f [%+.4f,%+.4f]\n",
			iteration, status, candidateSummary.SolvedCount, candidateSummary.Games, candidateSummary.FailureCountedMeanGuesses,
			againstSupervised.PairedMeanDifference, againstSupervised.PairedBootstrap95.Lower, againstSupervised.PairedBootstrap95.Upper)
		rollout = ppo.Rollout{} // no replay buffer: discard this iteration's trajectories.
		if accepted {
			if err := publishBestPayload(layout, entry); err != nil {
				return result, err
			}
			if _, err := layout.Promote(iteration); err != nil {
				return result, err
			}
			acceptedActor.Finalize()
			acceptedCritic.Finalize()
			acceptedActor = candidateActor
			acceptedCritic = candidateCritic
			acceptedEvaluation = candidateEvaluation
			acceptedEntry = entry
			result.AcceptedIteration = iteration
			result.BestPPOCheckpoint = entry.ActorOnlyDir
			acceptedPPO = true
		} else {
			candidateActor.Finalize()
			candidateCritic.Finalize()
		}
	}
	if acceptedPPO {
		convincing := result.Iterations[len(result.Iterations)-1].AgainstSupervised.Classification == evaluation.ConvincinglyImproved
		if convincing {
			result.Conclusion = "inconclusive"
			result.ConclusionReason = "first seed was convincingly improved, but a second-seed replication is required before acceptance"
		} else {
			result.Conclusion = "inconclusive"
			result.ConclusionReason = "a strict-gate candidate was promoted, but its paired interval did not establish convincing improvement"
		}
	} else {
		result.Conclusion = "rejected"
		result.ConclusionReason = "no PPO candidate passed every strict greedy-development acceptance gate; the supervised baseline remains accepted"
	}
	finalIdentity, err := identifyBaseline(supervisedCheckpoint)
	if err != nil {
		return result, err
	}
	if finalIdentity != baselineIdentity {
		return result, errors.New("immutable supervised checkpoint identity changed during PPO experiment")
	}
	return result, nil
}

func saveIteration(entry rlcheckpoints.IterationLayout, actor *ppo.ActorSession, critic *ppo.CriticSession, rolloutSeed int64, evaluationArtifact any) error {
	actorChecksum, err := ppo.VariableChecksum(actor.Store, "/wordle_policy/")
	if err != nil {
		return err
	}
	criticChecksum, err := ppo.VariableChecksum(critic.Store, "/ppo_critic/")
	if err != nil {
		return err
	}
	state := rlcheckpoints.IterationState{
		SchemaVersion: rlcheckpoints.SchemaVersion, Iteration: entry.Iteration, RolloutSeed: rolloutSeed,
		ActorSteps: actor.Trainer.GlobalStep(), CriticSteps: critic.Trainer.GlobalStep(), ActorChecksum: actorChecksum, CriticChecksum: criticChecksum,
	}
	stateJSON, _ := json.Marshal(state)
	actor.Store.SetParam(runStateParameter, string(stateJSON))
	critic.Store.SetParam(runStateParameter, string(stateJSON))
	if err := ppo.SaveStoreCheckpoint(actor.Store, entry.ActorDir); err != nil {
		return fmt.Errorf("save actor checkpoint: %w", err)
	}
	if err := ppo.SaveStoreCheckpoint(critic.Store, entry.CriticDir); err != nil {
		return fmt.Errorf("save critic checkpoint: %w", err)
	}
	if err := ppo.ExportActorOnly(actor, entry.ActorOnlyDir); err != nil {
		return fmt.Errorf("export actor-only checkpoint: %w", err)
	}
	if err := entry.WriteState(state); err != nil {
		return err
	}
	return entry.WriteEvaluation(evaluationArtifact)
}

func publishBestPayload(layout rlcheckpoints.Layout, entry rlcheckpoints.IterationLayout) error {
	actorCriticTarget := filepath.Join(layout.BestDir, "actor-critic")
	actorOnlyTarget := filepath.Join(layout.BestDir, "actor-only")
	if err := replaceDirectoryCopy(entry.ActorCriticDir, actorCriticTarget); err != nil {
		return err
	}
	return replaceDirectoryCopy(entry.ActorOnlyDir, actorOnlyTarget)
}

func replaceDirectoryCopy(source, target string) error {
	parent := filepath.Dir(target)
	temporary, err := os.MkdirTemp(parent, ".best-payload-")
	if err != nil {
		return err
	}
	installed := false
	defer func() {
		if !installed {
			_ = os.RemoveAll(temporary)
		}
	}()
	if err := copyDirectoryContents(source, temporary); err != nil {
		return err
	}
	backup := target + ".previous"
	_ = os.RemoveAll(backup)
	if _, err := os.Stat(target); err == nil {
		if err := os.Rename(target, backup); err != nil {
			return err
		}
	}
	if err := os.Rename(temporary, target); err != nil {
		_ = os.Rename(backup, target)
		return err
	}
	installed = true
	_ = os.RemoveAll(backup)
	return nil
}

func copyDirectoryContents(source, target string) error {
	entries, err := os.ReadDir(source)
	if err != nil {
		return err
	}
	for _, entry := range entries {
		sourcePath := filepath.Join(source, entry.Name())
		targetPath := filepath.Join(target, entry.Name())
		if entry.IsDir() {
			if err := os.Mkdir(targetPath, 0o755); err != nil {
				return err
			}
			if err := copyDirectoryContents(sourcePath, targetPath); err != nil {
				return err
			}
			continue
		}
		if entry.Type()&os.ModeSymlink != 0 {
			return fmt.Errorf("refuse checkpoint symlink %s", sourcePath)
		}
		contents, err := os.ReadFile(sourcePath)
		if err != nil {
			return err
		}
		if err := os.WriteFile(targetPath, contents, 0o644); err != nil {
			return err
		}
	}
	return nil
}

func identifyBaseline(dir string) (CheckpointIdentity, error) {
	absolute, err := filepath.Abs(dir)
	if err != nil {
		return CheckpointIdentity{}, err
	}
	entries, err := os.ReadDir(absolute)
	if err != nil {
		return CheckpointIdentity{}, fmt.Errorf("read supervised checkpoint directory: %w", err)
	}
	jsonFiles := make([]string, 0)
	for _, entry := range entries {
		if !entry.IsDir() && strings.HasPrefix(entry.Name(), "checkpoint-") && strings.HasSuffix(entry.Name(), ".json") {
			jsonFiles = append(jsonFiles, entry.Name())
		}
	}
	if len(jsonFiles) == 0 {
		return CheckpointIdentity{}, errors.New("supervised checkpoint directory contains no checkpoint JSON")
	}
	sort.Strings(jsonFiles)
	jsonName := jsonFiles[len(jsonFiles)-1]
	binName := strings.TrimSuffix(jsonName, ".json") + ".bin"
	jsonHash, err := hashFile(filepath.Join(absolute, jsonName))
	if err != nil {
		return CheckpointIdentity{}, err
	}
	binHash, err := hashFile(filepath.Join(absolute, binName))
	if err != nil {
		return CheckpointIdentity{}, err
	}
	updatePattern := regexp.MustCompile(`step-0*([0-9]+)\.json$`)
	match := updatePattern.FindStringSubmatch(jsonName)
	if len(match) != 2 {
		return CheckpointIdentity{}, fmt.Errorf("selected supervised checkpoint %q does not identify a training step", jsonName)
	}
	bestUpdate, err := strconv.Atoi(match[1])
	if err != nil {
		return CheckpointIdentity{}, err
	}
	runDir := filepath.Clean(filepath.Join(absolute, "..", ".."))
	metadataContents, err := os.ReadFile(filepath.Join(runDir, "metadata.json"))
	if err != nil {
		return CheckpointIdentity{}, fmt.Errorf("read supervised run metadata: %w", err)
	}
	var metadata struct {
		Repositories struct {
			MachineLearning struct {
				Commit string `json:"commit"`
			} `json:"machine_learning"`
		} `json:"repositories"`
	}
	if err := json.Unmarshal(metadataContents, &metadata); err != nil {
		return CheckpointIdentity{}, fmt.Errorf("decode supervised run metadata: %w", err)
	}
	if metadata.Repositories.MachineLearning.Commit == "" {
		return CheckpointIdentity{}, errors.New("supervised run metadata has no machine-learning commit")
	}
	return CheckpointIdentity{
		Directory: absolute, JSONFile: jsonName, JSONSHA256: jsonHash, BinaryFile: binName, BinarySHA256: binHash,
		BestUpdate: bestUpdate, SourceRun: filepath.Base(runDir), SourceCommit: metadata.Repositories.MachineLearning.Commit,
	}, nil
}

func pathsOverlap(left, right string) bool {
	left = filepath.Clean(left)
	right = filepath.Clean(right)
	return pathWithin(left, right) || pathWithin(right, left)
}

func pathWithin(root, path string) bool {
	relative, err := filepath.Rel(root, path)
	return err == nil && relative != ".." && !strings.HasPrefix(relative, ".."+string(filepath.Separator)) && !filepath.IsAbs(relative)
}

func hashFile(path string) (string, error) {
	contents, err := os.ReadFile(path)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(contents)
	return fmt.Sprintf("%x", digest[:]), nil
}

func writeJSONAtomic(path string, value any) error {
	contents, err := json.MarshalIndent(value, "", "  ")
	if err != nil {
		return err
	}
	contents = append(contents, '\n')
	temporary, err := os.CreateTemp(filepath.Dir(path), ".json-")
	if err != nil {
		return err
	}
	temporaryPath := temporary.Name()
	defer os.Remove(temporaryPath)
	if _, err := temporary.Write(contents); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Sync(); err != nil {
		_ = temporary.Close()
		return err
	}
	if err := temporary.Close(); err != nil {
		return err
	}
	return os.Rename(temporaryPath, path)
}
