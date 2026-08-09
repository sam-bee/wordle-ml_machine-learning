package productionreport

import (
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"strings"

	"github.com/sam-bee/wordle-ml_machine-learning/proofrun"
)

const pairedTop1ArtifactName = "best-paired-top1.json"

// SeedReplicationOptions identifies the original production run, its one
// predeclared seed replication, and the report path owned by this experiment.
type SeedReplicationOptions struct {
	RunsDir          string
	OriginalRunID    string
	ReplicationRunID string
	OutputPath       string
}

// SeedReplicationReport is the complete validation-only robustness comparison
// for the two independently initialized, otherwise fixed runs.
type SeedReplicationReport struct {
	Original            SeedCheckpoint       `json:"original"`
	Replication         SeedCheckpoint       `json:"replication"`
	Delta               Delta                `json:"delta_replication_minus_original"`
	Failures            FailureSetComparison `json:"failure_sets"`
	PairedTop1          pairedTop1           `json:"paired_validation_teacher_top1"`
	Games               []PairedGame         `json:"paired_games"`
	LowerValidationLoss string               `json:"lower_validation_loss_run_id"`
}

// SeedCheckpoint adds the fixed seed, final loss, and named failures to the
// independently verified best-checkpoint summary.
type SeedCheckpoint struct {
	Checkpoint
	Seed                int64    `json:"seed"`
	FinalValidationLoss float64  `json:"final_validation_loss"`
	FailedSolutions     []string `json:"failed_solutions"`
}

// FailureSetComparison makes overlap and seed-specific failures explicit.
type FailureSetComparison struct {
	Both            []string `json:"both"`
	OriginalOnly    []string `json:"original_only"`
	ReplicationOnly []string `json:"replication_only"`
}

// PairedGame compares the same validation solution under both best
// checkpoints. GuessDelta is replication minus original.
type PairedGame struct {
	Solution           string `json:"solution"`
	OriginalSolved     bool   `json:"original_solved"`
	ReplicationSolved  bool   `json:"replication_solved"`
	OriginalGuesses    int    `json:"original_guesses"`
	ReplicationGuesses int    `json:"replication_guesses"`
	GuessDelta         int    `json:"guess_delta_replication_minus_original"`
}

type pairedTop1 struct {
	OriginalRunID         string `json:"original_run_id"`
	ReplicationRunID      string `json:"replication_run_id"`
	OriginalSeed          int64  `json:"original_seed"`
	ReplicationSeed       int64  `json:"replication_seed"`
	OriginalBestUpdate    int64  `json:"original_best_update"`
	ReplicationBestUpdate int64  `json:"replication_best_update"`
	ValidationSplitHash   string `json:"validation_split_hash"`
	Examples              int    `json:"examples"`
	BothTeacherTop1       int    `json:"both_teacher_top1"`
	OriginalOnly          int    `json:"original_only_teacher_top1"`
	ReplicationOnly       int    `json:"replication_only_teacher_top1"`
	NeitherTeacherTop1    int    `json:"neither_teacher_top1"`
	OriginalSelectionHash string `json:"original_teacher_top1_vector_sha256"`
	ReplicationHash       string `json:"replication_teacher_top1_vector_sha256"`
}

// ValidateSeedReplication verifies both complete artifact sets, their paired
// validation-state artifact, and all paired validation-game trajectories.
func ValidateSeedReplication(options SeedReplicationOptions) (SeedReplicationReport, error) {
	if strings.TrimSpace(options.RunsDir) == "" || strings.TrimSpace(options.OriginalRunID) == "" || strings.TrimSpace(options.ReplicationRunID) == "" {
		return SeedReplicationReport{}, errors.New("runs directory, original run ID, and replication run ID are required")
	}
	if options.OriginalRunID == options.ReplicationRunID {
		return SeedReplicationReport{}, errors.New("original and replication run IDs must differ")
	}
	original, err := loadRun(options.RunsDir, options.OriginalRunID, string(proofrun.Production), productionUpdates)
	if err != nil {
		return SeedReplicationReport{}, fmt.Errorf("original production run %q: %w", options.OriginalRunID, err)
	}
	replication, err := loadRun(options.RunsDir, options.ReplicationRunID, string(proofrun.SeedReplication), productionUpdates)
	if err != nil {
		return SeedReplicationReport{}, fmt.Errorf("seed replication run %q: %w", options.ReplicationRunID, err)
	}
	normalized := replication.config
	normalized.Stage, normalized.Seed = original.config.Stage, original.config.Seed
	if normalized != original.config {
		return SeedReplicationReport{}, errors.New("replication configuration differs from production beyond stage identity and seed")
	}
	if !sameSeedReplicationInputs(original, replication) {
		return SeedReplicationReport{}, errors.New("original and replication runs do not record identical model and frozen train/validation inputs")
	}
	if !replication.metadata.FinalTestSealed || len(replication.metadata.Splits.Test) != 0 {
		return SeedReplicationReport{}, errors.New("replication metadata does not prove the final-test wordlist remained unopened")
	}
	if original.evaluation.ValidationSplitHash == "" || original.evaluation.ValidationSplitHash != replication.evaluation.ValidationSplitHash {
		return SeedReplicationReport{}, errors.New("original and replication evaluations have different validation split hashes")
	}
	paired, err := loadPairedTop1(replication, original)
	if err != nil {
		return SeedReplicationReport{}, err
	}
	games, err := pairGames(original.evaluation.Games.Games, replication.evaluation.Games.Games)
	if err != nil {
		return SeedReplicationReport{}, err
	}
	originalCheckpoint := seedCheckpoint(original)
	replicationCheckpoint := seedCheckpoint(replication)
	lower := originalCheckpoint.RunID
	if replicationCheckpoint.Loss < originalCheckpoint.Loss {
		lower = replicationCheckpoint.RunID
	} else if replicationCheckpoint.Loss == originalCheckpoint.Loss {
		lower = "tie"
	}
	return SeedReplicationReport{
		Original: originalCheckpoint, Replication: replicationCheckpoint,
		Delta:      difference(replicationCheckpoint.Checkpoint, originalCheckpoint.Checkpoint),
		Failures:   failureSets(originalCheckpoint.FailedSolutions, replicationCheckpoint.FailedSolutions),
		PairedTop1: paired, Games: games, LowerValidationLoss: lower,
	}, nil
}

// WriteSeedReplication validates first and atomically publishes a standalone
// report. It explicitly refuses the existing production report's filename.
func WriteSeedReplication(options SeedReplicationOptions) (SeedReplicationReport, error) {
	if strings.TrimSpace(options.OutputPath) == "" {
		return SeedReplicationReport{}, errors.New("output path is required")
	}
	if filepath.Base(options.OutputPath) == "production-training-report.md" {
		return SeedReplicationReport{}, errors.New("seed replication must not overwrite the production training report")
	}
	report, err := ValidateSeedReplication(options)
	if err != nil {
		return SeedReplicationReport{}, err
	}
	if err := writeAtomically(options.OutputPath, []byte(report.Markdown())); err != nil {
		return SeedReplicationReport{}, err
	}
	return report, nil
}

// Markdown renders the two-seed validation comparison and all 100 paired game
// outcomes without presenting the experiment as a statistical study.
func (report SeedReplicationReport) Markdown() string {
	var b strings.Builder
	b.WriteString("# Independent-seed production replication\n\n")
	b.WriteString("<!-- seedreplicationreport: complete -->\n\n")
	b.WriteString("This is one predeclared independent-initialization robustness check, not model tuning and not a comprehensive statistical study. The two runs use identical model, data, objective, optimiser, sampling, cadence, and 10,000-update configuration; only the recorded seed and run identity differ.\n\n")
	b.WriteString("| Run | Seed | Final update / best update | Final validation loss | Best validation loss / top-1 / top-5 / top-16 | Games solved / mean guesses | Guess-count distribution (1..6) | Failed solutions |\n")
	b.WriteString("| --- | ---: | ---: | ---: | --- | --- | --- | --- |\n")
	writeSeedCheckpointRow(&b, "original", report.Original)
	writeSeedCheckpointRow(&b, "replication", report.Replication)
	fmt.Fprintf(&b, "| replication − original | — | — | %+.4f | %+.4f / %+.3f / %+.3f / %+.3f | %+d / %+.3f | [%+d, %+d, %+d, %+d, %+d, %+d] | — |\n",
		report.Replication.FinalValidationLoss-report.Original.FinalValidationLoss,
		report.Delta.Loss, report.Delta.Top1, report.Delta.Top5, report.Delta.Top16,
		report.Delta.Solved, report.Delta.MeanGuesses,
		report.Delta.GuessCounts[0], report.Delta.GuessCounts[1], report.Delta.GuessCounts[2], report.Delta.GuessCounts[3], report.Delta.GuessCounts[4], report.Delta.GuessCounts[5])
	fmt.Fprintf(&b, "\nUnder the predeclared lowest-best-validation-loss rule, `%s` has the lower validation loss. This statement does not change selected-model documentation.\n", report.LowerValidationLoss)
	b.WriteString("\n## Failure sets\n\n")
	fmt.Fprintf(&b, "- Failed under both: %s\n", formatWords(report.Failures.Both))
	fmt.Fprintf(&b, "- Original only: %s\n", formatWords(report.Failures.OriginalOnly))
	fmt.Fprintf(&b, "- Replication only: %s\n", formatWords(report.Failures.ReplicationOnly))
	b.WriteString("\n## Paired validation states\n\n")
	b.WriteString("| Both selected teacher top-1 | Original only | Replication only | Neither | Total |\n")
	b.WriteString("| ---: | ---: | ---: | ---: | ---: |\n")
	fmt.Fprintf(&b, "| %d | %d | %d | %d | %d |\n", report.PairedTop1.BothTeacherTop1, report.PairedTop1.OriginalOnly, report.PairedTop1.ReplicationOnly, report.PairedTop1.NeitherTeacherTop1, report.PairedTop1.Examples)
	b.WriteString("\n## Paired validation games\n\n")
	b.WriteString("Guess delta is replication minus original. Failed games still record all six accepted guesses.\n\n")
	b.WriteString("| Solution | Original | Replication | Original guesses | Replication guesses | Guess delta |\n")
	b.WriteString("| --- | --- | --- | ---: | ---: | ---: |\n")
	for _, game := range report.Games {
		fmt.Fprintf(&b, "| %s | %s | %s | %d | %d | %+d |\n", game.Solution, solvedLabel(game.OriginalSolved), solvedLabel(game.ReplicationSolved), game.OriginalGuesses, game.ReplicationGuesses, game.GuessDelta)
	}
	b.WriteString("\n## Verified artifacts\n\n")
	writeSeedArtifacts(&b, "Original", report.Original)
	writeSeedArtifacts(&b, "Replication", report.Replication)
	fmt.Fprintf(&b, "- Paired teacher-top-1 artifact: `%s`\n", filepath.Join(report.Replication.RunDir, "evaluations", pairedTop1ArtifactName))
	fmt.Fprintf(&b, "- Validation split hash: `%s`; independent best-metric reproduction tolerance: `%g`\n", report.Replication.ValidationSplitHash, report.Replication.ReproductionTolerance)
	b.WriteString("\nThe sealed final-test split remained unopened: neither training, checkpoint selection, paired state comparison, gameplay evaluation, nor this report loads or evaluates final-test examples. All measurements above are from the fixed validation split.\n")
	return b.String()
}

func writeSeedCheckpointRow(b *strings.Builder, label string, checkpoint SeedCheckpoint) {
	fmt.Fprintf(b, "| %s `%s` | %d | %d / %d | %.4f | %.4f / %.3f / %.3f / %.3f | %d/%d / %.3f | [%d, %d, %d, %d, %d, %d] | %s |\n",
		label, checkpoint.RunID, checkpoint.Seed, checkpoint.Updates, checkpoint.BestUpdate, checkpoint.FinalValidationLoss,
		checkpoint.Loss, checkpoint.Top1, checkpoint.Top5, checkpoint.Top16, checkpoint.Solved, checkpoint.Games, checkpoint.MeanGuesses,
		checkpoint.GuessCounts[0], checkpoint.GuessCounts[1], checkpoint.GuessCounts[2], checkpoint.GuessCounts[3], checkpoint.GuessCounts[4], checkpoint.GuessCounts[5], formatWords(checkpoint.FailedSolutions))
}

func writeSeedArtifacts(b *strings.Builder, label string, checkpoint SeedCheckpoint) {
	fmt.Fprintf(b, "- %s `%s` (seed %d; TensorBoard: `tensorboard --logdir %s`)\n", label, checkpoint.RunDir, checkpoint.Seed, filepath.Join(checkpoint.RunDir, "events"))
	fmt.Fprintf(b, "  - best checkpoint: `%s`; independent evaluation and trajectories: `%s` and `%s`\n", filepath.Join(checkpoint.RunDir, "checkpoints", "best"), filepath.Join(checkpoint.RunDir, "evaluations", "best-games100.json"), filepath.Join(checkpoint.RunDir, "evaluations", "best-games100.jsonl"))
	fmt.Fprintf(b, "  - commits: machine-learning `%s`; synthetic-data `%s`; game-engine `%s`\n", checkpoint.MachineLearningCommit, checkpoint.SyntheticDataCommit, checkpoint.GameEngineCommit)
}

func seedCheckpoint(run loadedRun) SeedCheckpoint {
	return SeedCheckpoint{
		Checkpoint: run.checkpoint(), Seed: run.config.Seed, FinalValidationLoss: run.final.FinalValidation.Loss,
		FailedSolutions: append([]string(nil), run.evaluation.Games.Summary.FailedSolutions...),
	}
}

func sameSeedReplicationInputs(original, replication loadedRun) bool {
	return original.metadata.ModelParameterCount == replication.metadata.ModelParameterCount &&
		reflect.DeepEqual(original.metadata.Dataset, replication.metadata.Dataset) &&
		reflect.DeepEqual(original.metadata.Vocabulary, replication.metadata.Vocabulary) &&
		reflect.DeepEqual(original.metadata.Splits.Training, replication.metadata.Splits.Training) &&
		reflect.DeepEqual(original.metadata.Splits.Validation, replication.metadata.Splits.Validation)
}

func loadPairedTop1(replication, original loadedRun) (pairedTop1, error) {
	path := filepath.Join(replication.layout.EvaluationsDir, pairedTop1ArtifactName)
	if err := requireRegular(path); err != nil {
		return pairedTop1{}, err
	}
	contents, err := os.ReadFile(path)
	if err != nil {
		return pairedTop1{}, err
	}
	var paired pairedTop1
	if err := json.Unmarshal(contents, &paired); err != nil {
		return pairedTop1{}, fmt.Errorf("decode paired teacher-top-1 artifact: %w", err)
	}
	if paired.OriginalRunID != original.layout.ID || paired.ReplicationRunID != replication.layout.ID || paired.OriginalSeed != original.config.Seed || paired.ReplicationSeed != replication.config.Seed || paired.OriginalBestUpdate != original.final.BestValidationStep || paired.ReplicationBestUpdate != replication.final.BestValidationStep || paired.ValidationSplitHash != original.evaluation.ValidationSplitHash {
		return pairedTop1{}, errors.New("paired teacher-top-1 artifact has mismatched run identity")
	}
	if paired.Examples != 2500 || paired.BothTeacherTop1+paired.OriginalOnly+paired.ReplicationOnly+paired.NeitherTeacherTop1 != paired.Examples {
		return pairedTop1{}, errors.New("paired teacher-top-1 counts do not cover 2,500 validation states")
	}
	if paired.BothTeacherTop1+paired.OriginalOnly != int(math.Round(original.final.BestValidation.Top1*float64(paired.Examples))) || paired.BothTeacherTop1+paired.ReplicationOnly != int(math.Round(replication.final.BestValidation.Top1*float64(paired.Examples))) {
		return pairedTop1{}, errors.New("paired teacher-top-1 counts do not match saved best accuracies")
	}
	for _, hash := range []string{paired.OriginalSelectionHash, paired.ReplicationHash} {
		decoded, err := hex.DecodeString(hash)
		if err != nil || len(decoded) != 32 {
			return pairedTop1{}, errors.New("paired teacher-top-1 artifact has an invalid selection hash")
		}
	}
	return paired, nil
}

func pairGames(original, replication []game) ([]PairedGame, error) {
	if len(original) != 100 || len(replication) != 100 {
		return nil, errors.New("paired game comparison requires two complete 100-game trajectories")
	}
	paired := make([]PairedGame, len(original))
	for index := range original {
		if original[index].Solution != replication[index].Solution {
			return nil, fmt.Errorf("paired game %d solutions differ: %q and %q", index, original[index].Solution, replication[index].Solution)
		}
		paired[index] = PairedGame{
			Solution: original[index].Solution, OriginalSolved: original[index].Solved, ReplicationSolved: replication[index].Solved,
			OriginalGuesses: original[index].Guesses, ReplicationGuesses: replication[index].Guesses,
			GuessDelta: replication[index].Guesses - original[index].Guesses,
		}
	}
	return paired, nil
}

func failureSets(original, replication []string) FailureSetComparison {
	originalSet, replicationSet := make(map[string]bool, len(original)), make(map[string]bool, len(replication))
	for _, solution := range original {
		originalSet[solution] = true
	}
	for _, solution := range replication {
		replicationSet[solution] = true
	}
	var comparison FailureSetComparison
	for _, solution := range original {
		if replicationSet[solution] {
			comparison.Both = append(comparison.Both, solution)
		} else {
			comparison.OriginalOnly = append(comparison.OriginalOnly, solution)
		}
	}
	for _, solution := range replication {
		if !originalSet[solution] {
			comparison.ReplicationOnly = append(comparison.ReplicationOnly, solution)
		}
	}
	return comparison
}

func formatWords(words []string) string {
	if len(words) == 0 {
		return "none"
	}
	return strings.Join(words, ", ")
}

func solvedLabel(solved bool) string {
	if solved {
		return "solved"
	}
	return "failed"
}
