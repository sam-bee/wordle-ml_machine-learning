package proofrun

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"os"
	"path/filepath"
	"reflect"
	"slices"
	"strings"
	"time"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/sam-bee/wordle-ml_machine-learning/imitationdata"
	"github.com/sam-bee/wordle-ml_machine-learning/policy"
	"github.com/sam-bee/wordle-ml_machine-learning/proofgames"
	"github.com/sam-bee/wordle-ml_machine-learning/proofmetrics"
	"github.com/sam-bee/wordle-ml_machine-learning/runmetadata"
	"github.com/sam-bee/wordle-ml_machine-learning/runstate"
	"github.com/sam-bee/wordle-ml_machine-learning/supervised"
	"github.com/sam-bee/wordle-ml_machine-learning/tensorboard"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

// Options are the deliberately small command surface for a proof run.
type Options struct {
	DataDir string
	RunsDir string
	RunID   string
	Stage   Stage
	StopAt  int64
}

// Result is written to final-metrics.json and printed by the command.
type Result struct {
	Stage                    Stage                           `json:"stage"`
	GlobalUpdate             int64                           `json:"global_update"`
	InitialValidation        Metrics                         `json:"initial_validation"`
	FinalValidation          Metrics                         `json:"final_validation"`
	BestValidation           Metrics                         `json:"best_validation"`
	BestValidationStep       int64                           `json:"best_validation_step"`
	InitialTraining          Metrics                         `json:"initial_training"`
	FinalTraining            Metrics                         `json:"final_training"`
	ValidationImprovements   int                             `json:"validation_improvements"`
	StoppedNormally          bool                            `json:"stopped_normally,omitempty"`
	Passed                   bool                            `json:"passed"`
	Warnings                 []string                        `json:"warnings,omitempty"`
	DataOverlapAudit         imitationdata.StateOverlapAudit `json:"data_overlap_audit"`
	MajorGroupLearning       MajorGroupLearning              `json:"major_group_learning"`
	OverfitProof             *OverfitProof                   `json:"overfit_proof,omitempty"`
	InitialValidationDetails ValidationSnapshot              `json:"initial_validation_details"`
	FinalValidationDetails   ValidationSnapshot              `json:"final_validation_details"`
	BestValidationDetails    ValidationSnapshot              `json:"best_validation_details"`
	ValidationSnapshots      []ValidationSnapshot            `json:"validation_snapshots,omitempty"`
	ResumeProof              *ResumeProof                    `json:"resume_proof,omitempty"`
	TelemetryProof           *MiniTelemetryProof             `json:"telemetry_proof,omitempty"`
	Evaluations              map[string]json.RawMessage      `json:"evaluations,omitempty"`
}

// InitialGamesEvaluation identifies the run-zero gameplay artifact recorded
// from an independently restored, pre-update checkpoint.
type InitialGamesEvaluation struct {
	RunID               string                `json:"run_id"`
	Stage               Stage                 `json:"stage"`
	Mode                string                `json:"mode"`
	Checkpoint          string                `json:"checkpoint"`
	CheckpointUpdate    int64                 `json:"checkpoint_update"`
	ValidationSplitHash string                `json:"validation_split_hash"`
	Validation          proofmetrics.Result   `json:"validation"`
	Games               proofgames.Evaluation `json:"games"`
}

// ResumeProof is the persisted, CPU-verifiable record that a mini-stage
// checkpoint at update 500 resumes the uninterrupted sampler sequence.
type ResumeProof struct {
	CheckpointNextRecordIDs             []int `json:"checkpoint_next_record_ids"`
	UninterruptedReferenceNextRecordIDs []int `json:"uninterrupted_reference_next_record_ids"`
	ResumeFromUpdate                    int64 `json:"resume_from_update,omitempty"`
	FirstResumedScalarUpdate            int64 `json:"first_resumed_scalar_update,omitempty"`
	Completed                           bool  `json:"completed"`
}

// MajorGroupLearning records sufficiently populated turn and shortlist groups
// whose validation loss improves at the best checkpoint. The two dimensions
// are kept separate so broad learning cannot be claimed from only one narrow
// slice of the state space.
type MajorGroupLearning struct {
	MinimumExamples int      `json:"minimum_examples"`
	TurnCount       int      `json:"turn_count"`
	TurnGroups      []string `json:"turn_groups"`
	ShortlistCount  int      `json:"shortlist_count"`
	ShortlistGroups []string `json:"shortlist_groups"`
	Count           int      `json:"count"`
	Groups          []string `json:"groups"`
}

// OverfitProof is the durable evidence for every one-batch overfit gate. It
// contains measurements taken from the fixed 128-example batch, rather than
// the EMA training curve used for ordinary progress reporting.
type OverfitProof struct {
	InitialFixedBatch               Metrics `json:"initial_fixed_batch"`
	FinalFixedBatch                 Metrics `json:"final_fixed_batch"`
	LossReducedAtLeastNinetyPercent bool    `json:"loss_reduced_at_least_90_percent"`
	Top1AtLeastNinetyFivePercent    bool    `json:"top1_at_least_95_percent"`
	DiagnosticsFinite               bool    `json:"diagnostics_finite"`
	ParametersFinite                bool    `json:"parameters_finite"`
	NonBiasWeightChanged            bool    `json:"non_bias_weight_changed"`
	CheckpointPredictionsReproduced bool    `json:"checkpoint_predictions_reproduced"`
}

// Metrics is the scalar subset needed by the proof gates.
type Metrics struct {
	Loss  float64 `json:"loss"`
	Top1  float64 `json:"top1"`
	Top5  float64 `json:"top5"`
	Top16 float64 `json:"top16"`
}

// validationBatchSize divides the fixed 2,500-record validation split exactly,
// avoiding a smaller final batch that would otherwise bias mean-of-batches
// metrics in GoMLX's metric accumulator.
const validationBatchSize = 100

// Run creates or resumes one fixed proof stage. It never opens the final test
// split. A mini run stopped at update 500 returns normally after writing the
// checkpoint needed for the explicit resume exercise.
func Run(options Options, stdout io.Writer) (Result, error) {
	config, err := ConfigFor(options.Stage)
	if err != nil {
		return Result{}, err
	}
	if strings.TrimSpace(options.DataDir) == "" || strings.TrimSpace(options.RunsDir) == "" {
		return Result{}, errors.New("data and runs directories must not be empty")
	}

	layout, resumed, err := prepareLayoutForRun(options.RunsDir, options.RunID, config, options.StopAt)
	if err != nil {
		return Result{}, err
	}
	if err := WriteOrValidateConfig(layout, config, resumed); err != nil {
		return Result{}, err
	}
	logFile, err := os.OpenFile(layout.TrainingLogPath, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return Result{}, fmt.Errorf("open training log: %w", err)
	}
	defer func() { _ = logFile.Close() }()
	log := func(format string, args ...any) {
		_, _ = fmt.Fprintf(logFile, format+"\n", args...)
	}

	vocab, err := vocabulary.Load(options.DataDir)
	if err != nil {
		return Result{}, fmt.Errorf("load vocabulary: %w", err)
	}
	trainData, err := imitationdata.Load(vocab, filepath.Join(options.DataDir, "imitation"), imitationdata.Train)
	if err != nil {
		return Result{}, fmt.Errorf("load training data: %w", err)
	}
	validationData, err := imitationdata.Load(vocab, filepath.Join(options.DataDir, "imitation"), imitationdata.Validation)
	if err != nil {
		return Result{}, fmt.Errorf("load validation data: %w", err)
	}
	dataOverlapAudit, err := imitationdata.AuditStateOverlap(trainData, validationData)
	if err != nil {
		return Result{}, fmt.Errorf("audit train/validation model-state overlap: %w", err)
	}
	warnings := make([]string, 0, 1)
	if dataOverlapAudit.OverlappingUniqueStates != 0 {
		warnings = append(warnings, fmt.Sprintf(
			"%d of %d unique validation model states also occur in training; their teacher top-1 labels agree. This is state-distribution overlap, not solution-ID split overlap.",
			dataOverlapAudit.OverlappingUniqueStates, dataOverlapAudit.ValidationUniqueStates,
		))
	}
	// Test data deliberately remains unopened during all proof stages.
	sourceData := trainData
	if config.Stage == Mini {
		sourceData, err = imitationdata.Load(vocab, filepath.Join(options.DataDir, "imitation"), imitationdata.Mini)
		if err != nil {
			return Result{}, fmt.Errorf("load mini training data: %w", err)
		}
	}
	opening, found, err := trainData.FindOpening()
	if err != nil {
		return Result{}, fmt.Errorf("find opening record: %w", err)
	}
	if !found {
		return Result{}, errors.New("training data has no opening record")
	}

	backend, err := compute.NewWithConfig("xla:cuda")
	if err != nil {
		return Result{}, fmt.Errorf("create xla:cuda backend: %w", err)
	}
	defer backend.Finalize()
	session, err := supervised.New(supervised.Config{
		Policy:       policy.Config{NumSolutions: vocabulary.NumSolutions, NumActions: vocabulary.NumActions},
		LearningRate: config.LearningRate,
		Seed:         config.Seed,
	}, backend)
	if err != nil {
		return Result{}, fmt.Errorf("create supervised session: %w", err)
	}
	defer session.Finalize()
	latest, err := supervised.NewCheckpoint(session.Store, layout.LatestCheckpointDir)
	if err != nil {
		return Result{}, fmt.Errorf("open latest checkpoint: %w", err)
	}
	events, err := tensorboard.New(layout.EventsDir)
	if err != nil {
		return Result{}, fmt.Errorf("open TensorBoard events: %w", err)
	}
	eventsClosed := false
	defer func() {
		if !eventsClosed {
			_ = events.Close()
		}
	}()

	state, err := initialState(session, resumed, config)
	if err != nil {
		return Result{}, err
	}
	if session.Trainer.GlobalStep() != state.GlobalUpdate {
		return Result{}, fmt.Errorf("checkpoint global update %d differs from run state %d", session.Trainer.GlobalStep(), state.GlobalUpdate)
	}
	if err := ValidateResumeState(config, state); err != nil {
		return Result{}, fmt.Errorf("validate resume state: %w", err)
	}
	if resumed {
		if err := warmValidationShapes(session, validationData, opening); err != nil {
			return Result{}, fmt.Errorf("warm restored inference before metadata validation: %w", err)
		}
		if err := validateManifestIdentity(layout, options.DataDir, config, trainData, session, backend); err != nil {
			return Result{}, err
		}
	}
	log("stage=%s resumed=%t global_update=%d", config.Stage, resumed, state.GlobalUpdate)

	if !resumed {
		if err := session.Trainer.ResetTrainMetrics(); err != nil {
			return Result{}, fmt.Errorf("reset training metrics: %w", err)
		}
	}
	initialValidation := Metrics{}
	bestValidation := Metrics{}
	initialValidationDetails := ValidationSnapshot{}
	finalValidationDetails := ValidationSnapshot{}
	bestValidationDetails := ValidationSnapshot{}
	validationSnapshots := make([]ValidationSnapshot, 0, config.TargetUpdates/config.ValidationEvery+1)
	bestStep := int64(0)
	initialTraining := Metrics{}
	finalTraining := Metrics{}
	finalValidation := Metrics{}
	validationImprovements := 0
	var resumeProof *ResumeProof
	var telemetryProof *MiniTelemetryProof
	evaluations := make(map[string]json.RawMessage)
	var sampler *imitationdata.TrainingSampler
	var uninterruptedReference *imitationdata.TrainingSampler
	if config.Stage != Overfit {
		sampler, err = imitationdata.NewTrainingSampler(sourceData, opening, config.BatchSize, config.Seed, imitationdata.Cursor{Epoch: state.DatasetEpoch, Offset: state.ShuffledCursor})
		if err != nil {
			return Result{}, fmt.Errorf("restore training sampler: %w", err)
		}
		if resumed {
			if err := validateSamplerPeek(state, sampler); err != nil {
				return Result{}, err
			}
		}
		if config.Stage == Mini && !resumed && options.StopAt != 0 {
			uninterruptedReference, err = imitationdata.NewTrainingSampler(sourceData, opening, config.BatchSize, config.Seed, imitationdata.Cursor{})
			if err != nil {
				return Result{}, fmt.Errorf("create uninterrupted sampler reference: %w", err)
			}
		}
	}
	progressResult := func(globalUpdate int64) Result {
		return Result{
			Stage:                    config.Stage,
			GlobalUpdate:             globalUpdate,
			InitialValidation:        initialValidation,
			FinalValidation:          finalValidation,
			BestValidation:           bestValidation,
			BestValidationStep:       bestStep,
			InitialTraining:          initialTraining,
			FinalTraining:            finalTraining,
			ValidationImprovements:   validationImprovements,
			StoppedNormally:          options.StopAt != 0 && globalUpdate == options.StopAt,
			Warnings:                 slices.Clone(warnings),
			DataOverlapAudit:         dataOverlapAudit,
			MajorGroupLearning:       majorGroupLearning(initialValidationDetails, bestValidationDetails),
			InitialValidationDetails: initialValidationDetails,
			FinalValidationDetails:   finalValidationDetails,
			BestValidationDetails:    bestValidationDetails,
			ValidationSnapshots:      validationSnapshots,
			ResumeProof:              cloneResumeProof(resumeProof),
			TelemetryProof:           cloneMiniTelemetryProof(telemetryProof),
			Evaluations:              cloneEvaluations(evaluations),
		}
	}
	if !resumed {
		evaluatedInitial, samples, err := evaluateDetailed(session, validationData, opening, func(action int) string {
			word, _ := vocab.ActionWord(action)
			return word
		})
		if err != nil {
			return Result{}, fmt.Errorf("initial validation: %w", err)
		}
		initialValidationDetails = evaluatedInitial
		if err := validateValidationSnapshotEvidence(initialValidationDetails, validationData.Len()); err != nil {
			return Result{}, fmt.Errorf("initial validation evidence: %w", err)
		}
		initialValidation = initialValidationDetails.Metrics
		initialValidationDetails.Update = 0
		bestValidation, bestStep = initialValidation, 0
		bestValidationDetails = initialValidationDetails
		finalValidationDetails = initialValidationDetails
		finalValidation = initialValidation
		validationSnapshots = append(validationSnapshots, initialValidationDetails)
		if err := writeManifest(layout, options.DataDir, config, trainData, session, backend); err != nil {
			return Result{}, err
		}
		state.BestValidation = &runstate.BestValidation{Value: initialValidation.Loss, Update: 0}
		if sampler != nil {
			cursor := sampler.Cursor()
			state.DatasetEpoch, state.ShuffledCursor, state.NextRecordIDs = cursor.Epoch, cursor.Offset, sampler.Peek()
		}
		if err := writeValidationTelemetry(events, 0, initialValidationDetails, samples, session); err != nil {
			return Result{}, err
		}
		if err := saveLatest(layout, latest, session, state); err != nil {
			return Result{}, err
		}
		if err := copyLatestCheckpointAtomically(latest, layout.InitialCheckpointDir); err != nil {
			return Result{}, fmt.Errorf("save initial checkpoint: %w", err)
		}
		if err := copyLatestCheckpointAtomically(latest, layout.BestCheckpointDir); err != nil {
			return Result{}, fmt.Errorf("save initial best checkpoint: %w", err)
		}
		verificationExamples, err := checkpointVerificationExamples(opening, validationData)
		if err != nil {
			return Result{}, err
		}
		if err := verifyCheckpointPredictions(backend, session, layout, config, verificationExamples); err != nil {
			return Result{}, fmt.Errorf("step-zero checkpoint reload: %w", err)
		}
		if config.Stage == Overfit {
			baseline, err := evaluateInitialGames10(context.Background(), backend, layout, options.RunID, config, vocab, opening, initialValidationDetails.Details)
			if err != nil {
				return Result{}, fmt.Errorf("run-zero games: %w", err)
			}
			encoded, err := json.Marshal(baseline)
			if err != nil {
				return Result{}, fmt.Errorf("encode run-zero games result: %w", err)
			}
			evaluations["initial-games10"] = encoded
			if err := events.WriteScalars(0, proofgames.TensorBoardScalars(baseline.Games)...); err != nil {
				return Result{}, fmt.Errorf("write run-zero game telemetry: %w", err)
			}
			if err := events.Flush(); err != nil {
				return Result{}, fmt.Errorf("flush run-zero game telemetry: %w", err)
			}
			log("games checkpoint=initial step=0 solved_fraction=%g mean_guesses=%g failures=%d", baseline.Games.Summary.SolvedFraction, baseline.Games.Summary.MeanGuesses, baseline.Games.Summary.Failures)
		}
		initialResult := progressResult(0)
		if err := validatePriorValidationEvidence(initialResult, validationData.Len()); err != nil {
			return Result{}, fmt.Errorf("persist initial validation evidence: %w", err)
		}
		if err := writeFinalMetrics(layout, initialResult); err != nil {
			return Result{}, err
		}
		log("validation step=0 loss=%g top1=%g top5=%g top16=%g opening_guess=%s", initialValidation.Loss, initialValidation.Top1, initialValidation.Top5, initialValidation.Top16, initialValidationDetails.OpeningWord)
	} else {
		previous, err := loadPriorResult(layout)
		if err != nil {
			return Result{}, err
		}
		initialValidation, initialTraining, finalTraining, finalValidation = previous.InitialValidation, previous.InitialTraining, previous.FinalTraining, previous.FinalValidation
		validationImprovements = previous.ValidationImprovements
		bestStep = state.BestValidation.Update
		bestValidation = previous.BestValidation
		initialValidationDetails, finalValidationDetails, bestValidationDetails = previous.InitialValidationDetails, previous.FinalValidationDetails, previous.BestValidationDetails
		if err := validatePriorValidationEvidence(previous, validationData.Len()); err != nil {
			return Result{}, fmt.Errorf("prior validation evidence: %w", err)
		}
		validationSnapshots = append(validationSnapshots, previous.ValidationSnapshots...)
		telemetryProof = cloneMiniTelemetryProof(previous.TelemetryProof)
		evaluations = cloneEvaluations(previous.Evaluations)
		if previous.GlobalUpdate != state.GlobalUpdate {
			return Result{}, fmt.Errorf("prior final metrics update %d differs from checkpoint state update %d", previous.GlobalUpdate, state.GlobalUpdate)
		}
		if previous.DataOverlapAudit != dataOverlapAudit {
			return Result{}, fmt.Errorf("prior data-overlap audit %+v differs from current audit %+v", previous.DataOverlapAudit, dataOverlapAudit)
		}
		if bestValidation.Loss != state.BestValidation.Value {
			return Result{}, fmt.Errorf("prior best validation loss %g differs from checkpoint state %g", bestValidation.Loss, state.BestValidation.Value)
		}
		if config.Stage == Mini {
			resumeProof, err = resumeProofForResume(state, previous.ResumeProof)
			if err != nil {
				return Result{}, err
			}
		}
	}

	var overfitBatch []imitationdata.Example
	if config.Stage == Overfit {
		if resumed {
			return Result{}, errors.New("overfit proof runs are not resumable because their initial gate measurements are intentionally in-memory")
		}
		overfitBatch, err = SelectOverfitBatch(trainData, config.Seed)
		if err != nil {
			return Result{}, fmt.Errorf("construct overfit batch: %w", err)
		}
	}
	initialOverfitMetrics := Metrics{}
	weightsBefore := map[string][]float32(nil)
	if config.Stage == Overfit {
		initialOverfitMetrics, err = evaluateExampleMetrics(session, overfitBatch)
		if err != nil {
			return Result{}, fmt.Errorf("initial overfit batch evaluation: %w", err)
		}
		weightsBefore, err = trainableWeights(session.Store)
		if err != nil {
			return Result{}, err
		}
	}

	limit := config.TargetUpdates
	if options.StopAt != 0 {
		limit = options.StopAt
	}
	if state.GlobalUpdate > limit {
		return Result{}, fmt.Errorf("checkpoint is already at update %d, beyond requested limit %d", state.GlobalUpdate, limit)
	}
	for session.Trainer.GlobalStep() < limit {
		inputStarted := time.Now()
		var examples []imitationdata.Example
		if config.Stage == Overfit {
			examples = overfitBatch
		} else {
			examples, err = sampler.Next()
			if err != nil {
				return Result{}, fmt.Errorf("sample training batch: %w", err)
			}
		}
		batch, err := imitationdata.MaterializeBatch(examples)
		if err != nil {
			return Result{}, fmt.Errorf("materialize training batch: %w", err)
		}
		inputWait := time.Since(inputStarted)
		stepStarted := time.Now()
		trainMetrics, err := session.Trainer.TrainStep(batch)
		if err != nil {
			return Result{}, fmt.Errorf("train update %d: %w", session.Trainer.GlobalStep()+1, err)
		}
		metrics, err := trainingMetricsFromTensors(trainMetrics)
		if err != nil {
			return Result{}, fmt.Errorf("read training metrics: %w", err)
		}
		step := session.Trainer.GlobalStep()
		batchDuration := time.Since(stepStarted)
		if !metrics.Finite() {
			return Result{}, fmt.Errorf("non-finite training metrics at update %d", step)
		}
		if initialTraining == (Metrics{}) {
			initialTraining = metrics
		}
		finalTraining = metrics
		diagnostics, err := session.Diagnostics()
		if err != nil {
			return Result{}, fmt.Errorf("read optimizer diagnostics at update %d: %w", step, err)
		}
		if err := validateTrainingDiagnostics(diagnostics); err != nil {
			return Result{}, fmt.Errorf("training safety check at update %d: %w", step, err)
		}
		if step%config.ScalarEvery == 0 {
			openingMetrics, openingWord, err := evaluateOpening(session, opening, func(action int) string { word, _ := vocab.ActionWord(action); return word })
			if err != nil {
				return Result{}, fmt.Errorf("opening telemetry update %d: %w", step, err)
			}
			epoch := int64(0)
			if sampler != nil {
				epoch = sampler.Cursor().Epoch
			}
			if err := writeTrainingTelemetry(events, step, metrics, diagnostics, examples, epoch, step*int64(config.BatchSize), batchDuration, inputWait, openingMetrics); err != nil {
				return Result{}, err
			}
			if resumed && config.Stage == Mini && resumeProof != nil && resumeProof.FirstResumedScalarUpdate == 0 {
				resumeProof.FirstResumedScalarUpdate = step
			}
			log("train step=%d opening_guess=%s opening_loss=%g opening_rank=%d", step, openingWord, openingMetrics.Loss, openingMetrics.TeacherRank)
		}
		if step%config.ValidationEvery != 0 && step != limit {
			continue
		}
		evaluatedValidation, samples, err := evaluateDetailed(session, validationData, opening, func(action int) string { word, _ := vocab.ActionWord(action); return word })
		if err != nil {
			return Result{}, fmt.Errorf("validation update %d: %w", step, err)
		}
		finalValidationDetails = evaluatedValidation
		finalValidation = finalValidationDetails.Metrics
		finalValidationDetails.Update = step
		if err := validateValidationSnapshotEvidence(finalValidationDetails, validationData.Len()); err != nil {
			return Result{}, fmt.Errorf("validation evidence at update %d: %w", step, err)
		}
		if !finalValidation.Finite() {
			return Result{}, fmt.Errorf("non-finite validation metrics at update %d", step)
		}
		validationSnapshots = append(validationSnapshots, finalValidationDetails)
		if err := writeValidationTelemetry(events, step, finalValidationDetails, samples, session); err != nil {
			return Result{}, err
		}
		if state.BestValidation == nil || finalValidation.Loss < state.BestValidation.Value {
			state.BestValidation = &runstate.BestValidation{Value: finalValidation.Loss, Update: step}
			bestValidation, bestStep = finalValidation, step
			bestValidationDetails = finalValidationDetails
		}
		if finalValidation.Top1 > initialValidation.Top1 && finalValidation.Top5 > initialValidation.Top5 && finalValidation.Top16 > initialValidation.Top16 {
			validationImprovements++
		}
		state.GlobalUpdate = step
		if sampler != nil {
			cursor := sampler.Cursor()
			state.DatasetEpoch, state.ShuffledCursor, state.NextRecordIDs = cursor.Epoch, cursor.Offset, sampler.Peek()
		}
		if err := saveLatest(layout, latest, session, state); err != nil {
			return Result{}, err
		}
		if config.Stage == Mini && !resumed && options.StopAt != 0 && step == options.StopAt {
			resumeProof, err = resumeProofForStop(state, uninterruptedReference, int(step))
			if err != nil {
				return Result{}, err
			}
		}
		if bestStep == step {
			if err := copyLatestCheckpointAtomically(latest, layout.BestCheckpointDir); err != nil {
				return Result{}, fmt.Errorf("save best checkpoint: %w", err)
			}
		}
		if resumed && config.Stage == Mini && step == config.TargetUpdates {
			if resumeProof == nil || resumeProof.FirstResumedScalarUpdate == 0 {
				return Result{}, errors.New("mini resume reached completion before recording a resumed scalar update")
			}
			resumeProof.Completed = true
		}
		progress := progressResult(step)
		if err := validatePriorValidationEvidence(progress, validationData.Len()); err != nil {
			return Result{}, fmt.Errorf("persist validation evidence at update %d: %w", step, err)
		}
		if err := writeFinalMetrics(layout, progress); err != nil {
			return Result{}, err
		}
		log("validation step=%d loss=%g top1=%g top5=%g top16=%g opening_guess=%s validation_seconds=%g", step, finalValidation.Loss, finalValidation.Top1, finalValidation.Top5, finalValidation.Top16, finalValidationDetails.OpeningWord, finalValidationDetails.DurationSeconds)
	}

	result := progressResult(session.Trainer.GlobalStep())
	if err := events.Close(); err != nil {
		return Result{}, fmt.Errorf("close TensorBoard events: %w", err)
	}
	eventsClosed = true
	if config.Stage == Mini && !result.StoppedNormally {
		proof, err := VerifyMiniTensorBoardEvents(layout.EventsDir)
		if err != nil {
			return Result{}, fmt.Errorf("verify continuous mini TensorBoard telemetry: %w", err)
		}
		telemetryProof = &proof
		result.TelemetryProof = cloneMiniTelemetryProof(telemetryProof)
	}
	var gateErr error
	if config.Stage == Overfit && !result.StoppedNormally {
		proof, err := enforceOverfitGate(session, backend, layout, overfitBatch, initialOverfitMetrics, weightsBefore)
		result.OverfitProof = &proof
		if err != nil {
			gateErr = err
		} else {
			result.Passed = true
		}
	}
	if config.Stage == Mini && !result.StoppedNormally {
		result.Passed = miniGatePassed(result)
		if !result.Passed {
			gateErr = fmt.Errorf("mini proof gate failed: train loss %g -> %g, top1 %g -> %g, top16 %g -> %g, resume proof %+v", result.InitialTraining.Loss, result.FinalTraining.Loss, result.InitialTraining.Top1, result.FinalTraining.Top1, result.InitialTraining.Top16, result.FinalTraining.Top16, result.ResumeProof)
		}
	}
	if config.Stage == Full {
		result.Passed = fullGatePassed(result)
		if !result.Passed {
			gateErr = fmt.Errorf("full proof gate failed: training loss %g -> %g, validation initial=%+v best=%+v, improving validation checkpoints %d, improving major groups turns=%d/%v shortlist=%d/%v (minimum %d examples)", result.InitialTraining.Loss, result.FinalTraining.Loss, result.InitialValidation, result.BestValidation, result.ValidationImprovements, result.MajorGroupLearning.TurnCount, result.MajorGroupLearning.TurnGroups, result.MajorGroupLearning.ShortlistCount, result.MajorGroupLearning.ShortlistGroups, result.MajorGroupLearning.MinimumExamples)
		}
	}
	if err := validatePriorValidationEvidence(result, validationData.Len()); err != nil {
		return Result{}, fmt.Errorf("persist final validation evidence: %w", err)
	}
	if err := writeFinalMetrics(layout, result); err != nil {
		return Result{}, err
	}
	if gateErr != nil {
		return result, gateErr
	}
	fmt.Fprintf(stdout, "stage=%s global_update=%d passed=%t\n", result.Stage, result.GlobalUpdate, result.Passed)
	return result, nil
}

func prepareLayoutForRun(root, runID string, config Config, stopAt int64) (runstate.Layout, bool, error) {
	layout, err := runstate.Open(root, runID)
	if err == nil {
		if err := validateStopAtForRun(config, true, stopAt); err != nil {
			return runstate.Layout{}, false, err
		}
		return layout, true, nil
	}
	planned, planErr := runstate.New(root, runID)
	if planErr != nil {
		return runstate.Layout{}, false, planErr
	}
	if _, statErr := os.Stat(planned.Dir); !errors.Is(statErr, os.ErrNotExist) {
		if statErr == nil {
			return runstate.Layout{}, false, fmt.Errorf("run path %q exists but is not an openable run directory", planned.Dir)
		}
		return runstate.Layout{}, false, fmt.Errorf("inspect planned run directory %q: %w", planned.Dir, statErr)
	}
	if err := validateStopAtForRun(config, false, stopAt); err != nil {
		return runstate.Layout{}, false, err
	}
	layout, createErr := runstate.Create(root, runID)
	if createErr != nil {
		return runstate.Layout{}, false, fmt.Errorf("open or create run %q: open error: %v; create error: %w", runID, err, createErr)
	}
	return layout, false, nil
}

func initialState(session *supervised.Session, resumed bool, config Config) (runstate.State, error) {
	if !resumed {
		return runstate.State{ShuffleSeed: config.Seed}, nil
	}
	state, err := runstate.LoadCheckpointState(session.Store)
	if err != nil {
		return runstate.State{}, fmt.Errorf("load state embedded in latest checkpoint: %w", err)
	}
	if state.BestValidation == nil {
		return runstate.State{}, errors.New("resumed checkpoint has no best validation state")
	}
	return state, nil
}

func validateSamplerPeek(state runstate.State, sampler *imitationdata.TrainingSampler) error {
	if sampler == nil {
		return errors.New("training sampler must not be nil")
	}
	peek := sampler.Peek()
	if !slices.Equal(state.NextRecordIDs, peek) {
		return fmt.Errorf("checkpoint next-record IDs %v do not match resumed sampler %v", state.NextRecordIDs, peek)
	}
	return nil
}

func resumeProofForStop(state runstate.State, uninterrupted *imitationdata.TrainingSampler, completedBatches int) (*ResumeProof, error) {
	if uninterrupted == nil {
		return nil, errors.New("uninterrupted sampler reference must not be nil")
	}
	if err := uninterrupted.AdvanceBatches(completedBatches); err != nil {
		return nil, fmt.Errorf("advance uninterrupted sampler reference: %w", err)
	}
	referenceIDs := uninterrupted.Peek()
	if !slices.Equal(state.NextRecordIDs, referenceIDs) {
		return nil, fmt.Errorf("checkpoint next-record IDs %v differ from uninterrupted reference %v", state.NextRecordIDs, referenceIDs)
	}
	return &ResumeProof{
		CheckpointNextRecordIDs:             slices.Clone(state.NextRecordIDs),
		UninterruptedReferenceNextRecordIDs: referenceIDs,
	}, nil
}

func resumeProofForResume(state runstate.State, previous *ResumeProof) (*ResumeProof, error) {
	if state.GlobalUpdate != 500 {
		return nil, fmt.Errorf("mini resume checkpoint is at update %d, want 500", state.GlobalUpdate)
	}
	if previous == nil {
		return nil, errors.New("prior mini result lacks resume proof")
	}
	if !slices.Equal(previous.CheckpointNextRecordIDs, previous.UninterruptedReferenceNextRecordIDs) {
		return nil, errors.New("prior mini resume proof has mismatched checkpoint and uninterrupted IDs")
	}
	if !slices.Equal(state.NextRecordIDs, previous.CheckpointNextRecordIDs) {
		return nil, errors.New("resumed checkpoint IDs differ from the persisted mini resume proof")
	}
	proof := cloneResumeProof(previous)
	proof.ResumeFromUpdate = state.GlobalUpdate
	return proof, nil
}

func cloneResumeProof(proof *ResumeProof) *ResumeProof {
	if proof == nil {
		return nil
	}
	clone := *proof
	clone.CheckpointNextRecordIDs = slices.Clone(proof.CheckpointNextRecordIDs)
	clone.UninterruptedReferenceNextRecordIDs = slices.Clone(proof.UninterruptedReferenceNextRecordIDs)
	return &clone
}

func cloneEvaluations(evaluations map[string]json.RawMessage) map[string]json.RawMessage {
	if len(evaluations) == 0 {
		return nil
	}
	clone := make(map[string]json.RawMessage, len(evaluations))
	for name, contents := range evaluations {
		clone[name] = slices.Clone(contents)
	}
	return clone
}

func cloneMiniTelemetryProof(proof *MiniTelemetryProof) *MiniTelemetryProof {
	if proof == nil {
		return nil
	}
	clone := &MiniTelemetryProof{
		TrainingSteps:       slices.Clone(proof.TrainingSteps),
		ValidationSteps:     slices.Clone(proof.ValidationSteps),
		HistogramStepsByTag: make(map[string][]int64, len(proof.HistogramStepsByTag)),
	}
	for tag, steps := range proof.HistogramStepsByTag {
		clone.HistogramStepsByTag[tag] = slices.Clone(steps)
	}
	return clone
}

func checkpointVerificationExamples(opening imitationdata.Example, validation *imitationdata.Data) ([]imitationdata.Example, error) {
	if validation == nil || validation.Split() != imitationdata.Validation {
		return nil, errors.New("checkpoint verification requires validation data")
	}
	for index := 0; index < validation.Len(); index++ {
		example, err := validation.Example(index)
		if err != nil {
			return nil, fmt.Errorf("expand checkpoint verification example %d: %w", index, err)
		}
		if example.Turn > 0 {
			return []imitationdata.Example{opening, example}, nil
		}
	}
	return nil, errors.New("validation data has no post-opening state for checkpoint verification")
}

func miniGatePassed(result Result) bool {
	proof := result.ResumeProof
	return result.GlobalUpdate == 1000 &&
		result.InitialTraining.Finite() &&
		result.FinalTraining.Finite() &&
		result.FinalTraining.Loss < result.InitialTraining.Loss*0.5 &&
		result.FinalTraining.Top1 >= result.InitialTraining.Top1+0.1 &&
		result.FinalTraining.Top16 >= result.InitialTraining.Top16+0.1 &&
		proof != nil &&
		proof.ResumeFromUpdate == 500 &&
		proof.FirstResumedScalarUpdate == 510 &&
		proof.Completed &&
		slices.Equal(proof.CheckpointNextRecordIDs, proof.UninterruptedReferenceNextRecordIDs) &&
		result.TelemetryProof != nil
}

func fullGatePassed(result Result) bool {
	return result.GlobalUpdate == 2000 &&
		result.InitialTraining.Finite() && result.FinalTraining.Finite() &&
		result.FinalTraining.Loss < result.InitialTraining.Loss &&
		result.BestValidation.Loss <= result.InitialValidation.Loss*0.95 &&
		result.BestValidation.Top1 > result.InitialValidation.Top1 &&
		result.BestValidation.Top5 > result.InitialValidation.Top5 &&
		result.BestValidation.Top16 > result.InitialValidation.Top16 &&
		result.ValidationImprovements >= 2 &&
		result.MajorGroupLearning.TurnCount >= 2 &&
		result.MajorGroupLearning.ShortlistCount >= 2
}

func evaluateInitialGames10(
	ctx context.Context,
	backend compute.Backend,
	layout runstate.Layout,
	runID string,
	config Config,
	vocab *vocabulary.Vocabulary,
	opening imitationdata.Example,
	validation proofmetrics.Result,
) (InitialGamesEvaluation, error) {
	restored, err := supervised.New(supervised.Config{
		Policy:       policy.Config{NumSolutions: vocabulary.NumSolutions, NumActions: vocabulary.NumActions},
		LearningRate: config.LearningRate,
		Seed:         config.Seed,
	}, backend)
	if err != nil {
		return InitialGamesEvaluation{}, fmt.Errorf("create independent initial-checkpoint session: %w", err)
	}
	defer restored.Finalize()
	if _, err := supervised.NewCheckpoint(restored.Store, layout.InitialCheckpointDir); err != nil {
		return InitialGamesEvaluation{}, fmt.Errorf("load independent initial checkpoint: %w", err)
	}
	state, err := runstate.LoadCheckpointState(restored.Store)
	if err != nil {
		return InitialGamesEvaluation{}, fmt.Errorf("read initial checkpoint state: %w", err)
	}
	if state.GlobalUpdate != 0 || restored.Trainer.GlobalStep() != 0 {
		return InitialGamesEvaluation{}, fmt.Errorf("initial checkpoint is at update %d (trainer %d), want zero", state.GlobalUpdate, restored.Trainer.GlobalStep())
	}
	if err := warmInference(restored, []imitationdata.Example{opening}); err != nil {
		return InitialGamesEvaluation{}, fmt.Errorf("warm independent initial-checkpoint inference: %w", err)
	}
	before, err := StoreFingerprint(restored.Store)
	if err != nil {
		return InitialGamesEvaluation{}, fmt.Errorf("fingerprint initial checkpoint before games: %w", err)
	}
	evaluation, err := proofgames.EvaluateInitialGames10(ctx, restored, vocab)
	if err != nil {
		return InitialGamesEvaluation{}, err
	}
	after, err := StoreFingerprint(restored.Store)
	if err != nil {
		return InitialGamesEvaluation{}, fmt.Errorf("fingerprint initial checkpoint after games: %w", err)
	}
	if before != after {
		return InitialGamesEvaluation{}, fmt.Errorf("run-zero gameplay mutated Store: before %s, after %s", before, after)
	}
	gamesJSONL, err := proofgames.JSONL(evaluation.Games)
	if err != nil {
		return InitialGamesEvaluation{}, fmt.Errorf("encode run-zero trajectories: %w", err)
	}
	artifact := InitialGamesEvaluation{
		RunID:               runID,
		Stage:               Overfit,
		Mode:                "games10",
		Checkpoint:          "initial",
		CheckpointUpdate:    0,
		ValidationSplitHash: vocab.Hashes().Validation,
		Validation:          validation,
		Games:               evaluation,
	}
	if err := layout.WriteEvaluationGamesJSONL("initial", "games10", gamesJSONL); err != nil {
		return InitialGamesEvaluation{}, fmt.Errorf("write immutable run-zero trajectories: %w", err)
	}
	if err := layout.WriteValidationGamesJSONL(gamesJSONL); err != nil {
		return InitialGamesEvaluation{}, fmt.Errorf("write canonical run-zero trajectories: %w", err)
	}
	if err := layout.WriteEvaluationJSON("initial", "games10", artifact); err != nil {
		return InitialGamesEvaluation{}, fmt.Errorf("write immutable run-zero evaluation: %w", err)
	}
	return artifact, nil
}

func saveLatest(layout runstate.Layout, latest interface{ Save() error }, session *supervised.Session, state runstate.State) error {
	state.GlobalUpdate = session.Trainer.GlobalStep()
	if err := runstate.SaveCheckpointState(session.Store, state); err != nil {
		return fmt.Errorf("embed run state in checkpoint: %w", err)
	}
	if err := latest.Save(); err != nil {
		return fmt.Errorf("save latest checkpoint: %w", err)
	}
	if err := layout.WriteStateMirror(state); err != nil {
		return fmt.Errorf("write latest checkpoint state mirror: %w", err)
	}
	return nil
}

func evaluateExampleMetrics(session *supervised.Session, examples []imitationdata.Example) (Metrics, error) {
	var collector proofmetrics.Collector
	if err := evaluateDiagnosticExamples(session, examples, &collector, nil, false); err != nil {
		return Metrics{}, err
	}
	result := collector.Result()
	metrics := Metrics{
		Loss:  result.Loss,
		Top1:  result.Top1Accuracy,
		Top5:  result.Top5Accuracy,
		Top16: result.Top16Accuracy,
	}
	if !metrics.Finite() {
		return Metrics{}, fmt.Errorf("fixed-batch evaluation has non-finite metrics: %+v", metrics)
	}
	return metrics, nil
}

// trainingMetricsFromTensors decodes Trainer.TrainStep's fixed output order:
// batch loss, EMA loss, top-1, top-5, top-16. The EMA loss is the stable
// training curve and the top-k tensors begin at index two.
func trainingMetricsFromTensors(values []*tensors.Tensor) (Metrics, error) {
	defer finalize(values)
	if len(values) != 5 {
		return Metrics{}, fmt.Errorf("got %d training metrics, want batch loss, EMA loss, top1, top5, top16", len(values))
	}
	indices := [...]int{1, 2, 3, 4}
	flat := make([]float64, len(indices))
	for outputIndex, metricIndex := range indices {
		value, err := tensors.CopyFlatData[float32](values[metricIndex])
		if err != nil || len(value) != 1 {
			if err == nil {
				err = errors.New("training metric is not scalar")
			}
			return Metrics{}, err
		}
		flat[outputIndex] = float64(value[0])
	}
	return Metrics{Loss: flat[0], Top1: flat[1], Top5: flat[2], Top16: flat[3]}, nil
}

func (metrics Metrics) Finite() bool {
	for _, value := range []float64{metrics.Loss, metrics.Top1, metrics.Top5, metrics.Top16} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return false
		}
	}
	return true
}

const majorGroupMinimumExamples = 25

func majorGroupLearning(initial, best ValidationSnapshot) MajorGroupLearning {
	appendImproved := func(before, after []proofmetrics.GroupResult) []string {
		groups := make([]string, 0, len(before))
		for index := range before {
			if index >= len(after) || before[index].Examples < majorGroupMinimumExamples || after[index].Examples < majorGroupMinimumExamples {
				continue
			}
			if after[index].Loss < before[index].Loss {
				groups = append(groups, before[index].Name)
			}
		}
		return groups
	}
	turns := appendImproved(initial.Details.ByTurn, best.Details.ByTurn)
	shortlists := appendImproved(initial.Details.ByShortlistBucket, best.Details.ByShortlistBucket)
	groups := make([]string, 0, len(turns)+len(shortlists))
	for _, name := range turns {
		groups = append(groups, "turn/"+name)
	}
	for _, name := range shortlists {
		groups = append(groups, "shortlist/"+name)
	}
	return MajorGroupLearning{
		MinimumExamples: majorGroupMinimumExamples,
		TurnCount:       len(turns),
		TurnGroups:      turns,
		ShortlistCount:  len(shortlists),
		ShortlistGroups: shortlists,
		Count:           len(groups),
		Groups:          groups,
	}
}

func validateValidationSnapshotEvidence(snapshot ValidationSnapshot, expectedExamples int) error {
	if expectedExamples <= 0 {
		return fmt.Errorf("expected validation examples must be positive, got %d", expectedExamples)
	}
	if snapshot.Details.Examples != expectedExamples {
		return fmt.Errorf("details contain %d examples, want %d", snapshot.Details.Examples, expectedExamples)
	}
	detailMetrics := Metrics{
		Loss:  snapshot.Details.Loss,
		Top1:  snapshot.Details.Top1Accuracy,
		Top5:  snapshot.Details.Top5Accuracy,
		Top16: snapshot.Details.Top16Accuracy,
	}
	if !detailMetrics.Finite() || snapshot.Metrics != detailMetrics {
		return fmt.Errorf("summary metrics %+v differ from finite details %+v", snapshot.Metrics, detailMetrics)
	}
	if snapshot.Details.EntropyExamples != expectedExamples || !finiteMetric(snapshot.Details.OutputEntropy) || snapshot.Details.OutputEntropy < 0 ||
		snapshot.Details.RawArgmaxUnavailable < 0 || snapshot.Details.RawArgmaxUnavailable > expectedExamples || snapshot.Details.MaskedArgmaxViolations != 0 {
		return errors.New("overall entropy or action-selection counters are invalid")
	}
	turnNames := [...]string{"turn_1", "turn_2", "turn_3", "turn_4", "turn_5", "turn_6"}
	if err := validateValidationGroups("turn", snapshot.Details.ByTurn, turnNames[:], expectedExamples); err != nil {
		return err
	}
	shortlistNames := [...]string{"1", "2-5", "6-20", "21-100", ">100"}
	if err := validateValidationGroups("shortlist", snapshot.Details.ByShortlistBucket, shortlistNames[:], expectedExamples); err != nil {
		return err
	}
	opening := snapshot.Details.Opening
	// TeacherRank is the opening teacher action's rank among every legal model
	// action, not its position in the retained teacher top-16 labels. An
	// untrained model can therefore assign it a rank well above 16.
	if !opening.Present || !finiteMetric(opening.Loss) || opening.TeacherRank <= 0 || opening.HighestGuess < 0 {
		return errors.New("opening-state evidence is absent or invalid")
	}
	return nil
}

func validateValidationGroups(label string, groups []proofmetrics.GroupResult, names []string, expectedExamples int) error {
	if len(groups) != len(names) {
		return fmt.Errorf("%s groups = %d, want %d", label, len(groups), len(names))
	}
	examples := 0
	for index, name := range names {
		group := groups[index]
		if group.Name != name {
			return fmt.Errorf("%s group %d = %q, want %q", label, index, group.Name, name)
		}
		if group.Examples < 0 || group.EntropyExamples != group.Examples || group.RawArgmaxUnavailable < 0 || group.RawArgmaxUnavailable > group.Examples || group.MaskedArgmaxViolations != 0 ||
			!finiteMetric(group.Loss) || !finiteMetric(group.Top1Accuracy) || !finiteMetric(group.Top5Accuracy) || !finiteMetric(group.Top16Accuracy) || !finiteMetric(group.OutputEntropy) ||
			group.Top1Accuracy < 0 || group.Top1Accuracy > 1 || group.Top5Accuracy < 0 || group.Top5Accuracy > 1 || group.Top16Accuracy < 0 || group.Top16Accuracy > 1 || group.OutputEntropy < 0 {
			return fmt.Errorf("%s group %q has invalid metrics or counters", label, name)
		}
		examples += group.Examples
	}
	if examples != expectedExamples {
		return fmt.Errorf("%s groups contain %d examples, want %d", label, examples, expectedExamples)
	}
	return nil
}

func validatePriorValidationEvidence(result Result, expectedExamples int) error {
	checks := []struct {
		name     string
		snapshot ValidationSnapshot
		metrics  Metrics
		update   int64
	}{
		{"initial", result.InitialValidationDetails, result.InitialValidation, 0},
		{"final", result.FinalValidationDetails, result.FinalValidation, result.GlobalUpdate},
		{"best", result.BestValidationDetails, result.BestValidation, result.BestValidationStep},
	}
	for _, check := range checks {
		if err := validateValidationSnapshotEvidence(check.snapshot, expectedExamples); err != nil {
			return fmt.Errorf("%s snapshot: %w", check.name, err)
		}
		if check.snapshot.Metrics != check.metrics || check.snapshot.Update != check.update {
			return fmt.Errorf("%s snapshot summary or update differs from the top-level result", check.name)
		}
	}
	if len(result.ValidationSnapshots) == 0 {
		return errors.New("validation snapshot history is empty")
	}
	for index, snapshot := range result.ValidationSnapshots {
		if err := validateValidationSnapshotEvidence(snapshot, expectedExamples); err != nil {
			return fmt.Errorf("history snapshot %d: %w", index, err)
		}
	}
	return nil
}

func finiteMetric(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func finalize(values []*tensors.Tensor) {
	for _, value := range values {
		if value != nil {
			_ = value.FinalizeAll()
		}
	}
}

func verifyCheckpointPredictions(backend compute.Backend, session *supervised.Session, layout runstate.Layout, config Config, examples []imitationdata.Example) error {
	before, err := predictions(session, examples)
	if err != nil {
		return err
	}
	restored, err := supervised.New(supervised.Config{Policy: session.Policy.Config(), LearningRate: config.LearningRate, Seed: config.Seed}, backend)
	if err != nil {
		return err
	}
	defer restored.Finalize()
	if _, err := supervised.NewCheckpoint(restored.Store, layout.LatestCheckpointDir); err != nil {
		return fmt.Errorf("open independently restored checkpoint: %w", err)
	}
	after, err := predictions(restored, examples)
	if err != nil {
		return err
	}
	if len(before) != len(after) {
		return errors.New("checkpoint reload returned a different prediction shape")
	}
	for index := range before {
		if math.Abs(float64(before[index]-after[index])) > 1e-5 {
			return fmt.Errorf("checkpoint prediction differs at %d: %g vs %g", index, before[index], after[index])
		}
	}
	return nil
}

func enforceOverfitGate(session *supervised.Session, backend compute.Backend, layout runstate.Layout, examples []imitationdata.Example, initial Metrics, weightsBefore map[string][]float32) (OverfitProof, error) {
	proof := OverfitProof{InitialFixedBatch: initial}
	if !initial.Finite() {
		return proof, errors.New("overfit gate failed: initial fixed-batch metrics are non-finite")
	}
	finalMetrics, err := evaluateExampleMetrics(session, examples)
	if err != nil {
		return proof, fmt.Errorf("final overfit evaluation: %w", err)
	}
	proof.FinalFixedBatch = finalMetrics
	if !finalMetrics.Finite() {
		return proof, errors.New("overfit gate failed: final fixed-batch metrics are non-finite")
	}
	if finalMetrics.Loss > initial.Loss*0.1 {
		return proof, fmt.Errorf("overfit gate failed: loss fell from %g to %g, need at least 90%% reduction", initial.Loss, finalMetrics.Loss)
	}
	proof.LossReducedAtLeastNinetyPercent = true
	if finalMetrics.Top1 < 0.95 {
		return proof, fmt.Errorf("overfit gate failed: top-1 agreement %g, need at least 0.95", finalMetrics.Top1)
	}
	proof.Top1AtLeastNinetyFivePercent = true
	diagnostics, err := session.Diagnostics()
	if err != nil {
		return proof, fmt.Errorf("read final overfit diagnostics: %w", err)
	}
	if err := validateTrainingDiagnostics(diagnostics); err != nil {
		return proof, fmt.Errorf("overfit gate failed: final diagnostics: %w", err)
	}
	proof.DiagnosticsFinite = true
	proof.ParametersFinite = diagnostics.ParametersFinite
	changed, err := nonBiasWeightChanged(session.Store, weightsBefore)
	if err != nil {
		return proof, err
	}
	if !changed {
		return proof, errors.New("overfit gate failed: no non-bias trainable weight changed")
	}
	proof.NonBiasWeightChanged = true
	before, err := predictions(session, examples)
	if err != nil {
		return proof, err
	}
	restored, err := supervised.New(supervised.Config{Policy: session.Policy.Config(), LearningRate: 1e-3, Seed: Seed}, backend)
	if err != nil {
		return proof, err
	}
	defer restored.Finalize()
	if _, err := supervised.NewCheckpoint(restored.Store, layout.LatestCheckpointDir); err != nil {
		return proof, fmt.Errorf("reload overfit checkpoint: %w", err)
	}
	after, err := predictions(restored, examples)
	if err != nil {
		return proof, err
	}
	if len(before) != len(after) {
		return proof, errors.New("reloaded overfit checkpoint returned a different prediction shape")
	}
	for index := range before {
		if math.Abs(float64(before[index]-after[index])) > 1e-5 {
			return proof, fmt.Errorf("reloaded overfit checkpoint prediction differs at %d: %g vs %g", index, before[index], after[index])
		}
	}
	proof.CheckpointPredictionsReproduced = true
	return proof, nil
}

func trainableWeights(store *model.Store) (map[string][]float32, error) {
	weights := make(map[string][]float32)
	for variable := range store.IterVariables() {
		if !variable.Trainable || strings.Contains(variable.Path(), "bias") {
			continue
		}
		values, err := tensors.CopyFlatData[float32](variable.MustValue())
		if err != nil {
			return nil, err
		}
		weights[variable.Path()] = values
	}
	return weights, nil
}

func nonBiasWeightChanged(store *model.Store, before map[string][]float32) (bool, error) {
	for variable := range store.IterVariables() {
		if !variable.Trainable || strings.Contains(variable.Path(), "bias") {
			continue
		}
		old, found := before[variable.Path()]
		if !found {
			continue
		}
		current, err := tensors.CopyFlatData[float32](variable.MustValue())
		if err != nil {
			return false, err
		}
		if len(old) != len(current) {
			return false, fmt.Errorf("weight %q shape changed during overfit run", variable.Path())
		}
		for index := range old {
			if old[index] != current[index] {
				return true, nil
			}
		}
	}
	return false, nil
}

func predictions(session *supervised.Session, examples []imitationdata.Example) ([]float32, error) {
	batch, err := imitationdata.MaterializeBatch(examples)
	if err != nil {
		return nil, err
	}
	defer finalize(batch.Inputs)
	defer finalize(batch.Labels)
	output, beta, err := session.Predict(batch.Inputs[0], batch.Inputs[1], batch.Inputs[2], batch.Inputs[3], batch.Inputs[4])
	if err != nil {
		return nil, err
	}
	defer func() { _ = output.FinalizeAll() }()
	defer func() { _ = beta.FinalizeAll() }()
	return tensors.CopyFlatData[float32](output)
}

func writeFinalMetrics(layout runstate.Layout, result Result) error {
	if err := layout.WriteFinalMetricsJSON(result); err != nil {
		return fmt.Errorf("write final metrics: %w", err)
	}
	return nil
}

func loadPriorResult(layout runstate.Layout) (Result, error) {
	contents, err := os.ReadFile(layout.FinalMetricsPath)
	if err != nil {
		return Result{}, fmt.Errorf("read prior proof result %q: %w", layout.FinalMetricsPath, err)
	}
	var result Result
	if err := json.Unmarshal(contents, &result); err != nil {
		return Result{}, fmt.Errorf("decode prior proof result %q: %w", layout.FinalMetricsPath, err)
	}
	if !result.InitialValidation.Finite() {
		return Result{}, fmt.Errorf("prior proof result %q lacks a finite step-zero validation result", layout.FinalMetricsPath)
	}
	return result, nil
}

func writeManifest(layout runstate.Layout, dataDir string, config Config, trainData *imitationdata.Data, session *supervised.Session, backend compute.Backend) error {
	manifest, err := collectManifest(dataDir, config, trainData, session, backend)
	if err != nil {
		return err
	}
	if err := layout.WriteMetadata(manifest); err != nil {
		return fmt.Errorf("write immutable run metadata: %w", err)
	}
	return nil
}

func collectManifest(dataDir string, config Config, trainData *imitationdata.Data, session *supervised.Session, backend compute.Backend) (runmetadata.Manifest, error) {
	mlRepository, err := repositoryPath("WORDLEML_MACHINE_LEARNING_REPO_DIR")
	if err != nil {
		return runmetadata.Manifest{}, err
	}
	syntheticRepository, err := repositoryPath("WORDLEML_SYNTHETIC_REPO_DIR")
	if err != nil {
		return runmetadata.Manifest{}, err
	}
	gameRepository, err := repositoryPath("WORDLEML_GAME_ENGINE_REPO_DIR")
	if err != nil {
		return runmetadata.Manifest{}, err
	}
	runtimeMetadata, err := selectedRuntimeMetadata(backend)
	if err != nil {
		return runmetadata.Manifest{}, err
	}
	effectiveConfig, err := json.Marshal(config)
	if err != nil {
		return runmetadata.Manifest{}, fmt.Errorf("encode effective proof configuration: %w", err)
	}
	imitationDir := filepath.Join(dataDir, "imitation")
	// The sealed test WDIT files are deliberately outside this manifest: even
	// hashing them opens the teacher records.  The canonical test wordlist is
	// retained below as a split identity only.
	datasetFiles := make([]string, 0, 6)
	for _, split := range []string{"mini", "train", "validation"} {
		datasetFiles = append(datasetFiles, filepath.Join(imitationDir, "wordle-"+split+".bin"), filepath.Join(imitationDir, "wordle-"+split+".json"))
	}
	manifest, err := runmetadata.Collect(runmetadata.CollectOptions{
		RepositoryRoot:            mlRepository,
		MachineLearningRepository: mlRepository,
		SyntheticDataRepository:   syntheticRepository,
		GameEngineRepository:      gameRepository,
		DatasetFormat:             "WDIT",
		DatasetVersion:            fmt.Sprintf("%d", trainData.Metadata().Version),
		DatasetFiles:              datasetFiles,
		Vocabulary: []string{
			filepath.Join(dataDir, "wordlist-action-space-4739.csv"),
			filepath.Join(dataDir, "wordlist-valid-solutions-all-2309.csv"),
		},
		TrainingSplit:   []string{filepath.Join(dataDir, "wordlist-valid-solutions-train-2109.csv")},
		ValidationSplit: []string{filepath.Join(dataDir, "wordlist-valid-solutions-validation-100.csv")},
		// This hashes the sealed test split without loading its WDIT records.
		TestSplit:           []string{filepath.Join(dataDir, "wordlist-valid-solutions-test-100.csv")},
		ModelParameterCount: trainableParameterCount(session.Store),
		Runtime:             runtimeMetadata,
		Seed:                config.Seed,
		EffectiveConfig:     effectiveConfig,
	})
	if err != nil {
		return runmetadata.Manifest{}, fmt.Errorf("collect immutable run metadata: %w", err)
	}
	return manifest, nil
}

func validateManifestIdentity(layout runstate.Layout, dataDir string, config Config, trainData *imitationdata.Data, session *supervised.Session, backend compute.Backend) error {
	contents, err := os.ReadFile(layout.MetadataPath)
	if err != nil {
		return fmt.Errorf("read immutable run metadata %q: %w", layout.MetadataPath, err)
	}
	var stored runmetadata.Manifest
	if err := json.Unmarshal(contents, &stored); err != nil {
		return fmt.Errorf("decode immutable run metadata %q: %w", layout.MetadataPath, err)
	}
	if err := stored.Validate(); err != nil {
		return fmt.Errorf("validate immutable run metadata %q: %w", layout.MetadataPath, err)
	}
	current, err := collectManifest(dataDir, config, trainData, session, backend)
	if err != nil {
		return err
	}
	same, err := manifestsHaveSameIdentity(stored, current)
	if err != nil {
		return fmt.Errorf("compare immutable run metadata: %w", err)
	}
	if !same {
		return errors.New("current run inputs or environment do not match immutable metadata.json")
	}
	return nil
}

// manifestsHaveSameIdentity compares semantic manifest values. metadata.json
// is indented on disk, and json.RawMessage preserves that indentation when it
// is decoded. Comparing the raw EffectiveConfig bytes would therefore reject
// every otherwise-identical resume against the compact json.Marshal output
// collected by the new process.
func manifestsHaveSameIdentity(stored, current runmetadata.Manifest) (bool, error) {
	// Collection time is the sole intentionally changing field. Every input
	// identity, hash, commit, runtime fact, and effective config must agree.
	current.CollectedAt = stored.CollectedAt
	for label, manifest := range map[string]*runmetadata.Manifest{
		"stored":  &stored,
		"current": &current,
	} {
		var configuration any
		if err := json.Unmarshal(manifest.EffectiveConfig, &configuration); err != nil {
			return false, fmt.Errorf("decode %s effective config: %w", label, err)
		}
		canonical, err := json.Marshal(configuration)
		if err != nil {
			return false, fmt.Errorf("canonicalize %s effective config: %w", label, err)
		}
		manifest.EffectiveConfig = canonical
	}
	return reflect.DeepEqual(stored, current), nil
}

func repositoryPath(name string) (string, error) {
	path := strings.TrimSpace(os.Getenv(name))
	if path == "" {
		return "", fmt.Errorf("%s must name the repository used for this run", name)
	}
	return path, nil
}

func selectedRuntimeMetadata(backend compute.Backend) (runmetadata.RuntimeMetadata, error) {
	probe, err := runmetadata.ProbeRuntime(context.Background(), runmetadata.ProbeOptions{
		PJRTDetails:  map[string]string{"backend_description": backend.Description()},
		GoMLXDetails: map[string]string{"backend_name": backend.Name()},
	})
	if err != nil {
		return runmetadata.RuntimeMetadata{}, fmt.Errorf("probe selected CUDA runtime: %w", err)
	}
	return runmetadata.RuntimeMetadata{
		Backend:      "xla:cuda",
		GoMLXDetails: probe.GoMLXDetails,
		GPUDetails:   probe.GPUDetails,
		CUDADetails:  probe.CUDADetails,
		PJRTDetails:  probe.PJRTDetails,
	}, nil
}

func trainableParameterCount(store *model.Store) int64 {
	var count int64
	for variable := range store.IterVariables() {
		if variable.Trainable {
			count += int64(variable.Shape().Size())
		}
	}
	return count
}
