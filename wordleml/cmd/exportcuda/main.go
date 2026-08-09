// Command exportcuda restores one completed GoMLX checkpoint and converts its
// policy weights into the small, neutral FP32 CUDA artifact.  GoMLX is kept
// entirely here: the exported model, golden vectors, and game references are
// consumed by GoMLX-free commands and the cgo/CUDA web application.
package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/sam-bee/wordle-ml_machine-learning/cudamodel"
	"github.com/sam-bee/wordle-ml_machine-learning/cudaref"
	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/policy"
	"github.com/sam-bee/wordle-ml_machine-learning/proofeval"
	"github.com/sam-bee/wordle-ml_machine-learning/proofrun"
	"github.com/sam-bee/wordle-ml_machine-learning/runmetadata"
	"github.com/sam-bee/wordle-ml_machine-learning/runstate"
	"github.com/sam-bee/wordle-ml_machine-learning/supervised"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

const (
	defaultCheckpoint            = "best"
	exportDirectoryFormat        = "cuda-f32-v1"
	targetGoldenVectorCount      = 32
	portableMaximumAbsoluteError = 1e-3
)

type config struct {
	dataDir    string
	runsDir    string
	runID      string
	checkpoint string
	output     string
}

type loadedCheckpoint struct {
	session            *supervised.Session
	state              runstate.State
	checkpointIdentity string
}

type capturedPosition struct {
	position gameeval.Position
	logits   []float32
}

func main() {
	if err := run(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintf(os.Stderr, "exportcuda: %v\n", err)
		os.Exit(1)
	}
}

func run(args []string, stdout, stderr io.Writer) error {
	configuration, err := parseConfig(args, stderr)
	if err != nil {
		return err
	}
	layout, err := runstate.Open(configuration.runsDir, configuration.runID)
	if err != nil {
		return fmt.Errorf("open run: %w", err)
	}
	trainingConfig, err := proofeval.ReadConfig(layout)
	if err != nil {
		return fmt.Errorf("read immutable run configuration: %w", err)
	}
	manifest, err := readAndValidateRunManifest(layout, configuration.dataDir, trainingConfig)
	if err != nil {
		return err
	}
	final, err := readAndValidateFinalMetrics(layout, trainingConfig)
	if err != nil {
		return err
	}
	vocab, err := vocabulary.LoadWithoutFinalTest(configuration.dataDir)
	if err != nil {
		return fmt.Errorf("load sealed-test vocabulary: %w", err)
	}
	backend, err := compute.NewWithConfig("xla:cuda")
	if err != nil {
		return fmt.Errorf("create xla:cuda exporter backend: %w", err)
	}
	defer backend.Finalize()
	if err := runmetadata.VerifyEvaluationRuntime(manifest, backend.Name(), backend.Description()); err != nil {
		return fmt.Errorf("verify exporter runtime identity: %w", err)
	}
	loaded, err := restoreCheckpoint(backend, layout, trainingConfig, configuration.checkpoint, final)
	if err != nil {
		return err
	}
	defer loaded.session.Finalize()

	if err := materializePolicy(loaded.session, vocab); err != nil {
		return err
	}
	weights, err := exportTrainableWeights(loaded.session)
	if err != nil {
		return err
	}
	if len(weights) != cudamodel.ParameterCount {
		return fmt.Errorf("exported %d weights, want %d", len(weights), cudamodel.ParameterCount)
	}
	vectors, games, err := generateReferences(context.Background(), loaded.session, vocab)
	if err != nil {
		return err
	}
	vectors, err = chooseRepresentatives(vectors, targetGoldenVectorCount)
	if err != nil {
		return err
	}

	hashes := vocab.Hashes()
	modelManifest := cudamodel.Manifest{
		Format:                   cudamodel.Format,
		Endianness:               "little",
		DType:                    "float32",
		RunID:                    configuration.runID,
		Checkpoint:               configuration.checkpoint,
		CheckpointUpdate:         int(loaded.state.GlobalUpdate),
		TrainingCommit:           manifest.Repositories.MachineLearning.Commit,
		NumSolutions:             cudamodel.NumSolutions,
		NumActions:               cudamodel.NumActions,
		CandidateStatsSize:       cudamodel.CandidateStatsSize,
		NumTurns:                 cudamodel.NumTurns,
		TrunkSize:                cudamodel.TrunkSize,
		ParameterCount:           cudamodel.ParameterCount,
		WeightsFile:              cudamodel.WeightsFilename,
		SolutionVocabularySHA256: hashes.Solutions,
		ActionVocabularySHA256:   hashes.Actions,
		Tensors:                  cudamodel.ExpectedTensors(),
	}
	expectedHashes := cudamodel.VocabularyHashes{Solutions: hashes.Solutions, Actions: hashes.Actions}

	if err := cudaref.PublishDirectory(configuration.output, func(staging string) error {
		weightsSHA, err := cudaref.WriteFloat32LE(filepath.Join(staging, cudamodel.WeightsFilename), weights)
		if err != nil {
			return fmt.Errorf("write exported weights: %w", err)
		}
		modelManifest.WeightsSHA256 = weightsSHA
		if err := cudamodel.ValidateManifest(modelManifest, expectedHashes); err != nil {
			return fmt.Errorf("validate constructed model manifest: %w", err)
		}
		if err := cudaref.WriteJSON(filepath.Join(staging, "manifest.json"), modelManifest); err != nil {
			return fmt.Errorf("write model manifest: %w", err)
		}
		portable, err := cudamodel.Load(staging, expectedHashes)
		if err != nil {
			return fmt.Errorf("load just-exported portable model: %w", err)
		}
		comparison, err := comparePortable(vectors, portable)
		if err != nil {
			return err
		}
		if comparison.MaximumAbsolute > portableMaximumAbsoluteError {
			return fmt.Errorf("portable evaluator maximum absolute logit error %.9g exceeds strict tolerance %.9g", comparison.MaximumAbsolute, portableMaximumAbsoluteError)
		}
		if comparison.Top1Agreement != len(vectors) {
			return fmt.Errorf("portable evaluator raw top-action agreement is %d/%d", comparison.Top1Agreement, len(vectors))
		}
		if _, err := cudaref.WriteGoldenVectors(staging, vectors); err != nil {
			return fmt.Errorf("write golden vectors: %w", err)
		}
		games.RunID = configuration.runID
		games.Checkpoint = configuration.checkpoint
		games.CheckpointUpdate = loaded.state.GlobalUpdate
		if err := cudaref.WriteJSON(filepath.Join(staging, cudaref.GoldenGamesFilename), games); err != nil {
			return fmt.Errorf("write golden games: %w", err)
		}
		report := cudaref.ExportReport{
			Format:             cudamodel.Format,
			RunID:              configuration.runID,
			Checkpoint:         configuration.checkpoint,
			CheckpointUpdate:   loaded.state.GlobalUpdate,
			CheckpointIdentity: loaded.checkpointIdentity,
			TrainingCommit:     manifest.Repositories.MachineLearning.Commit,
			ParameterCount:     cudamodel.ParameterCount,
			WeightsSHA256:      weightsSHA,
			GoldenVectorCount:  len(vectors),
			GoldenGames:        cudaref.GoldenGamesFilename,
			PortableComparison: comparison,
		}
		if err := cudaref.WriteJSON(filepath.Join(staging, cudaref.ExportReportFilename), report); err != nil {
			return fmt.Errorf("write export report: %w", err)
		}
		return nil
	}); err != nil {
		return err
	}
	fmt.Fprintf(stdout, "exported run=%s checkpoint=%s update=%d vectors=%d output=%s\n", configuration.runID, configuration.checkpoint, loaded.state.GlobalUpdate, len(vectors), configuration.output)
	return nil
}

func parseConfig(args []string, stderr io.Writer) (config, error) {
	dataDir := os.Getenv("WORDLEML_DATA_DIR")
	if dataDir == "" {
		dataDir = "../data"
	}
	runsDir := os.Getenv("WORDLEML_RUNS_DIR")
	if runsDir == "" {
		runsDir = filepath.Join(filepath.Dir(dataDir), "runs")
	}
	flags := flag.NewFlagSet("exportcuda", flag.ContinueOnError)
	flags.SetOutput(stderr)
	flags.Usage = func() {
		fmt.Fprintln(stderr, "usage: exportcuda -run-id=<completed-run-id> [-checkpoint=best] [flags]")
		flags.PrintDefaults()
	}
	var result config
	flags.StringVar(&result.dataDir, "data-dir", dataDir, "directory containing frozen vocabulary and validation data")
	flags.StringVar(&result.runsDir, "runs-dir", runsDir, "directory containing completed runs")
	flags.StringVar(&result.runID, "run-id", "", "required completed run identifier")
	flags.StringVar(&result.checkpoint, "checkpoint", defaultCheckpoint, "checkpoint selector (currently best only)")
	flags.StringVar(&result.output, "output", "", "export destination (default runs/<run>/exports/cuda-f32-v1/<checkpoint>)")
	if err := flags.Parse(args); err != nil {
		return config{}, err
	}
	if flags.NArg() != 0 {
		return config{}, fmt.Errorf("unexpected argument %q", flags.Arg(0))
	}
	if err := runstate.ValidateRunID(result.runID); err != nil {
		return config{}, err
	}
	if result.checkpoint != defaultCheckpoint {
		return config{}, fmt.Errorf("-checkpoint=%q is unsupported; only %q is currently exportable", result.checkpoint, defaultCheckpoint)
	}
	if strings.TrimSpace(result.dataDir) == "" || strings.TrimSpace(result.runsDir) == "" {
		return config{}, errors.New("-data-dir and -runs-dir must not be empty")
	}
	if result.output == "" {
		result.output = filepath.Join(result.runsDir, result.runID, "exports", exportDirectoryFormat, result.checkpoint)
	}
	if strings.TrimSpace(result.output) == "" {
		return config{}, errors.New("-output must not be empty")
	}
	return result, nil
}

func readAndValidateRunManifest(layout runstate.Layout, dataDir string, trainingConfig proofrun.Config) (runmetadata.Manifest, error) {
	contents, err := os.ReadFile(layout.MetadataPath)
	if err != nil {
		return runmetadata.Manifest{}, fmt.Errorf("read run metadata: %w", err)
	}
	var manifest runmetadata.Manifest
	if err := json.Unmarshal(contents, &manifest); err != nil {
		return runmetadata.Manifest{}, fmt.Errorf("decode run metadata: %w", err)
	}
	if err := runmetadata.VerifyEvaluationInputs(manifest, dataDir); err != nil {
		return runmetadata.Manifest{}, fmt.Errorf("validate run vocabulary and validation inputs: %w", err)
	}
	if manifest.FinalTestSealed != true {
		return runmetadata.Manifest{}, errors.New("CUDA export requires a run whose final test remained sealed")
	}
	if manifest.ModelParameterCount != cudamodel.ParameterCount {
		return runmetadata.Manifest{}, fmt.Errorf("run metadata model parameter count %d, want %d", manifest.ModelParameterCount, cudamodel.ParameterCount)
	}
	if manifest.Runtime.Backend != "xla:cuda" {
		return runmetadata.Manifest{}, fmt.Errorf("run metadata backend %q, want xla:cuda", manifest.Runtime.Backend)
	}
	var effective proofrun.Config
	if err := json.Unmarshal(manifest.EffectiveConfig, &effective); err != nil {
		return runmetadata.Manifest{}, fmt.Errorf("decode effective configuration: %w", err)
	}
	if effective != trainingConfig {
		return runmetadata.Manifest{}, errors.New("metadata effective configuration differs from immutable run config")
	}
	return manifest, nil
}

func readAndValidateFinalMetrics(layout runstate.Layout, trainingConfig proofrun.Config) (proofrun.Result, error) {
	contents, err := os.ReadFile(layout.FinalMetricsPath)
	if err != nil {
		return proofrun.Result{}, fmt.Errorf("read final metrics: %w", err)
	}
	var final proofrun.Result
	if err := json.Unmarshal(contents, &final); err != nil {
		return proofrun.Result{}, fmt.Errorf("decode final metrics: %w", err)
	}
	if !proofrun.IsProductionStyle(trainingConfig.Stage) || final.Stage != trainingConfig.Stage || !final.Passed || final.GlobalUpdate != trainingConfig.TargetUpdates {
		return proofrun.Result{}, errors.New("run has not completed a passed production-style immutable training configuration")
	}
	for _, check := range []struct {
		name    string
		metrics proofrun.Metrics
	}{
		{"initial training", final.InitialTraining},
		{"final training", final.FinalTraining},
		{"initial validation", final.InitialValidation},
		{"final validation", final.FinalValidation},
		{"best validation", final.BestValidation},
	} {
		if !validMetrics(check.metrics) {
			return proofrun.Result{}, fmt.Errorf("final metrics have invalid %s metrics", check.name)
		}
	}
	if final.BestValidationStep < 0 || final.BestValidationStep > final.GlobalUpdate {
		return proofrun.Result{}, errors.New("final metrics have an invalid best checkpoint update")
	}
	for _, check := range []struct {
		name     string
		snapshot proofrun.ValidationSnapshot
		metrics  proofrun.Metrics
		update   int64
	}{
		{"initial", final.InitialValidationDetails, final.InitialValidation, 0},
		{"final", final.FinalValidationDetails, final.FinalValidation, final.GlobalUpdate},
		{"best", final.BestValidationDetails, final.BestValidation, final.BestValidationStep},
	} {
		if check.snapshot.Update != check.update || check.snapshot.Metrics != check.metrics || !validMetrics(check.snapshot.Metrics) {
			return proofrun.Result{}, fmt.Errorf("final metrics %s validation snapshot does not match its top-level identity", check.name)
		}
		details := proofrun.Metrics{
			Loss:  check.snapshot.Details.Loss,
			Top1:  check.snapshot.Details.Top1Accuracy,
			Top5:  check.snapshot.Details.Top5Accuracy,
			Top16: check.snapshot.Details.Top16Accuracy,
		}
		if check.snapshot.Details.Examples <= 0 || details != check.metrics || !validMetrics(details) {
			return proofrun.Result{}, fmt.Errorf("final metrics %s validation details do not match their snapshot", check.name)
		}
	}
	wantSnapshots := int(trainingConfig.TargetUpdates/trainingConfig.ValidationEvery) + 1
	if len(final.ValidationSnapshots) != wantSnapshots {
		return proofrun.Result{}, fmt.Errorf("final metrics contain %d validation snapshots, want %d", len(final.ValidationSnapshots), wantSnapshots)
	}
	for index, snapshot := range final.ValidationSnapshots {
		wantUpdate := int64(index) * trainingConfig.ValidationEvery
		if snapshot.Update != wantUpdate || !validMetrics(snapshot.Metrics) {
			return proofrun.Result{}, fmt.Errorf("final metrics snapshot %d is not a finite update-%d validation result", index, wantUpdate)
		}
	}
	if final.ProductionSafety == nil || !final.ProductionSafety.LossFinite || !final.ProductionSafety.GradientsFinite || !final.ProductionSafety.ParametersFinite || final.ProductionSafety.UpdatesChecked != trainingConfig.TargetUpdates {
		return proofrun.Result{}, errors.New("final metrics lack complete production safety evidence")
	}
	return final, nil
}

func validMetrics(metrics proofrun.Metrics) bool {
	return metrics.Finite() &&
		metrics.Top1 >= 0 && metrics.Top1 <= 1 &&
		metrics.Top5 >= metrics.Top1 && metrics.Top5 <= 1 &&
		metrics.Top16 >= metrics.Top5 && metrics.Top16 <= 1
}

func restoreCheckpoint(backend compute.Backend, layout runstate.Layout, trainingConfig proofrun.Config, selector string, final proofrun.Result) (loadedCheckpoint, error) {
	if selector != defaultCheckpoint {
		return loadedCheckpoint{}, fmt.Errorf("unsupported checkpoint selector %q", selector)
	}
	session, err := supervised.New(supervised.Config{
		Policy:       policyConfig(),
		LearningRate: trainingConfig.LearningRate,
		Seed:         trainingConfig.Seed,
	}, backend)
	if err != nil {
		return loadedCheckpoint{}, fmt.Errorf("create fresh exporter session: %w", err)
	}
	loaded := loadedCheckpoint{session: session}
	keep := false
	defer func() {
		if !keep {
			session.Finalize()
		}
	}()
	handler, err := supervised.NewCheckpoint(session.Store, layout.BestCheckpointDir)
	if err != nil {
		return loadedCheckpoint{}, fmt.Errorf("restore best checkpoint: %w", err)
	}
	if err := validateCheckpointVariableTable(handler); err != nil {
		return loadedCheckpoint{}, err
	}
	identities, err := handler.ListCheckpoints()
	if err != nil || len(identities) == 0 {
		if err != nil {
			return loadedCheckpoint{}, fmt.Errorf("list best checkpoint identities: %w", err)
		}
		return loadedCheckpoint{}, errors.New("best checkpoint selector has no checkpoint identities")
	}
	loaded.checkpointIdentity = identities[len(identities)-1]
	state, err := runstate.LoadCheckpointState(session.Store)
	if err != nil {
		return loadedCheckpoint{}, fmt.Errorf("read restored checkpoint state: %w", err)
	}
	if err := proofrun.ValidateResumeState(trainingConfig, state); err != nil {
		return loadedCheckpoint{}, fmt.Errorf("validate restored checkpoint state: %w", err)
	}
	if session.Trainer.GlobalStep() != state.GlobalUpdate {
		return loadedCheckpoint{}, fmt.Errorf("restored global step %d differs from checkpoint state %d", session.Trainer.GlobalStep(), state.GlobalUpdate)
	}
	if state.GlobalUpdate != final.BestValidationStep || state.BestValidation == nil || state.BestValidation.Update != state.GlobalUpdate || state.BestValidation.Value != final.BestValidation.Loss {
		return loadedCheckpoint{}, errors.New("restored best checkpoint differs from final-metrics best checkpoint identity")
	}
	if !strings.Contains(loaded.checkpointIdentity, fmt.Sprintf("step-%08d", state.GlobalUpdate)) {
		return loadedCheckpoint{}, fmt.Errorf("selected checkpoint identity %q does not name restored update %d", loaded.checkpointIdentity, state.GlobalUpdate)
	}
	loaded.state = state
	keep = true
	return loaded, nil
}

func policyConfig() policy.Config {
	return policy.Config{NumSolutions: cudamodel.NumSolutions, NumActions: cudamodel.NumActions}
}

func validateCheckpointVariableTable(handler *checkpoint.Handler) error {
	if handler == nil {
		return errors.New("checkpoint handler is nil")
	}
	loaded := handler.LoadedVariables()
	expected := make(map[string]cudamodel.Tensor, len(cudamodel.ExpectedTensors()))
	for _, tensor := range cudamodel.ExpectedTensors() {
		expected[tensor.SourceName] = tensor
		if _, found := loaded[tensor.SourceName]; !found {
			return fmt.Errorf("checkpoint lacks expected trainable variable %q", tensor.SourceName)
		}
	}
	for sourceName := range loaded {
		if strings.HasPrefix(sourceName, "var:/wordle_policy/") {
			if _, known := expected[sourceName]; !known {
				return fmt.Errorf("checkpoint has unexpected policy variable %q", sourceName)
			}
		}
	}
	return nil
}

func materializePolicy(session *supervised.Session, vocab *vocabulary.Vocabulary) error {
	inputs, err := openingInputs(vocab)
	if err != nil {
		return err
	}
	available := make([]float32, cudamodel.NumActions)
	for action := range available {
		available[action] = 1
	}
	raw, masked, beta, err := session.PredictDiagnostics(
		[][]float32{inputs.CandidateMask}, [][]float32{inputs.CandidateStats}, []int32{inputs.Turn},
		[][]float32{inputs.RemainingActionMask}, [][]float32{available},
	)
	if err != nil {
		return fmt.Errorf("materialize checkpoint policy: %w", err)
	}
	defer func() { _ = raw.FinalizeAll(); _ = masked.FinalizeAll(); _ = beta.FinalizeAll() }()
	if _, err := tensors.CopyFlatData[float32](raw); err != nil {
		return fmt.Errorf("read materialized opening logits: %w", err)
	}
	return nil
}

func openingInputs(vocab *vocabulary.Vocabulary) (modelstate.Inputs, error) {
	encoder, err := modelstate.NewEncoder(vocab)
	if err != nil {
		return modelstate.Inputs{}, err
	}
	bits := make([]byte, modelstate.CandidateBitsetBytes)
	for index := range bits {
		bits[index] = 0xff
	}
	bits[len(bits)-1] = (1 << (vocabulary.NumSolutions % 8)) - 1
	return encoder.Encode(bits, 0)
}

func exportTrainableWeights(session *supervised.Session) ([]float32, error) {
	if session == nil || session.Store == nil {
		return nil, errors.New("export session Store is nil")
	}
	expected := cudamodel.ExpectedTensors()
	variables := make(map[string]*model.Variable, len(expected))
	parameterCount := 0
	for variable := range session.Store.IterVariables() {
		if !variable.Trainable {
			continue
		}
		parameterCount += variable.Shape().Size()
		source := "var:" + variable.Path()
		if _, exists := variables[source]; exists {
			return nil, fmt.Errorf("trainable variable %q appears more than once", source)
		}
		variables[source] = variable
	}
	if len(variables) != len(expected) || parameterCount != cudamodel.ParameterCount {
		return nil, fmt.Errorf("materialized trainable variables=%d parameters=%d, want variables=%d parameters=%d", len(variables), parameterCount, len(expected), cudamodel.ParameterCount)
	}
	weights := make([]float32, cudamodel.ParameterCount)
	for _, tensor := range expected {
		variable, found := variables[tensor.SourceName]
		if !found {
			return nil, fmt.Errorf("materialized trainable variable %q is absent", tensor.SourceName)
		}
		if err := validateSourceShape(variable.Shape().Dimensions, tensor); err != nil {
			return nil, fmt.Errorf("variable %q: %w", tensor.SourceName, err)
		}
		values, err := tensors.CopyFlatData[float32](variable.MustValue())
		if err != nil {
			return nil, fmt.Errorf("copy variable %q: %w", tensor.SourceName, err)
		}
		if len(values) != tensor.Count {
			return nil, fmt.Errorf("variable %q has %d values, want %d", tensor.SourceName, len(values), tensor.Count)
		}
		destination := weights[tensor.Offset : tensor.Offset+tensor.Count]
		if denseSourceIsInputMajor(tensor) {
			outputs, inputs := tensor.Shape[0], tensor.Shape[1]
			for output := range outputs {
				for input := range inputs {
					destination[output*inputs+input] = values[input*outputs+output]
				}
			}
		} else {
			copy(destination, values)
		}
	}
	for sourceName := range variables {
		found := false
		for _, tensor := range expected {
			if sourceName == tensor.SourceName {
				found = true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("unexpected materialized trainable variable %q", sourceName)
		}
	}
	return weights, nil
}

func validateSourceShape(shape []int, tensor cudamodel.Tensor) error {
	want := tensor.Shape
	if denseSourceIsInputMajor(tensor) {
		want = []int{want[1], want[0]}
	}
	if !slices.Equal(shape, want) {
		return fmt.Errorf("shape %v, want source shape %v for exported shape %v", shape, want, tensor.Shape)
	}
	return nil
}

// denseSourceIsInputMajor identifies the four Dense kernel tensors whose
// GoMLX checkpoint layout is [inputs, outputs]. Embeddings are also rank two,
// but their [turn, feature] table is already the portable output-major layout
// and must be copied unchanged.
func denseSourceIsInputMajor(tensor cudamodel.Tensor) bool {
	return strings.HasSuffix(tensor.Name, ".weight")
}

func generateReferences(ctx context.Context, session *supervised.Session, vocab *vocabulary.Vocabulary) ([]cudaref.Vector, cudaref.Games, error) {
	if session == nil || vocab == nil {
		return nil, cudaref.Games{}, errors.New("session and vocabulary are required")
	}
	captures := make([]capturedPosition, 0, vocabulary.NumValidationSolutions*4)
	score := func(_ context.Context, position gameeval.Position) ([]float32, error) {
		rawTensor, maskedTensor, betaTensor, err := session.PredictDiagnostics(
			[][]float32{position.Inputs.CandidateMask}, [][]float32{position.Inputs.CandidateStats}, []int32{position.Inputs.Turn},
			[][]float32{position.Inputs.RemainingActionMask}, [][]float32{position.AvailableActionMask},
		)
		if err != nil {
			return nil, err
		}
		defer func() { _ = rawTensor.FinalizeAll(); _ = maskedTensor.FinalizeAll(); _ = betaTensor.FinalizeAll() }()
		raw, err := tensors.CopyFlatData[float32](rawTensor)
		if err != nil {
			return nil, err
		}
		if _, err := cudaref.RawTopAction(raw); err != nil {
			return nil, err
		}
		captures = append(captures, capturedPosition{position: clonePosition(position), logits: append([]float32(nil), raw...)})
		return raw, nil
	}
	evaluator, err := gameeval.New(gameeval.Config{Vocabulary: vocab, Score: score})
	if err != nil {
		return nil, cudaref.Games{}, err
	}
	evaluation, err := evaluator.Evaluate(ctx, vocab.Validation())
	if err != nil {
		return nil, cudaref.Games{}, fmt.Errorf("regenerate fixed validation game references: %w", err)
	}
	vectors, err := vectorsFromCaptures(captures, evaluation, vocab)
	if err != nil {
		return nil, cudaref.Games{}, err
	}
	return vectors, cudaref.Games{Evaluation: evaluation}, nil
}

func clonePosition(position gameeval.Position) gameeval.Position {
	inputs := position.Inputs
	inputs.CandidateMask = append([]float32(nil), inputs.CandidateMask...)
	inputs.CandidateStats = append([]float32(nil), inputs.CandidateStats...)
	inputs.RemainingActionMask = append([]float32(nil), inputs.RemainingActionMask...)
	return gameeval.Position{
		Inputs:              inputs,
		AvailableActionMask: append([]float32(nil), position.AvailableActionMask...),
		Turn:                position.Turn,
		CandidateSolutions:  append([]string(nil), position.CandidateSolutions...),
		History:             append([]gameeval.HistoryTurn(nil), position.History...),
	}
}

func vectorsFromCaptures(captures []capturedPosition, evaluation gameeval.Evaluation, vocab *vocabulary.Vocabulary) ([]cudaref.Vector, error) {
	result := make([]cudaref.Vector, 0, len(captures))
	captureIndex := 0
	for gameIndex, game := range evaluation.Games {
		for turnIndex, turn := range game.Turns {
			if captureIndex >= len(captures) {
				return nil, errors.New("game evaluation has more turns than captured GoMLX calls")
			}
			capture := captures[captureIndex]
			captureIndex++
			if capture.position.Turn != turn.Turn-1 {
				return nil, fmt.Errorf("game %d turn %d capture turn=%d", gameIndex, turnIndex, capture.position.Turn)
			}
			rawTop, selected, err := cudaref.SelectAvailableAction(capture.logits, capture.position.AvailableActionMask)
			if err != nil {
				return nil, err
			}
			selectedFromGame, found := vocab.ActionID(turn.Guess)
			if !found || rawTop != turn.RawTopActionID || selected != selectedFromGame {
				return nil, fmt.Errorf("game %d turn %d selection differs from captured reference", gameIndex, turnIndex)
			}
			margin, err := cudaref.TopTwoMargin(capture.logits)
			if err != nil {
				return nil, err
			}
			selectedWasCandidate := capture.position.Inputs.RemainingActionMask[selected] != 0
			selectionKind := "probe"
			if selectedWasCandidate {
				selectionKind = "candidate"
			}
			result = append(result, cudaref.Vector{
				ID:                  fmt.Sprintf("validation-%03d-turn-%d", gameIndex, capture.position.Turn),
				Inputs:              capture.position.Inputs,
				AvailableActionMask: capture.position.AvailableActionMask,
				RawLogits:           capture.logits,
				RawTopActionID:      rawTop,
				SelectedActionID:    selected,
				TopTwoMargin:        margin,
				Provenance: cudaref.Provenance{
					Solution:             game.Solution,
					Turn:                 capture.position.Turn,
					CandidateCount:       len(capture.position.CandidateSolutions),
					History:              append([]gameeval.HistoryTurn(nil), capture.position.History...),
					RawTopWasAvailable:   capture.position.AvailableActionMask[rawTop] != 0,
					SelectedWasCandidate: selectedWasCandidate,
					SelectionKind:        selectionKind,
					ShortlistSizeBefore:  turn.ShortlistSizeBefore,
					ShortlistSizeAfter:   turn.ShortlistSizeAfter,
				},
			})
		}
	}
	if captureIndex != len(captures) {
		return nil, fmt.Errorf("captured %d GoMLX calls but consumed %d game turns", len(captures), captureIndex)
	}
	return result, nil
}

func chooseRepresentatives(all []cudaref.Vector, target int) ([]cudaref.Vector, error) {
	if target <= 0 || len(all) < target {
		return nil, fmt.Errorf("need at least %d captured positions, got %d", target, len(all))
	}
	selected := make([]cudaref.Vector, 0, target)
	used := make([]bool, len(all))
	addFirst := func(label string, predicate func(cudaref.Vector) bool) error {
		for index, vector := range all {
			if !used[index] && predicate(vector) {
				used[index] = true
				selected = append(selected, vector)
				return nil
			}
		}
		return fmt.Errorf("validation references have no %s position", label)
	}
	if err := addFirst("opening", func(vector cudaref.Vector) bool {
		return vector.Provenance.Turn == 0 && vector.Provenance.CandidateCount == vocabulary.NumSolutions
	}); err != nil {
		return nil, err
	}
	for turn := 0; turn < cudamodel.NumTurns; turn++ {
		current := turn
		if err := addFirst(fmt.Sprintf("turn-%d", turn), func(vector cudaref.Vector) bool { return vector.Provenance.Turn == current }); err != nil {
			return nil, err
		}
	}
	for _, bucket := range []struct {
		name  string
		match func(int) bool
	}{
		{"large-candidate-set", func(count int) bool { return count > 100 }},
		{"medium-candidate-set", func(count int) bool { return count >= 6 && count <= 100 }},
		{"small-candidate-set", func(count int) bool { return count <= 5 }},
	} {
		if err := addFirst(bucket.name, func(vector cudaref.Vector) bool { return bucket.match(vector.Provenance.CandidateCount) }); err != nil {
			return nil, err
		}
	}
	if err := addFirst("selected-candidate", func(vector cudaref.Vector) bool { return vector.Provenance.SelectedWasCandidate }); err != nil {
		return nil, err
	}
	if err := addFirst("selected-probe", func(vector cudaref.Vector) bool { return !vector.Provenance.SelectedWasCandidate }); err != nil {
		return nil, err
	}
	for index, vector := range all {
		if len(selected) == target {
			break
		}
		if !used[index] {
			used[index] = true
			selected = append(selected, vector)
		}
	}
	if len(selected) != target {
		return nil, fmt.Errorf("selected %d representative positions, want %d", len(selected), target)
	}
	return selected, nil
}

func comparePortable(vectors []cudaref.Vector, portable *cudamodel.Model) (cudaref.Comparison, error) {
	comparison, err := cudaref.CompareLogits(vectors, func(vector cudaref.Vector) ([]float32, error) {
		logits, err := portable.Forward(vector.Inputs)
		if err != nil {
			return nil, err
		}
		_, selected, err := cudaref.SelectAvailableAction(logits, vector.AvailableActionMask)
		if err != nil {
			return nil, err
		}
		if selected != vector.SelectedActionID {
			return nil, fmt.Errorf("vector %q selected action %d, want %d", vector.ID, selected, vector.SelectedActionID)
		}
		return logits, nil
	})
	if err != nil {
		return cudaref.Comparison{}, fmt.Errorf("compare GoMLX and portable evaluator: %w", err)
	}
	return comparison, nil
}
