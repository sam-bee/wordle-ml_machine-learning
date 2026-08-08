// Package supervised builds the small supervised-training wrapper around the
// Wordle policy. The policy remains responsible only for its four model inputs;
// action availability is a training-time concern.
package supervised

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/metric"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/sam-bee/wordle-ml_machine-learning/policy"
)

const (
	// GlobalGradientClipNorm is the fixed proof-run clipping threshold. It is a
	// global L2 norm applied before Adam updates its moments, not a per-value
	// clipping threshold on the eventual update.
	GlobalGradientClipNorm = 5.0

	diagnosticsScope = "/supervised_diagnostics"
	// gradientNormsScope mirrors each exact trainable-variable path beneath a
	// dedicated diagnostics prefix. This makes checkpointed norms easy for
	// telemetry to discover without coupling it to policy layer names.
	gradientNormsScope = diagnosticsScope + "/gradient_norms"

	preclipGradientNormName   = "preclip_global_gradient_norm"
	appliedGradientNormName   = "applied_global_gradient_norm"
	gradientFiniteName        = "gradients_finite"
	parameterFiniteName       = "parameters_finite"
	parameterNormName         = "parameter_norm"
	updateToParameterNormName = "update_to_parameter_norm"
	learningRateName          = "learning_rate"
)

// Config contains the fixed policy dimensions and the few supervised-training
// choices needed for an experiment.
type Config struct {
	Policy       policy.Config
	LearningRate float64
	Seed         int64
}

// TrainingDiagnostics are updated by each optimizer step. They are also held
// as non-trainable Store variables so checkpoints retain the last observed
// proof-run diagnostics along with the optimizer state.
type TrainingDiagnostics struct {
	PreclipGlobalGradientNorm float32
	AppliedGlobalGradientNorm float32
	GradientsFinite           bool
	ParametersFinite          bool
	ParameterNorm             float32
	UpdateToParameterNorm     float32
	LearningRate              float32
}

// NamedNorm identifies a pre-clipping gradient L2 norm by its exact trainable
// Store path, for example "/wordle_policy/base_logits/dense/weights".
type NamedNorm struct {
	Path string
	Norm float32
}

// Session owns one policy, its parameter store, trainer, and a separate
// inference executor. Inference reads the same Store but never applies an
// optimizer update.
type Session struct {
	Policy    *policy.Model
	Store     *model.Store
	Trainer   *train.Trainer
	Inference *model.Exec
}

// New constructs a deterministic FP32 Adam training session with fixed
// global-gradient-norm clipping.
func New(config Config, backend compute.Backend) (*Session, error) {
	if config.LearningRate <= 0 || math.IsNaN(config.LearningRate) || math.IsInf(config.LearningRate, 0) {
		return nil, fmt.Errorf("learning rate must be positive, got %g", config.LearningRate)
	}
	if config.Seed == 0 {
		return nil, fmt.Errorf("seed must be non-zero")
	}
	if config.Policy.NumActions < 16 {
		return nil, fmt.Errorf("number of actions must be at least 16 for top-16 accuracy, got %d", config.Policy.NumActions)
	}
	if backend == nil {
		return nil, fmt.Errorf("backend must not be nil")
	}
	policyModel, err := policy.New(config.Policy)
	if err != nil {
		return nil, err
	}

	store := model.NewStore()
	store.SetParam(model.ParamInitialSeed, config.Seed)
	session := &Session{Policy: policyModel, Store: store}
	session.Trainer = train.NewTrainer(
		backend,
		store,
		session.Forward,
		maskedSparseCategoricalCrossEntropy,
		NewAdamWithGlobalNormClip(config.LearningRate, GlobalGradientClipNorm),
		trainMetrics(),
		evalMetrics(),
	)
	session.Inference, err = model.NewExec(backend, store, session.InferenceGraph)
	if err != nil {
		session.Finalize()
		return nil, fmt.Errorf("create inference executor: %w", err)
	}
	return session, nil
}

// Forward consumes the five tensors provided by imitationdata: candidate mask,
// candidate statistics, turn, remaining-solution action mask, and availability
// mask. The availability mask is deliberately applied after policy.Forward so it
// cannot change the policy architecture or suppress valid probe words.
func (s *Session) Forward(scope *model.Scope, inputs []*graph.Node) *graph.Node {
	logits, _ := s.forwardWithBeta(scope, inputs)
	return logits
}

// InferenceGraph is the no-update inference graph used by Inference. It returns
// finite raw logits for game evaluation and logit diagnostics, hard-masked
// logits for action selection, and the learned candidate bonus (beta).
func (s *Session) InferenceGraph(
	scope *model.Scope,
	candidateMask, candidateStats, turn, remainingActionMask, availableActions *graph.Node,
) (*graph.Node, *graph.Node, *graph.Node) {
	scope.Store().SetTraining(candidateMask.Graph(), false)
	rawLogits, beta := s.Policy.ForwardWithBeta(scope, candidateMask, candidateStats, turn, remainingActionMask)
	maskedLogits := ApplyAvailabilityMask(rawLogits, availableActions)
	return rawLogits, maskedLogits, beta
}

// PredictDiagnostics executes no-update inference and returns raw logits,
// hard-masked logits, and beta in that order. The caller owns all returned
// tensors and must finalize them when finished.
func (s *Session) PredictDiagnostics(inputs ...any) (rawLogits, maskedLogits, beta *tensors.Tensor, err error) {
	if s == nil || s.Inference == nil {
		return nil, nil, nil, fmt.Errorf("supervised session has no inference executor")
	}
	return s.Inference.Call3(inputs...)
}

// Predict executes the no-update inference executor and returns only the
// hard-masked logits plus beta for callers selecting an action. Use
// PredictDiagnostics when raw logits are needed for evaluation or telemetry.
// The caller owns the returned tensors and must finalize them when finished.
func (s *Session) Predict(inputs ...any) (logits, beta *tensors.Tensor, err error) {
	rawLogits, logits, beta, err := s.PredictDiagnostics(inputs...)
	if rawLogits != nil {
		_ = rawLogits.FinalizeAll()
	}
	return logits, beta, err
}

func (s *Session) forwardWithBeta(scope *model.Scope, inputs []*graph.Node) (*graph.Node, *graph.Node) {
	if len(inputs) != 5 {
		panic(fmt.Sprintf("supervised model needs 5 inputs, got %d", len(inputs)))
	}
	logits, beta := s.Policy.ForwardWithBeta(scope, inputs[0], inputs[1], inputs[2], inputs[3])
	return ApplyAvailabilityMask(logits, inputs[4]), beta
}

// Finalize releases every resource owned by the Session. It is safe to call
// more than once; a finalized Session cannot be used again.
func (s *Session) Finalize() {
	if s == nil {
		return
	}
	if s.Inference != nil {
		s.Inference.Finalize()
		s.Inference = nil
	}
	if s.Trainer != nil {
		// Trainer caches compiled train/eval executors independently of Store.
		s.Trainer.ResetComputationGraphs()
		s.Trainer = nil
	}
	if s.Store != nil {
		s.Store.Finalize()
		s.Store = nil
	}
	s.Policy = nil
}

// ApplyAvailabilityMask makes used actions impossible to select. A finite
// penalty is not enough: a sufficiently extreme raw logit could still win. The
// mask therefore uses Where with negative infinity, preserving legal logits
// exactly and making all unavailable logits unselectable by ArgMax/TopK.
func ApplyAvailabilityMask(logits, availableActions *graph.Node) *graph.Node {
	if err := logits.Shape().Check(dtypes.Float32, -1, -1); err != nil {
		panic(fmt.Errorf("logits: %w", err))
	}
	if err := availableActions.Shape().Check(dtypes.Float32, logits.Shape().Dimensions...); err != nil {
		panic(fmt.Errorf("available actions: %w", err))
	}
	available := graph.GreaterThan(availableActions, graph.ScalarZero(availableActions.Graph(), dtypes.Float32))
	return graph.Where(available, logits, graph.Infinity(logits.Graph(), logits.DType(), -1))
}

// maskedSparseCategoricalCrossEntropy deliberately consumes only labels[0].
// MaterializeBatch also supplies labels[1] with the teacher top-16 ranking for
// agreement metrics; it is neither a loss weight nor a second target.
//
// Hard masking uses -Inf. GoMLX's stock sparse cross-entropy expands targets to
// one-hot form, where 0 * -Inf can produce NaN. Only that expected negative
// infinity is replaced with a very small finite value for the loss: unexpected
// NaN or +Inf model outputs must remain visible to the training safety gates.
func maskedSparseCategoricalCrossEntropy(labels, predictions []*graph.Node) *graph.Node {
	if len(labels) == 0 {
		panic("masked sparse categorical cross-entropy needs labels[0]")
	}
	if len(predictions) == 0 {
		panic("masked sparse categorical cross-entropy needs logits")
	}
	logits := predictions[0]
	if !logits.DType().IsFloat() {
		panic(fmt.Sprintf("logits dtype must be floating point, got %s", logits.DType()))
	}
	lossLogits := graph.Where(
		graph.Equal(logits, graph.Infinity(logits.Graph(), logits.DType(), -1)),
		graph.Scalar(logits.Graph(), logits.DType(), -1e30),
		logits,
	)
	return loss.SparseCategoricalCrossEntropyLogits(
		[]*graph.Node{labels[0]},
		[]*graph.Node{lossLogits},
	)
}

// NewCheckpoint builds a checkpoint handler that retains the three newest
// checkpoints and resumes the newest one when it exists. The Store contains the
// model, Adam moments/counters, RNG state, global step, and diagnostics, all of
// which the normal GoMLX handler persists.
func NewCheckpoint(store *model.Store, dir string) (*checkpoint.Handler, error) {
	return checkpoint.Build(store).Dir(dir).Keep(3).Done()
}

func trainMetrics() []metric.Interface {
	return []metric.Interface{
		metric.NewExponentialMovingAverageMetric("Top-1 Accuracy", "top1", metric.AccuracyMetricType, topKAccuracyGraph(1), nil, 0.01),
		metric.NewExponentialMovingAverageMetric("Top-5 Accuracy", "top5", metric.AccuracyMetricType, topKAccuracyGraph(5), nil, 0.01),
		metric.NewExponentialMovingAverageMetric("Top-16 Accuracy", "top16", metric.AccuracyMetricType, topKAccuracyGraph(16), nil, 0.01),
	}
}

func evalMetrics() []metric.Interface {
	return []metric.Interface{
		metric.NewMeanMetric("Top-1 Accuracy", "top1", metric.AccuracyMetricType, topKAccuracyGraph(1), nil),
		metric.NewMeanMetric("Top-5 Accuracy", "top5", metric.AccuracyMetricType, topKAccuracyGraph(5), nil),
		metric.NewMeanMetric("Top-16 Accuracy", "top16", metric.AccuracyMetricType, topKAccuracyGraph(16), nil),
	}
}

// topKAccuracyGraph measures whether the model's single highest-ranked action
// appears anywhere in the teacher's ranked top-k list. Legacy one-label test
// batches have no ranking tensor, so they retain the previous top-1-label
// recall@k calculation as a compatibility fallback.
func topKAccuracyGraph(k int) metric.BaseMetricGraph {
	return func(_ *model.Scope, labels, predictions []*graph.Node) *graph.Node {
		logits := predictions[0]
		if len(labels) >= 2 {
			teacherRanking := labels[1]
			if !teacherRanking.DType().IsInt() {
				panic(fmt.Sprintf("teacher ranking dtype must be integer, got %s", teacherRanking.DType()))
			}
			if err := teacherRanking.Shape().Check(teacherRanking.DType(), logits.Shape().Dimensions[0], -1); err != nil {
				panic(fmt.Errorf("teacher ranking: %w", err))
			}
			if teacherRanking.Shape().Dimensions[1] < k {
				panic(fmt.Sprintf("teacher ranking has %d columns, need top-%d", teacherRanking.Shape().Dimensions[1], k))
			}
			teacherTopK := graph.Slice(teacherRanking, graph.AxisRange(), graph.AxisRangeFromStart(k))
			predicted := graph.ArgMax(logits, -1, teacherRanking.DType())
			predicted = graph.BroadcastToShape(graph.InsertAxes(predicted, -1), teacherTopK.Shape())
			matchesTeacher := graph.LogicalAny(graph.Equal(predicted, teacherTopK), -1)
			return graph.ReduceAllMean(graph.ConvertDType(matchesTeacher, logits.DType()))
		}

		label := graph.Squeeze(labels[0], -1)
		labelMask := graph.OneHot(label, logits.Shape().Dimensions[1], logits.DType())
		topKMask := graph.ConvertDType(graph.TopKMask(logits, k, -1), logits.DType())
		return graph.ReduceAllMean(graph.ReduceSum(graph.Mul(labelMask, topKMask), -1))
	}
}

// adamWithGradients is the public-gradient extension implemented by GoMLX Adam.
// The concrete Adam type is deliberately unexported, but its exported method is
// available through this narrow interface without changing the pinned library.
type adamWithGradients interface {
	optimizer.Interface
	UpdateGraphWithGradients(scope *model.Scope, gradients []*graph.Node, lossDType dtypes.DType)
}

// clippedAdam wraps the pinned GoMLX Adam update with true global-norm clipping
// before its moments are updated.
type clippedAdam struct {
	inner        adamWithGradients
	learningRate float64
	maxNorm      float64
}

var _ optimizer.Interface = (*clippedAdam)(nil)

// NewAdamWithGlobalNormClip creates fixed-learning-rate Adam with no weight
// decay and pre-Adam global gradient clipping.
func NewAdamWithGlobalNormClip(learningRate, maxNorm float64) optimizer.Interface {
	if learningRate <= 0 || math.IsNaN(learningRate) || math.IsInf(learningRate, 0) {
		panic(fmt.Sprintf("learning rate must be finite and positive, got %g", learningRate))
	}
	if maxNorm <= 0 || math.IsNaN(maxNorm) || math.IsInf(maxNorm, 0) {
		panic(fmt.Sprintf("global gradient clip norm must be finite and positive, got %g", maxNorm))
	}
	base := optimizer.Adam().LearningRate(learningRate).WeightDecay(0).Done()
	inner, ok := base.(adamWithGradients)
	if !ok {
		panic("pinned GoMLX Adam does not expose UpdateGraphWithGradients")
	}
	return &clippedAdam{inner: inner, learningRate: learningRate, maxNorm: maxNorm}
}

// UpdateGraph implements optimizer.Interface.
func (o *clippedAdam) UpdateGraph(scope *model.Scope, _ *graph.Graph, theLoss *graph.Node) {
	if !theLoss.Shape().IsScalar() {
		panic(fmt.Sprintf("optimizer requires a scalar loss, got %s", theLoss.Shape()))
	}
	o.UpdateGraphWithGradients(scope, scope.BuildTrainableVariablesGradientsGraph(theLoss), theLoss.DType())
}

// UpdateGraphWithGradients keeps gradient accumulation compatible with
// train.Trainer while applying the same global clipping to its averaged
// gradients before delegating to GoMLX Adam.
func (o *clippedAdam) UpdateGraphWithGradients(scope *model.Scope, gradients []*graph.Node, lossDType dtypes.DType) {
	if len(gradients) == 0 {
		panic("optimizer received no gradients")
	}
	g := gradients[0].Graph()
	variables := trainableVariablesInGraph(scope, g)
	if len(variables) != len(gradients) {
		panic(fmt.Sprintf("got %d gradients for %d trainable variables", len(gradients), len(variables)))
	}

	gradientNorm := stableGlobalL2Norm(gradients, lossDType)
	preclipNorm := gradientNorm.value
	clipLimit := graph.Scalar(g, lossDType, o.maxNorm)
	clippedGradients, appliedGradientNorm := gradientNorm.clip(gradients, clipLimit)
	gradientFinite := gradientNorm.finite
	publishGradientNorms(scope, variables, gradients, lossDType)
	before := make([]*graph.Node, len(variables))
	for index, variable := range variables {
		before[index] = variable.NodeValue(g)
	}

	o.inner.UpdateGraphWithGradients(scope, clippedGradients, lossDType)

	after := make([]*graph.Node, len(variables))
	updates := make([]*graph.Node, len(variables))
	for index, variable := range variables {
		after[index] = variable.NodeValue(g)
		updates[index] = graph.Sub(after[index], before[index])
	}
	parameterStats := stableGlobalL2Norm(after, lossDType)
	parameterNorm := parameterStats.value
	updateNorm := globalL2Norm(updates, lossDType)
	parameterFloor := graph.Scalar(g, lossDType, 1e-12)
	updateToParameterNorm := graph.Div(updateNorm, graph.Max(parameterNorm, parameterFloor))
	learningRate := optimizer.LearningRateVar(scope, lossDType, o.learningRate).NodeValue(g)
	publishTrainingDiagnostics(scope, TrainingDiagnosticsNodes{
		PreclipGlobalGradientNorm: preclipNorm,
		AppliedGlobalGradientNorm: appliedGradientNorm,
		GradientsFinite:           gradientFinite,
		ParametersFinite:          parameterStats.finite,
		ParameterNorm:             parameterNorm,
		UpdateToParameterNorm:     updateToParameterNorm,
		LearningRate:              learningRate,
	})
}

// Clear removes only the Adam state owned by the wrapped optimizer.
func (o *clippedAdam) Clear(scope *model.Scope) error {
	return o.inner.Clear(scope)
}

type TrainingDiagnosticsNodes struct {
	PreclipGlobalGradientNorm *graph.Node
	AppliedGlobalGradientNorm *graph.Node
	GradientsFinite           *graph.Node
	ParametersFinite          *graph.Node
	ParameterNorm             *graph.Node
	UpdateToParameterNorm     *graph.Node
	LearningRate              *graph.Node
}

func publishTrainingDiagnostics(scope *model.Scope, values TrainingDiagnosticsNodes) {
	diagnostics := scope.Store().Scope(diagnosticsScope)
	diagnostics.VariableWithValue(preclipGradientNormName, float32(0)).SetTrainable(false).SetNodeValue(values.PreclipGlobalGradientNorm)
	diagnostics.VariableWithValue(appliedGradientNormName, float32(0)).SetTrainable(false).SetNodeValue(values.AppliedGlobalGradientNorm)
	diagnostics.VariableWithValue(gradientFiniteName, true).SetTrainable(false).SetNodeValue(values.GradientsFinite)
	diagnostics.VariableWithValue(parameterFiniteName, true).SetTrainable(false).SetNodeValue(values.ParametersFinite)
	diagnostics.VariableWithValue(parameterNormName, float32(0)).SetTrainable(false).SetNodeValue(values.ParameterNorm)
	diagnostics.VariableWithValue(updateToParameterNormName, float32(0)).SetTrainable(false).SetNodeValue(values.UpdateToParameterNorm)
	diagnostics.VariableWithValue(learningRateName, float32(0)).SetTrainable(false).SetNodeValue(values.LearningRate)
}

// publishGradientNorms retains one pre-clipping L2 norm per trainable variable.
// The value is held in a non-trainable Store variable, so it is naturally
// checkpointed with the model and can be emitted as checkpoint-interval
// telemetry without rebuilding a training graph.
func publishGradientNorms(scope *model.Scope, variables []*model.Variable, gradients []*graph.Node, dtype dtypes.DType) {
	if len(variables) != len(gradients) {
		panic(fmt.Sprintf("got %d gradient norms for %d trainable variables", len(gradients), len(variables)))
	}
	for index, variable := range variables {
		path := gradientNormDiagnosticPath(variable.Path())
		diagnostic := scope.Store().VariableWithValue(path, float32(0)).SetTrainable(false)
		diagnostic.SetNodeValue(globalL2Norm([]*graph.Node{gradients[index]}, dtype))
	}
}

func gradientNormDiagnosticPath(trainablePath string) string {
	return model.JoinPath(gradientNormsScope, strings.TrimPrefix(trainablePath, "/"))
}

// Diagnostics returns values produced by the latest training update.
func (s *Session) Diagnostics() (TrainingDiagnostics, error) {
	if s == nil || s.Store == nil {
		return TrainingDiagnostics{}, fmt.Errorf("supervised session has no Store")
	}
	return ReadTrainingDiagnostics(s.Store)
}

// ReadTrainingDiagnostics reads the diagnostics stored by clippedAdam. It is
// useful after independently loading a checkpoint as well as through Session.
func ReadTrainingDiagnostics(store *model.Store) (TrainingDiagnostics, error) {
	if store == nil {
		return TrainingDiagnostics{}, fmt.Errorf("diagnostics Store is nil")
	}
	preclipNorm, err := diagnosticFloat(store, preclipGradientNormName)
	if err != nil {
		return TrainingDiagnostics{}, err
	}
	appliedNorm, err := diagnosticFloat(store, appliedGradientNormName)
	if err != nil {
		return TrainingDiagnostics{}, err
	}
	gradientsFinite, err := diagnosticBool(store, gradientFiniteName)
	if err != nil {
		return TrainingDiagnostics{}, err
	}
	parametersFinite, err := diagnosticBool(store, parameterFiniteName)
	if err != nil {
		return TrainingDiagnostics{}, err
	}
	parameterNorm, err := diagnosticFloat(store, parameterNormName)
	if err != nil {
		return TrainingDiagnostics{}, err
	}
	updateRatio, err := diagnosticFloat(store, updateToParameterNormName)
	if err != nil {
		return TrainingDiagnostics{}, err
	}
	learningRate, err := diagnosticFloat(store, learningRateName)
	if err != nil {
		return TrainingDiagnostics{}, err
	}
	return TrainingDiagnostics{
		PreclipGlobalGradientNorm: preclipNorm,
		AppliedGlobalGradientNorm: appliedNorm,
		GradientsFinite:           gradientsFinite,
		ParametersFinite:          parametersFinite,
		ParameterNorm:             parameterNorm,
		UpdateToParameterNorm:     updateRatio,
		LearningRate:              learningRate,
	}, nil
}

// ReadGradientNorms returns the latest pre-clipping L2 gradient norm for every
// materialized trainable variable, sorted by the original exact Store path.
// Before the first optimizer update it returns an empty slice. It also loads
// checkpointed norm variables lazily once their corresponding model variable
// has been materialized.
func ReadGradientNorms(store *model.Store) ([]NamedNorm, error) {
	if store == nil {
		return nil, fmt.Errorf("gradient-norm diagnostics Store is nil")
	}
	trainablePaths := make([]string, 0)
	for variable := range store.IterVariables() {
		if variable.Trainable {
			trainablePaths = append(trainablePaths, variable.Path())
		}
	}
	sort.Strings(trainablePaths)

	norms := make([]NamedNorm, 0, len(trainablePaths))
	for _, trainablePath := range trainablePaths {
		diagnostic := store.GetVariable(gradientNormDiagnosticPath(trainablePath))
		if diagnostic == nil {
			// There is no norm before the first optimizer update, or the variable
			// was added after the checkpoint whose diagnostics are being read.
			continue
		}
		// Checkpoint loaders restore tensor values but not Variable metadata, so
		// make the diagnostic non-trainable again as soon as it is lazily loaded.
		diagnostic.SetTrainable(false)
		value, err := diagnostic.Value()
		if err != nil {
			return nil, fmt.Errorf("read gradient norm %s: %w", trainablePath, err)
		}
		values, err := tensors.CopyFlatData[float32](value)
		if err != nil || len(values) != 1 {
			if err == nil {
				err = fmt.Errorf("want scalar, got %d values", len(values))
			}
			return nil, fmt.Errorf("read gradient norm %s: %w", trainablePath, err)
		}
		norms = append(norms, NamedNorm{Path: trainablePath, Norm: values[0]})
	}
	return norms, nil
}

func diagnosticFloat(store *model.Store, name string) (float32, error) {
	variable := store.GetVariable(model.JoinPath(diagnosticsScope, name))
	if variable == nil {
		return 0, fmt.Errorf("training diagnostics are unavailable before the first optimizer update")
	}
	variable.SetTrainable(false)
	value, err := variable.Value()
	if err != nil {
		return 0, fmt.Errorf("read diagnostic %s: %w", name, err)
	}
	values, err := tensors.CopyFlatData[float32](value)
	if err != nil || len(values) != 1 {
		if err == nil {
			err = fmt.Errorf("want scalar, got %d values", len(values))
		}
		return 0, fmt.Errorf("read diagnostic %s: %w", name, err)
	}
	return values[0], nil
}

func diagnosticBool(store *model.Store, name string) (bool, error) {
	variable := store.GetVariable(model.JoinPath(diagnosticsScope, name))
	if variable == nil {
		return false, fmt.Errorf("training diagnostics are unavailable before the first optimizer update")
	}
	variable.SetTrainable(false)
	value, err := variable.Value()
	if err != nil {
		return false, fmt.Errorf("read diagnostic %s: %w", name, err)
	}
	values, err := tensors.CopyFlatData[bool](value)
	if err != nil || len(values) != 1 {
		if err == nil {
			err = fmt.Errorf("want scalar, got %d values", len(values))
		}
		return false, fmt.Errorf("read diagnostic %s: %w", name, err)
	}
	return values[0], nil
}

func trainableVariablesInGraph(scope *model.Scope, g *graph.Graph) []*model.Variable {
	variables := make([]*model.Variable, 0)
	for variable := range scope.IterVariables() {
		if variable.Trainable && variable.InUseByGraph(g) {
			variables = append(variables, variable)
		}
	}
	return variables
}

// stableGlobalL2 carries an overflow-safe L2 norm representation. value is
// capped at the largest finite value of the calculation dtype when every input
// is finite; non-finite inputs deliberately produce +Inf so safety checks do
// not mistake them for a large hard-masked value.
type stableGlobalL2 struct {
	finite         *graph.Node
	normalizer     *graph.Node
	normalizedNorm *graph.Node
	value          *graph.Node
}

func stableGlobalL2Norm(nodes []*graph.Node, dtype dtypes.DType) stableGlobalL2 {
	if len(nodes) == 0 {
		panic("cannot calculate the norm of no nodes")
	}
	g := nodes[0].Graph()
	normalizer := graph.ScalarOne(g, dtype)
	for _, node := range nodes {
		value := node
		if value.DType() != dtype {
			value = graph.ConvertDType(value, dtype)
		}
		normalizer = graph.Max(normalizer, graph.ReduceAllMax(graph.Abs(value)))
	}

	sumSquares := graph.ScalarZero(g, dtype)
	for _, node := range nodes {
		value := node
		if value.DType() != dtype {
			value = graph.ConvertDType(value, dtype)
		}
		// Scaling by the largest absolute component keeps Square and ReduceSum
		// finite for any finite FP32 input.
		normalized := graph.Div(value, normalizer)
		sumSquares = graph.Add(sumSquares, graph.ReduceAllSum(graph.Square(normalized)))
	}
	normalizedNorm := graph.Sqrt(sumSquares)
	finite := allFinite(nodes)
	uncappedNorm := graph.Mul(normalizer, normalizedNorm)
	maxFinite := graph.Scalar(g, dtype, math.MaxFloat32)
	value := graph.Where(
		finite,
		graph.Min(uncappedNorm, maxFinite),
		graph.Infinity(g, dtype, 1),
	)
	return stableGlobalL2{
		finite:         finite,
		normalizer:     normalizer,
		normalizedNorm: normalizedNorm,
		value:          value,
	}
}

// clip applies global-norm clipping without multiplying extreme finite values
// by a subnormal global scale. For a clipped vector it first divides by the
// global normalizer, then applies clipLimit/normalizedNorm; this stays in a
// representable range even when the original norm exceeds Float32's range.
func (n stableGlobalL2) clip(gradients []*graph.Node, clipLimit *graph.Node) ([]*graph.Node, *graph.Node) {
	g := clipLimit.Graph()
	normalizedClipLimit := graph.Div(clipLimit, n.normalizer)
	needsClip := graph.And(n.finite, graph.GreaterThan(n.normalizedNorm, normalizedClipLimit))
	nonzeroNormalizedNorm := graph.Where(
		graph.Equal(n.normalizedNorm, graph.ScalarZero(g, n.normalizedNorm.DType())),
		graph.ScalarOne(g, n.normalizedNorm.DType()),
		n.normalizedNorm,
	)
	clippedMagnitude := graph.Div(clipLimit, nonzeroNormalizedNorm)

	clipped := make([]*graph.Node, len(gradients))
	for index, gradient := range gradients {
		value := gradient
		if value.DType() != n.normalizer.DType() {
			value = graph.ConvertDType(value, n.normalizer.DType())
		}
		value = graph.Mul(graph.Div(value, n.normalizer), clippedMagnitude)
		if value.DType() != gradient.DType() {
			value = graph.ConvertDType(value, gradient.DType())
		}
		// Leave NaN/+Inf gradients untouched. Their non-finiteness is surfaced
		// in diagnostics and by Adam rather than converted into a zero update.
		clipped[index] = graph.Where(needsClip, value, gradient)
	}

	applied := graph.Where(
		n.finite,
		graph.Where(needsClip, clipLimit, n.value),
		graph.Infinity(g, clipLimit.DType(), 1),
	)
	return clipped, applied
}

func globalL2Norm(nodes []*graph.Node, dtype dtypes.DType) *graph.Node {
	return stableGlobalL2Norm(nodes, dtype).value
}

func allFinite(nodes []*graph.Node) *graph.Node {
	if len(nodes) == 0 {
		panic("cannot test finiteness of no nodes")
	}
	g := nodes[0].Graph()
	finite := graph.ScalarOne(g, dtypes.Bool)
	for _, node := range nodes {
		finite = graph.And(finite, graph.LogicalAll(graph.IsFinite(node)))
	}
	return finite
}
