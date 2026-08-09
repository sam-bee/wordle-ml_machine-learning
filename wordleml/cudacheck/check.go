// Package cudacheck verifies exported Wordle CUDA models without importing
// GoMLX. It compares the recorded reference logits, the portable evaluator,
// and a CUDA scorer, while leaving Wordle legality in Go.
package cudacheck

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/sam-bee/wordle-ml_machine-learning/cudainfer"
	"github.com/sam-bee/wordle-ml_machine-learning/cudamodel"
	"github.com/sam-bee/wordle-ml_machine-learning/cudaref"
	"github.com/sam-bee/wordle-ml_machine-learning/gameeval"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

const (
	// DefaultTolerance is the initial absolute logit tolerance for portable and
	// CUDA parity. Callers must make any change visible in their report.
	DefaultTolerance = 1e-3
	// VerificationFormat identifies the machine-readable verifier report.
	VerificationFormat = "wordle-cuda-verification-v1"
)

// Scorer is the CUDA side of the Go/CUDA boundary used by verification and
// benchmarking. It deliberately returns raw logits only.
type Scorer interface {
	Score(context.Context, modelstate.Inputs) ([]float32, error)
}

// Identity records the immutable model and device facts attached to reports.
type Identity struct {
	RunID            string         `json:"run_id"`
	Checkpoint       string         `json:"checkpoint"`
	CheckpointUpdate int            `json:"checkpoint_update"`
	TrainingCommit   string         `json:"training_commit"`
	WeightsSHA256    string         `json:"weights_sha256"`
	ParameterCount   int            `json:"parameter_count"`
	Device           cudainfer.Info `json:"device"`
}

// IdentityFor returns the report identity for one validated model and backend.
func IdentityFor(manifest cudamodel.Manifest, device cudainfer.Info) Identity {
	return Identity{
		RunID:            manifest.RunID,
		Checkpoint:       manifest.Checkpoint,
		CheckpointUpdate: manifest.CheckpointUpdate,
		TrainingCommit:   manifest.TrainingCommit,
		WeightsSHA256:    manifest.WeightsSHA256,
		ParameterCount:   manifest.ParameterCount,
		Device:           device,
	}
}

// Selection is one deterministic host-side raw and legal action decision.
type Selection struct {
	RawTopActionID   int     `json:"raw_top_action_id"`
	SelectedActionID int     `json:"selected_action_id"`
	TopTwoMargin     float32 `json:"top_two_margin"`
}

// LogitComparison is one all-action comparison for a golden position.
type LogitComparison struct {
	Values           int     `json:"values"`
	MaximumAbsolute  float64 `json:"maximum_absolute_error"`
	MeanAbsolute     float64 `json:"mean_absolute_error"`
	WorstActionID    int     `json:"worst_action_id"`
	Top1Agreement    bool    `json:"top1_agreement"`
	Top5SetAgreement bool    `json:"top5_set_agreement"`
	WithinTolerance  bool    `json:"within_tolerance"`
}

// PairSummary aggregates all vector comparisons for one evaluator pair.
type PairSummary struct {
	Vectors                 int     `json:"vectors"`
	Values                  int     `json:"values"`
	MaximumAbsolute         float64 `json:"maximum_absolute_error"`
	MeanAbsolute            float64 `json:"mean_absolute_error"`
	WorstVectorID           string  `json:"worst_vector_id,omitempty"`
	WorstActionID           int     `json:"worst_action_id,omitempty"`
	Top1Agreement           int     `json:"top1_agreement"`
	Top5SetAgreement        int     `json:"top5_set_agreement"`
	SelectedActionAgreement int     `json:"selected_action_agreement"`
	ToleranceFailures       int     `json:"tolerance_failures"`

	totalAbsolute float64
}

// VectorVerification holds all three parity comparisons for one golden input.
type VectorVerification struct {
	ID                 string          `json:"id"`
	ReferencePortable  LogitComparison `json:"reference_vs_portable_go"`
	PortableCUDA       LogitComparison `json:"portable_go_vs_cuda"`
	ReferenceCUDA      LogitComparison `json:"reference_vs_cuda"`
	ReferenceSelection Selection       `json:"reference_selection"`
	PortableSelection  Selection       `json:"portable_selection"`
	CUDASelection      Selection       `json:"cuda_selection"`
}

// Divergence identifies the first exact raw or legal-action mismatch. It
// includes both raw-top values and margins so a near tie is explicit instead
// of silently accepted.
type Divergence struct {
	VectorID             string  `json:"vector_id"`
	Pair                 string  `json:"pair"`
	ExpectedRawTop       int     `json:"expected_raw_top_action_id"`
	ActualRawTop         int     `json:"actual_raw_top_action_id"`
	ExpectedSelected     int     `json:"expected_selected_action_id"`
	ActualSelected       int     `json:"actual_selected_action_id"`
	ExpectedRawTopLogit  float32 `json:"expected_raw_top_logit"`
	ActualRawTopLogit    float32 `json:"actual_raw_top_logit"`
	ExpectedTopTwoMargin float32 `json:"expected_top_two_margin"`
	ActualTopTwoMargin   float32 `json:"actual_top_two_margin"`
	NearTie              bool    `json:"near_tie"`
}

// VerificationReport contains numeric, selection, and complete-game evidence.
type VerificationReport struct {
	Format            string               `json:"format"`
	Identity          Identity             `json:"identity"`
	Tolerance         float64              `json:"absolute_tolerance"`
	GoldenVectorCount int                  `json:"golden_vector_count"`
	ReferencePortable PairSummary          `json:"reference_vs_portable_go"`
	PortableCUDA      PairSummary          `json:"portable_go_vs_cuda"`
	ReferenceCUDA     PairSummary          `json:"reference_vs_cuda"`
	Vectors           []VectorVerification `json:"vectors"`
	FirstDivergence   *Divergence          `json:"first_action_divergence,omitempty"`
	Games             GamesReport          `json:"games"`
	Failures          []string             `json:"failures,omitempty"`
	Passed            bool                 `json:"passed"`
}

// VerifyOptions supplies only GoMLX-free inputs to the complete verifier.
type VerifyOptions struct {
	Model      *cudamodel.Model
	Vocabulary *vocabulary.Vocabulary
	Golden     cudaref.GoldenSet
	Games      cudaref.Games
	Backend    Scorer
	Device     cudainfer.Info
	Tolerance  float64
}

// Verify compares every golden vector and then replays all 100 validation
// games through the supplied CUDA scorer. It records ordinary parity failures
// in the report; malformed input or unavailable dependencies return an error.
func Verify(ctx context.Context, options VerifyOptions) (VerificationReport, error) {
	if ctx == nil {
		return VerificationReport{}, fmt.Errorf("verification context is nil")
	}
	if options.Model == nil || options.Vocabulary == nil || options.Backend == nil {
		return VerificationReport{}, fmt.Errorf("model, vocabulary, and CUDA backend are required")
	}
	if options.Tolerance <= 0 || math.IsNaN(options.Tolerance) || math.IsInf(options.Tolerance, 0) {
		return VerificationReport{}, fmt.Errorf("absolute tolerance must be finite and positive, got %g", options.Tolerance)
	}
	if len(options.Vocabulary.Test()) != 0 || options.Vocabulary.Hashes().Test != "" {
		return VerificationReport{}, fmt.Errorf("verification vocabulary must be loaded without the final-test word list")
	}
	if len(options.Golden.Vectors) == 0 {
		return VerificationReport{}, fmt.Errorf("golden vector set is empty")
	}

	report := VerificationReport{
		Format:            VerificationFormat,
		Identity:          IdentityFor(options.Model.Manifest, options.Device),
		Tolerance:         options.Tolerance,
		GoldenVectorCount: len(options.Golden.Vectors),
		Vectors:           make([]VectorVerification, 0, len(options.Golden.Vectors)),
	}

	for _, vector := range options.Golden.Vectors {
		if err := ctx.Err(); err != nil {
			return report, err
		}
		if err := cudaref.ValidateVector(vector); err != nil {
			report.addFailure("validate golden vector %q: %v", vector.ID, err)
			continue
		}
		portable, err := options.Model.Forward(vector.Inputs)
		if err != nil {
			report.addFailure("portable evaluator vector %q: %v", vector.ID, err)
			continue
		}
		cuda, err := options.Backend.Score(ctx, vector.Inputs)
		if err != nil {
			report.addFailure("CUDA evaluator vector %q: %v", vector.ID, err)
			continue
		}
		vectorReport, err := compareVector(vector, portable, cuda, options.Tolerance)
		if err != nil {
			report.addFailure("compare vector %q: %v", vector.ID, err)
			continue
		}
		report.Vectors = append(report.Vectors, vectorReport)
		report.ReferencePortable.add(vector.ID, vectorReport.ReferencePortable, vectorReport.ReferenceSelection, vectorReport.PortableSelection)
		report.PortableCUDA.add(vector.ID, vectorReport.PortableCUDA, vectorReport.PortableSelection, vectorReport.CUDASelection)
		report.ReferenceCUDA.add(vector.ID, vectorReport.ReferenceCUDA, vectorReport.ReferenceSelection, vectorReport.CUDASelection)
		if !vectorReport.ReferencePortable.WithinTolerance {
			report.addFailure("reference and portable logits exceed tolerance for vector %q", vector.ID)
		}
		if !vectorReport.PortableCUDA.WithinTolerance {
			report.addFailure("portable and CUDA logits exceed tolerance for vector %q", vector.ID)
		}
		if !vectorReport.ReferenceCUDA.WithinTolerance {
			report.addFailure("reference and CUDA logits exceed tolerance for vector %q", vector.ID)
		}
		report.recordActionDivergence(vector, vectorReport.ReferenceSelection, vectorReport.PortableSelection, vector.RawLogits, portable, "reference_vs_portable_go")
		report.recordActionDivergence(vector, vectorReport.PortableSelection, vectorReport.CUDASelection, portable, cuda, "portable_go_vs_cuda")
		report.recordActionDivergence(vector, vectorReport.ReferenceSelection, vectorReport.CUDASelection, vector.RawLogits, cuda, "reference_vs_cuda")
	}

	report.Games = VerifyGames(ctx, options.Model.Manifest, options.Vocabulary, options.Games, options.Backend)
	if !report.Games.Exact {
		if report.Games.Error != "" {
			report.addFailure("validation games: %s", report.Games.Error)
		} else if report.Games.FirstDivergence != nil {
			report.addFailure("validation games diverged at solution %q turn %d field %s", report.Games.FirstDivergence.Solution, report.Games.FirstDivergence.Turn, report.Games.FirstDivergence.Field)
		} else {
			report.addFailure("validation games differ from reference")
		}
	}
	report.finish()
	return report, nil
}

func compareVector(vector cudaref.Vector, portable, cuda []float32, tolerance float64) (VectorVerification, error) {
	referenceSelection, err := SelectAction(vector.RawLogits, vector.AvailableActionMask)
	if err != nil {
		return VectorVerification{}, fmt.Errorf("select reference action: %w", err)
	}
	if referenceSelection.RawTopActionID != vector.RawTopActionID || referenceSelection.SelectedActionID != vector.SelectedActionID {
		return VectorVerification{}, fmt.Errorf("golden stored actions are raw=%d/selected=%d, recomputed raw=%d/selected=%d", vector.RawTopActionID, vector.SelectedActionID, referenceSelection.RawTopActionID, referenceSelection.SelectedActionID)
	}
	portableSelection, err := SelectAction(portable, vector.AvailableActionMask)
	if err != nil {
		return VectorVerification{}, fmt.Errorf("select portable action: %w", err)
	}
	cudaSelection, err := SelectAction(cuda, vector.AvailableActionMask)
	if err != nil {
		return VectorVerification{}, fmt.Errorf("select CUDA action: %w", err)
	}
	referencePortable, err := CompareLogits(vector.RawLogits, portable, tolerance)
	if err != nil {
		return VectorVerification{}, fmt.Errorf("reference vs portable: %w", err)
	}
	portableCUDA, err := CompareLogits(portable, cuda, tolerance)
	if err != nil {
		return VectorVerification{}, fmt.Errorf("portable vs CUDA: %w", err)
	}
	referenceCUDA, err := CompareLogits(vector.RawLogits, cuda, tolerance)
	if err != nil {
		return VectorVerification{}, fmt.Errorf("reference vs CUDA: %w", err)
	}
	return VectorVerification{
		ID:                 vector.ID,
		ReferencePortable:  referencePortable,
		PortableCUDA:       portableCUDA,
		ReferenceCUDA:      referenceCUDA,
		ReferenceSelection: referenceSelection,
		PortableSelection:  portableSelection,
		CUDASelection:      cudaSelection,
	}, nil
}

// CompareLogits compares two complete raw-logit vectors. It uses stable
// lower-action-ID tie handling for top-k agreement, matching gameeval.
func CompareLogits(expected, actual []float32, tolerance float64) (LogitComparison, error) {
	if len(expected) != vocabulary.NumActions || len(actual) != vocabulary.NumActions {
		return LogitComparison{}, fmt.Errorf("logit lengths are expected=%d actual=%d, want %d", len(expected), len(actual), vocabulary.NumActions)
	}
	if tolerance <= 0 || math.IsNaN(tolerance) || math.IsInf(tolerance, 0) {
		return LogitComparison{}, fmt.Errorf("absolute tolerance must be finite and positive, got %g", tolerance)
	}
	result := LogitComparison{Values: len(expected), WorstActionID: -1, Top1Agreement: true, Top5SetAgreement: true}
	var total float64
	for actionID, value := range expected {
		if !finite(value) {
			return LogitComparison{}, fmt.Errorf("expected logit at action %d is non-finite", actionID)
		}
		if !finite(actual[actionID]) {
			return LogitComparison{}, fmt.Errorf("actual logit at action %d is non-finite", actionID)
		}
		difference := math.Abs(float64(value) - float64(actual[actionID]))
		total += difference
		if result.WorstActionID < 0 || difference > result.MaximumAbsolute {
			result.MaximumAbsolute = difference
			result.WorstActionID = actionID
		}
	}
	result.MeanAbsolute = total / float64(len(expected))
	result.WithinTolerance = result.MaximumAbsolute <= tolerance
	result.Top1Agreement = rawTop(expected) == rawTop(actual)
	result.Top5SetAgreement = sameIDs(topK(expected, 5), topK(actual, 5))
	return result, nil
}

// SelectAction applies the exact existing Go-side finite validation, raw argmax,
// legal mask, and lower-ID tie rule. CUDA never receives available.
func SelectAction(logits, available []float32) (Selection, error) {
	if len(logits) != vocabulary.NumActions || len(available) != vocabulary.NumActions {
		return Selection{}, fmt.Errorf("scores/mask lengths are %d/%d, want %d", len(logits), len(available), vocabulary.NumActions)
	}
	raw, selected := -1, -1
	var rawValue, selectedValue float32
	for actionID, value := range logits {
		if !finite(value) {
			return Selection{}, fmt.Errorf("logit at action %d is non-finite", actionID)
		}
		if raw < 0 || value > rawValue {
			raw, rawValue = actionID, value
		}
		if available[actionID] != 0 && (selected < 0 || value > selectedValue) {
			selected, selectedValue = actionID, value
		}
	}
	if selected < 0 {
		return Selection{}, gameeval.ErrNoLegalAction
	}
	ordered := rankedTopK(logits, 2)
	margin := float32(0)
	if len(ordered) == 2 {
		margin = logits[ordered[0]] - logits[ordered[1]]
	}
	return Selection{RawTopActionID: raw, SelectedActionID: selected, TopTwoMargin: margin}, nil
}

func (summary *PairSummary) add(vectorID string, comparison LogitComparison, expected, actual Selection) {
	summary.Vectors++
	summary.Values += comparison.Values
	summary.totalAbsolute += comparison.MeanAbsolute * float64(comparison.Values)
	if summary.WorstVectorID == "" || comparison.MaximumAbsolute > summary.MaximumAbsolute {
		summary.MaximumAbsolute = comparison.MaximumAbsolute
		summary.WorstVectorID = vectorID
		summary.WorstActionID = comparison.WorstActionID
	}
	if comparison.Top1Agreement {
		summary.Top1Agreement++
	}
	if comparison.Top5SetAgreement {
		summary.Top5SetAgreement++
	}
	if expected.SelectedActionID == actual.SelectedActionID {
		summary.SelectedActionAgreement++
	}
	if !comparison.WithinTolerance {
		summary.ToleranceFailures++
	}
}

func (report *VerificationReport) recordActionDivergence(vector cudaref.Vector, expected, actual Selection, expectedLogits, actualLogits []float32, pair string) {
	if expected.RawTopActionID == actual.RawTopActionID && expected.SelectedActionID == actual.SelectedActionID {
		return
	}
	if expected.RawTopActionID != actual.RawTopActionID {
		report.addFailure("%s raw top action differs for vector %q: expected %d (%.9g), got %d (%.9g)", pair, vector.ID, expected.RawTopActionID, expectedLogits[expected.RawTopActionID], actual.RawTopActionID, actualLogits[actual.RawTopActionID])
	}
	if expected.SelectedActionID != actual.SelectedActionID {
		report.addFailure("%s selected action differs for vector %q: expected %d, got %d", pair, vector.ID, expected.SelectedActionID, actual.SelectedActionID)
	}
	if report.FirstDivergence != nil {
		return
	}
	report.FirstDivergence = &Divergence{
		VectorID:             vector.ID,
		Pair:                 pair,
		ExpectedRawTop:       expected.RawTopActionID,
		ActualRawTop:         actual.RawTopActionID,
		ExpectedSelected:     expected.SelectedActionID,
		ActualSelected:       actual.SelectedActionID,
		ExpectedRawTopLogit:  expectedLogits[expected.RawTopActionID],
		ActualRawTopLogit:    actualLogits[actual.RawTopActionID],
		ExpectedTopTwoMargin: expected.TopTwoMargin,
		ActualTopTwoMargin:   actual.TopTwoMargin,
		NearTie:              math.Abs(float64(expected.TopTwoMargin)) <= report.Tolerance,
	}
}

func (report *VerificationReport) addFailure(format string, arguments ...any) {
	report.Failures = append(report.Failures, fmt.Sprintf(format, arguments...))
}

func (report *VerificationReport) finish() {
	for _, summary := range []*PairSummary{&report.ReferencePortable, &report.PortableCUDA, &report.ReferenceCUDA} {
		if summary.Values != 0 {
			summary.MeanAbsolute = summary.totalAbsolute / float64(summary.Values)
		}
	}
	report.Passed = len(report.Failures) == 0
}

func rawTop(values []float32) int { return rankedTopK(values, 1)[0] }

func topK(values []float32, count int) []int {
	ids := rankedTopK(values, count)
	sort.Ints(ids)
	return ids
}

func rankedTopK(values []float32, count int) []int {
	ids := make([]int, len(values))
	for index := range values {
		ids[index] = index
	}
	sort.Slice(ids, func(left, right int) bool {
		if values[ids[left]] == values[ids[right]] {
			return ids[left] < ids[right]
		}
		return values[ids[left]] > values[ids[right]]
	})
	if count < len(ids) {
		ids = ids[:count]
	}
	return ids
}

func sameIDs(left, right []int) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if left[index] != right[index] {
			return false
		}
	}
	return true
}

func finite(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}
