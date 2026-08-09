package cudacheck

import (
	"context"
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/sam-bee/wordle-ml_machine-learning/cudaref"
)

// BenchmarkFormat identifies the machine-readable batch-one CUDA benchmark.
const BenchmarkFormat = "wordle-cuda-benchmark-v1"

// BenchmarkOptions is deliberately small and deterministic: one golden input,
// a cold call, warm-ups, then individual synchronous CUDA calls.
type BenchmarkOptions struct {
	Backend    Scorer
	Vector     cudaref.Vector
	Tolerance  float64
	Warmup     int
	Iterations int
}

// DurationSummary reports nanoseconds so JSON consumers need no duration
// string parser. Percentiles use the nearest-rank definition.
type DurationSummary struct {
	Count   int   `json:"count"`
	Minimum int64 `json:"minimum_ns"`
	Mean    int64 `json:"mean_ns"`
	P50     int64 `json:"p50_ns"`
	P95     int64 `json:"p95_ns"`
	Maximum int64 `json:"maximum_ns"`
}

// BenchmarkReport is the deterministic timing result, including immutable
// model and device identity supplied by the command after backend creation.
type BenchmarkReport struct {
	Identity   Identity        `json:"identity"`
	Format     string          `json:"format"`
	VectorID   string          `json:"vector_id"`
	Tolerance  float64         `json:"absolute_tolerance"`
	Warmup     int             `json:"warmup_iterations"`
	Iterations int             `json:"measured_iterations"`
	ColdCallNS int64           `json:"cold_call_ns"`
	WarmCall   DurationSummary `json:"warm_calls"`
	Passed     bool            `json:"passed"`
	Failure    string          `json:"failure,omitempty"`
}

// Benchmark runs and verifies every timed output. It returns a report for a
// verification failure so callers can persist the evidence before exiting.
func Benchmark(ctx context.Context, options BenchmarkOptions) (BenchmarkReport, error) {
	if ctx == nil {
		return BenchmarkReport{}, fmt.Errorf("benchmark context is nil")
	}
	if options.Backend == nil {
		return BenchmarkReport{}, fmt.Errorf("CUDA backend is required")
	}
	if options.Warmup < 0 || options.Iterations <= 0 {
		return BenchmarkReport{}, fmt.Errorf("warmup must be non-negative and iterations must be positive")
	}
	if options.Tolerance <= 0 || math.IsNaN(options.Tolerance) || math.IsInf(options.Tolerance, 0) {
		return BenchmarkReport{}, fmt.Errorf("absolute tolerance must be finite and positive, got %g", options.Tolerance)
	}
	if err := cudaref.ValidateVector(options.Vector); err != nil {
		return BenchmarkReport{}, fmt.Errorf("benchmark vector: %w", err)
	}
	report := BenchmarkReport{
		Format:     BenchmarkFormat,
		VectorID:   options.Vector.ID,
		Tolerance:  options.Tolerance,
		Warmup:     options.Warmup,
		Iterations: options.Iterations,
	}
	if duration, err := scoreAndVerify(ctx, options.Backend, options.Vector, options.Tolerance); err != nil {
		report.Failure = "cold call: " + err.Error()
		return report, nil
	} else {
		report.ColdCallNS = duration.Nanoseconds()
	}
	for iteration := 0; iteration < options.Warmup; iteration++ {
		if _, err := scoreAndVerify(ctx, options.Backend, options.Vector, options.Tolerance); err != nil {
			report.Failure = fmt.Sprintf("warmup iteration %d: %v", iteration, err)
			return report, nil
		}
	}
	durations := make([]time.Duration, 0, options.Iterations)
	for iteration := 0; iteration < options.Iterations; iteration++ {
		duration, err := scoreAndVerify(ctx, options.Backend, options.Vector, options.Tolerance)
		if err != nil {
			report.Failure = fmt.Sprintf("measured iteration %d: %v", iteration, err)
			return report, nil
		}
		durations = append(durations, duration)
	}
	report.WarmCall = summarizeDurations(durations)
	report.Passed = true
	return report, nil
}

func scoreAndVerify(ctx context.Context, backend Scorer, vector cudaref.Vector, tolerance float64) (time.Duration, error) {
	start := time.Now()
	actual, err := backend.Score(ctx, vector.Inputs)
	duration := time.Since(start)
	if err != nil {
		return duration, err
	}
	comparison, err := CompareLogits(vector.RawLogits, actual, tolerance)
	if err != nil {
		return duration, err
	}
	if !comparison.WithinTolerance {
		return duration, fmt.Errorf("maximum absolute error %.9g exceeds tolerance %.9g at action %d", comparison.MaximumAbsolute, tolerance, comparison.WorstActionID)
	}
	selection, err := SelectAction(actual, vector.AvailableActionMask)
	if err != nil {
		return duration, err
	}
	if selection.RawTopActionID != vector.RawTopActionID {
		return duration, fmt.Errorf("raw top action %d differs from reference %d", selection.RawTopActionID, vector.RawTopActionID)
	}
	if selection.SelectedActionID != vector.SelectedActionID {
		return duration, fmt.Errorf("selected action %d differs from reference %d", selection.SelectedActionID, vector.SelectedActionID)
	}
	return duration, nil
}

func summarizeDurations(values []time.Duration) DurationSummary {
	if len(values) == 0 {
		return DurationSummary{}
	}
	sorted := append([]time.Duration(nil), values...)
	sort.Slice(sorted, func(left, right int) bool { return sorted[left] < sorted[right] })
	var total int64
	for _, value := range sorted {
		total += value.Nanoseconds()
	}
	return DurationSummary{
		Count:   len(sorted),
		Minimum: sorted[0].Nanoseconds(),
		Mean:    total / int64(len(sorted)),
		P50:     percentile(sorted, 50).Nanoseconds(),
		P95:     percentile(sorted, 95).Nanoseconds(),
		Maximum: sorted[len(sorted)-1].Nanoseconds(),
	}
}

func percentile(sorted []time.Duration, percent int) time.Duration {
	index := (percent*len(sorted)+99)/100 - 1
	return sorted[index]
}
