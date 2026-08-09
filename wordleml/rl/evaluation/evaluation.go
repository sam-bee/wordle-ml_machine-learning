// Package evaluation contains the host-side, deterministic reporting used by
// the bounded PPO experiment. It deliberately knows nothing about GoMLX or
// Wordle transitions: the live game engine remains responsible for producing
// GameResult records.
package evaluation

import (
	"errors"
	"fmt"
	"math"
	"sort"
)

const (
	// MaxTurns is the fixed Wordle game length used throughout this project.
	MaxTurns = 6
	// FailureCountedGuesses is the score assigned to an unsolved game for the
	// primary deployment comparison. It deliberately differs from the six
	// accepted guesses recorded in a failed game.
	FailureCountedGuesses = MaxTurns + 1

	// DefaultBootstrapSamples gives a reasonably stable interval for the small
	// fixed development population without making the report command costly.
	DefaultBootstrapSamples = 10_000
	// DefaultMaxOldPolicyKL implements PPO's conservative per-iteration KL
	// guard. It is intentionally aligned with the trainer's epoch-stop target.
	DefaultMaxOldPolicyKL = 0.01
	// DefaultMaxSupervisedReferenceKL limits cumulative drift from the frozen,
	// successful supervised actor. It is a separate guard from old-policy KL.
	DefaultMaxSupervisedReferenceKL = 0.02
	// DefaultMinimumPolicyEntropy is a floor rather than a target. A policy
	// below it is treated as collapsed and cannot be promoted.
	DefaultMinimumPolicyEntropy = 0.10
)

// TurnRecord is one accepted engine transition from an evaluation game. Turn
// is one-based. ActionProbability is the masked categorical probability of
// the accepted action; it makes the reported opening directly inspectable.
// Reward and Terminal are retained so that the same records are useful to a
// future runner when it writes completed-game artifacts.
type TurnRecord struct {
	Turn              int     `json:"turn"`
	Guess             string  `json:"guess"`
	ActionID          int     `json:"action_id"`
	ActionProbability float64 `json:"action_probability"`
	PolicyEntropy     float64 `json:"policy_entropy"`
	Reward            float64 `json:"reward"`
	Terminal          bool    `json:"terminal"`
}

// GameResult is an authoritative, complete Wordle game outcome. SolutionID is
// environment-side metadata: it is used solely to pair development games and
// must never be provided to a model input. Guesses is the actual number of
// accepted guesses (one through six), including six for a terminal failure.
type GameResult struct {
	SolutionID              string       `json:"solution_id"`
	Solved                  bool         `json:"solved"`
	Guesses                 int          `json:"guesses"`
	Turns                   []TurnRecord `json:"turns"`
	AcceptedIllegalActions  int          `json:"accepted_illegal_actions"`
	AcceptedRepeatedActions int          `json:"accepted_repeated_actions"`
}

// Diagnostics are scalar measurements produced while making a PPO candidate.
// NumericallyStable must be set true only after the runner has checked loss,
// gradients, parameters, values, returns, and advantages for NaN/Inf. The
// evaluation package additionally rejects non-finite scalar diagnostics.
type Diagnostics struct {
	PolicyEntropy         float64 `json:"policy_entropy"`
	ApproxOldPolicyKL     float64 `json:"approx_old_policy_kl"`
	SupervisedReferenceKL float64 `json:"supervised_reference_kl"`
	ClipFraction          float64 `json:"clip_fraction"`
	CriticExplainedVar    float64 `json:"critic_explained_variance"`
	NumericallyStable     bool    `json:"numerically_stable"`
}

// Evaluation is the complete, greedy development evaluation for one policy.
// Games should be the fixed development split in any order; pairing is by
// SolutionID and reports are made stable by sorting IDs internally.
type Evaluation struct {
	Games       []GameResult `json:"games"`
	Diagnostics Diagnostics  `json:"diagnostics"`
}

// Summary is the machine-readable single-policy development report. Failed
// games contribute their real sixth guess to GuessCountHistogram but seven to
// FailureCountedMeanGuesses.
type Summary struct {
	Games                       int           `json:"games"`
	SolvedCount                 int           `json:"solved_count"`
	SolveRate                   float64       `json:"solve_rate"`
	MeanGuessesSolved           float64       `json:"mean_guesses_solved"`
	FailureCountedMeanGuesses   float64       `json:"failure_counted_mean_guesses"`
	GuessCountHistogram         [MaxTurns]int `json:"guess_count_histogram"`
	FailureCount                int           `json:"failure_count"`
	OpeningGuess                string        `json:"opening_guess"`
	OpeningActionProbability    float64       `json:"opening_action_probability"`
	AcceptedIllegalActionCount  int           `json:"accepted_illegal_action_count"`
	AcceptedRepeatedActionCount int           `json:"accepted_repeated_action_count"`
	MeanTurnPolicyEntropy       float64       `json:"mean_turn_policy_entropy"`
	Diagnostics                 Diagnostics   `json:"diagnostics"`
}

// ConfidenceInterval is a two-sided percentile bootstrap interval for a mean
// paired difference. Candidate minus baseline is used throughout, so negative
// values favour the candidate.
type ConfidenceInterval struct {
	Level   float64 `json:"level"`
	Lower   float64 `json:"lower"`
	Upper   float64 `json:"upper"`
	Samples int     `json:"samples"`
	Seed    int64   `json:"seed"`
}

// Acceptance records each mechanical promotion gate separately so a rejected
// candidate is easy to audit and the last accepted checkpoint can be retained.
type Acceptance struct {
	SolvedCountNonDecreasing        bool     `json:"solved_count_non_decreasing"`
	FailureCountedMeanImproved      bool     `json:"failure_counted_mean_improved"`
	NoAcceptedIllegalOrRepeated     bool     `json:"no_accepted_illegal_or_repeated_actions"`
	NumericallyStable               bool     `json:"numerically_stable"`
	OldPolicyKLControlled           bool     `json:"old_policy_kl_controlled"`
	SupervisedReferenceKLControlled bool     `json:"supervised_reference_kl_controlled"`
	EntropyNotCollapsed             bool     `json:"entropy_not_collapsed"`
	Accepted                        bool     `json:"accepted"`
	Reasons                         []string `json:"reasons,omitempty"`
}

// Classification is deliberately separate from promotion. A safe point
// improvement can be promoted inside the bounded run yet remain only promising
// until its paired confidence interval is entirely favourable.
type Classification string

const (
	Rejected             Classification = "rejected"
	Inconclusive         Classification = "inconclusive"
	Promising            Classification = "promising"
	ConvincinglyImproved Classification = "convincingly_improved"
)

// Comparison is the complete candidate-versus-baseline development report.
// Improved/Worsened/Unchanged compare per-solution failure-counted guesses.
type Comparison struct {
	Baseline             Summary            `json:"baseline"`
	Candidate            Summary            `json:"candidate"`
	PairedGames          int                `json:"paired_games"`
	Improved             int                `json:"improved"`
	Worsened             int                `json:"worsened"`
	Unchanged            int                `json:"unchanged"`
	NewlySolvedIDs       []string           `json:"newly_solved_ids"`
	NewlyFailedIDs       []string           `json:"newly_failed_ids"`
	PairedMeanDifference float64            `json:"paired_mean_difference"`
	PairedBootstrap95    ConfidenceInterval `json:"paired_bootstrap_95"`
	Acceptance           Acceptance         `json:"acceptance"`
	Classification       Classification     `json:"classification"`
}

// CompareOptions controls deterministic bootstrap reporting and promotion
// gates. Use DefaultCompareOptions as the normal PPO configuration; zero
// fields are filled with those cautious defaults for convenience.
type CompareOptions struct {
	BootstrapSamples         int     `json:"bootstrap_samples"`
	BootstrapSeed            int64   `json:"bootstrap_seed"`
	MaxOldPolicyKL           float64 `json:"max_old_policy_kl"`
	MaxSupervisedReferenceKL float64 `json:"max_supervised_reference_kl"`
	MinimumPolicyEntropy     float64 `json:"minimum_policy_entropy"`
}

// DefaultCompareOptions returns the fixed, cautious evaluation policy for the
// first bounded PPO pilot. The seed is explicit so reports reproduce exactly.
func DefaultCompareOptions(seed int64) CompareOptions {
	return CompareOptions{
		BootstrapSamples:         DefaultBootstrapSamples,
		BootstrapSeed:            seed,
		MaxOldPolicyKL:           DefaultMaxOldPolicyKL,
		MaxSupervisedReferenceKL: DefaultMaxSupervisedReferenceKL,
		MinimumPolicyEntropy:     DefaultMinimumPolicyEntropy,
	}
}

// Summarize validates complete authoritative games and returns their fixed
// development metrics. It does not inspect, construct, or mutate model input.
func Summarize(evaluation Evaluation) (Summary, error) {
	if err := validateDiagnostics(evaluation.Diagnostics); err != nil {
		return Summary{}, err
	}
	if len(evaluation.Games) == 0 {
		return Summary{}, errors.New("evaluation games must not be empty")
	}
	result := Summary{Games: len(evaluation.Games), Diagnostics: evaluation.Diagnostics}
	seenSolutions := make(map[string]struct{}, len(evaluation.Games))
	var solvedGuesses, failureCountedGuesses int
	var totalEntropy float64
	var entropyTurns int
	for gameIndex, game := range evaluation.Games {
		if err := validateGame(game); err != nil {
			return Summary{}, fmt.Errorf("game %d: %w", gameIndex, err)
		}
		if _, seen := seenSolutions[game.SolutionID]; seen {
			return Summary{}, fmt.Errorf("duplicate solution ID %q", game.SolutionID)
		}
		seenSolutions[game.SolutionID] = struct{}{}
		result.GuessCountHistogram[game.Guesses-1]++
		result.AcceptedIllegalActionCount += game.AcceptedIllegalActions
		result.AcceptedRepeatedActionCount += game.AcceptedRepeatedActions
		if gameIndex == 0 {
			result.OpeningGuess = game.Turns[0].Guess
			result.OpeningActionProbability = game.Turns[0].ActionProbability
		} else if game.Turns[0].Guess != result.OpeningGuess {
			return Summary{}, fmt.Errorf("opening guess for solution %q is %q, want deterministic greedy opening %q", game.SolutionID, game.Turns[0].Guess, result.OpeningGuess)
		}
		for _, turn := range game.Turns {
			totalEntropy += turn.PolicyEntropy
			entropyTurns++
		}
		if game.Solved {
			result.SolvedCount++
			solvedGuesses += game.Guesses
			failureCountedGuesses += game.Guesses
		} else {
			result.FailureCount++
			failureCountedGuesses += FailureCountedGuesses
		}
	}
	result.SolveRate = float64(result.SolvedCount) / float64(result.Games)
	if result.SolvedCount > 0 {
		result.MeanGuessesSolved = float64(solvedGuesses) / float64(result.SolvedCount)
	}
	result.FailureCountedMeanGuesses = float64(failureCountedGuesses) / float64(result.Games)
	result.MeanTurnPolicyEntropy = totalEntropy / float64(entropyTurns)
	return result, nil
}

// Compare validates and pairs two evaluations of exactly the same development
// population. It calculates the candidate-minus-baseline difference, a stable
// bootstrap interval, promotion gates, and a conservative classification.
func Compare(baseline, candidate Evaluation, options CompareOptions) (Comparison, error) {
	options, err := normalizeOptions(options)
	if err != nil {
		return Comparison{}, err
	}
	baselineSummary, err := Summarize(baseline)
	if err != nil {
		return Comparison{}, fmt.Errorf("summarize baseline: %w", err)
	}
	candidateSummary, err := Summarize(candidate)
	if err != nil {
		return Comparison{}, fmt.Errorf("summarize candidate: %w", err)
	}
	baselineByID, err := gamesByID(baseline.Games)
	if err != nil {
		return Comparison{}, fmt.Errorf("baseline games: %w", err)
	}
	candidateByID, err := gamesByID(candidate.Games)
	if err != nil {
		return Comparison{}, fmt.Errorf("candidate games: %w", err)
	}
	if len(baselineByID) != len(candidateByID) {
		return Comparison{}, fmt.Errorf("candidate games %d do not match baseline games %d", len(candidateByID), len(baselineByID))
	}
	ids := make([]string, 0, len(baselineByID))
	for id := range baselineByID {
		if _, found := candidateByID[id]; !found {
			return Comparison{}, fmt.Errorf("candidate has no game for baseline solution ID %q", id)
		}
		ids = append(ids, id)
	}
	sort.Strings(ids)

	comparison := Comparison{Baseline: baselineSummary, Candidate: candidateSummary, PairedGames: len(ids)}
	differences := make([]float64, 0, len(ids))
	for _, id := range ids {
		base := baselineByID[id]
		trial := candidateByID[id]
		baseScore := failureCountedGuess(base)
		candidateScore := failureCountedGuess(trial)
		difference := float64(candidateScore - baseScore)
		differences = append(differences, difference)
		switch {
		case difference < 0:
			comparison.Improved++
		case difference > 0:
			comparison.Worsened++
		default:
			comparison.Unchanged++
		}
		if !base.Solved && trial.Solved {
			comparison.NewlySolvedIDs = append(comparison.NewlySolvedIDs, id)
		}
		if base.Solved && !trial.Solved {
			comparison.NewlyFailedIDs = append(comparison.NewlyFailedIDs, id)
		}
	}
	comparison.PairedMeanDifference = mean(differences)
	comparison.PairedBootstrap95 = PairedBootstrap95(differences, options.BootstrapSeed, options.BootstrapSamples)
	comparison.Acceptance = acceptance(baselineSummary, candidateSummary, options)
	comparison.Classification = classify(comparison)
	return comparison, nil
}

// PairedBootstrap95 returns a deterministic percentile bootstrap confidence
// interval over paired candidate-minus-baseline differences. The small local
// SplitMix64 implementation fixes the sequence independently of Go's random
// package implementation and makes JSON reports reproducible across hosts.
func PairedBootstrap95(differences []float64, seed int64, samples int) ConfidenceInterval {
	interval := ConfidenceInterval{Level: 0.95, Samples: samples, Seed: seed}
	if len(differences) == 0 || samples <= 0 {
		return interval
	}
	means := make([]float64, samples)
	state := uint64(seed)
	for sample := range means {
		var sum float64
		for range differences {
			state = splitMix64(state)
			sum += differences[int(state%uint64(len(differences)))]
		}
		means[sample] = sum / float64(len(differences))
	}
	sort.Float64s(means)
	interval.Lower = percentile(means, 0.025)
	interval.Upper = percentile(means, 0.975)
	return interval
}

func normalizeOptions(options CompareOptions) (CompareOptions, error) {
	if options.BootstrapSamples == 0 {
		options.BootstrapSamples = DefaultBootstrapSamples
	}
	if options.MaxOldPolicyKL == 0 {
		options.MaxOldPolicyKL = DefaultMaxOldPolicyKL
	}
	if options.MaxSupervisedReferenceKL == 0 {
		options.MaxSupervisedReferenceKL = DefaultMaxSupervisedReferenceKL
	}
	if options.MinimumPolicyEntropy == 0 {
		options.MinimumPolicyEntropy = DefaultMinimumPolicyEntropy
	}
	if options.BootstrapSamples <= 0 {
		return CompareOptions{}, fmt.Errorf("bootstrap samples must be positive, got %d", options.BootstrapSamples)
	}
	if !finite(options.MaxOldPolicyKL) || options.MaxOldPolicyKL < 0 {
		return CompareOptions{}, fmt.Errorf("maximum old-policy KL must be finite and non-negative, got %v", options.MaxOldPolicyKL)
	}
	if !finite(options.MaxSupervisedReferenceKL) || options.MaxSupervisedReferenceKL < 0 {
		return CompareOptions{}, fmt.Errorf("maximum supervised-reference KL must be finite and non-negative, got %v", options.MaxSupervisedReferenceKL)
	}
	if !finite(options.MinimumPolicyEntropy) || options.MinimumPolicyEntropy < 0 {
		return CompareOptions{}, fmt.Errorf("minimum policy entropy must be finite and non-negative, got %v", options.MinimumPolicyEntropy)
	}
	return options, nil
}

func validateGame(result GameResult) error {
	if result.SolutionID == "" {
		return errors.New("solution ID must not be empty")
	}
	if result.Guesses < 1 || result.Guesses > MaxTurns {
		return fmt.Errorf("guesses %d outside fixed range 1..%d", result.Guesses, MaxTurns)
	}
	if !result.Solved && result.Guesses != MaxTurns {
		return fmt.Errorf("unsolved game has %d guesses, want %d", result.Guesses, MaxTurns)
	}
	if len(result.Turns) != result.Guesses {
		return fmt.Errorf("turn records %d, want %d", len(result.Turns), result.Guesses)
	}
	if result.AcceptedIllegalActions < 0 || result.AcceptedRepeatedActions < 0 {
		return errors.New("accepted illegal and repeated action counts must be non-negative")
	}
	for index, turn := range result.Turns {
		if turn.Turn != index+1 {
			return fmt.Errorf("turn record %d has turn %d, want %d", index, turn.Turn, index+1)
		}
		if turn.Guess == "" {
			return fmt.Errorf("turn %d guess must not be empty", turn.Turn)
		}
		if turn.ActionID < 0 {
			return fmt.Errorf("turn %d action ID %d must be non-negative", turn.Turn, turn.ActionID)
		}
		if !finite(turn.ActionProbability) || turn.ActionProbability < 0 || turn.ActionProbability > 1 {
			return fmt.Errorf("turn %d action probability %v outside [0,1]", turn.Turn, turn.ActionProbability)
		}
		if !finite(turn.PolicyEntropy) || turn.PolicyEntropy < 0 {
			return fmt.Errorf("turn %d policy entropy %v must be finite and non-negative", turn.Turn, turn.PolicyEntropy)
		}
		if !finite(turn.Reward) {
			return fmt.Errorf("turn %d reward %v must be finite", turn.Turn, turn.Reward)
		}
		if turn.Terminal != (index == len(result.Turns)-1) {
			return fmt.Errorf("turn %d terminal=%t does not match completed trajectory", turn.Turn, turn.Terminal)
		}
	}
	return nil
}

func validateDiagnostics(d Diagnostics) error {
	for name, value := range map[string]float64{
		"policy entropy":            d.PolicyEntropy,
		"approximate old-policy KL": d.ApproxOldPolicyKL,
		"supervised-reference KL":   d.SupervisedReferenceKL,
		"clip fraction":             d.ClipFraction,
		"critic explained variance": d.CriticExplainedVar,
	} {
		if !finite(value) {
			return fmt.Errorf("%s must be finite, got %v", name, value)
		}
	}
	if d.PolicyEntropy < 0 {
		return fmt.Errorf("policy entropy %v must be non-negative", d.PolicyEntropy)
	}
	if d.ApproxOldPolicyKL < 0 || d.SupervisedReferenceKL < 0 {
		return errors.New("KL diagnostics must be non-negative")
	}
	if d.ClipFraction < 0 || d.ClipFraction > 1 {
		return fmt.Errorf("clip fraction %v outside [0,1]", d.ClipFraction)
	}
	if d.CriticExplainedVar > 1 {
		return fmt.Errorf("critic explained variance %v exceeds 1", d.CriticExplainedVar)
	}
	return nil
}

func gamesByID(games []GameResult) (map[string]GameResult, error) {
	byID := make(map[string]GameResult, len(games))
	for _, game := range games {
		if _, found := byID[game.SolutionID]; found {
			return nil, fmt.Errorf("duplicate solution ID %q", game.SolutionID)
		}
		byID[game.SolutionID] = game
	}
	return byID, nil
}

func failureCountedGuess(game GameResult) int {
	if game.Solved {
		return game.Guesses
	}
	return FailureCountedGuesses
}

func acceptance(baseline, candidate Summary, options CompareOptions) Acceptance {
	result := Acceptance{
		SolvedCountNonDecreasing:        candidate.SolvedCount >= baseline.SolvedCount,
		FailureCountedMeanImproved:      candidate.FailureCountedMeanGuesses < baseline.FailureCountedMeanGuesses,
		NoAcceptedIllegalOrRepeated:     candidate.AcceptedIllegalActionCount == 0 && candidate.AcceptedRepeatedActionCount == 0,
		NumericallyStable:               candidate.Diagnostics.NumericallyStable,
		OldPolicyKLControlled:           candidate.Diagnostics.ApproxOldPolicyKL <= options.MaxOldPolicyKL,
		SupervisedReferenceKLControlled: candidate.Diagnostics.SupervisedReferenceKL <= options.MaxSupervisedReferenceKL,
		EntropyNotCollapsed:             candidate.Diagnostics.PolicyEntropy >= options.MinimumPolicyEntropy,
	}
	if !result.SolvedCountNonDecreasing {
		result.Reasons = append(result.Reasons, "development solved count decreased")
	}
	if !result.FailureCountedMeanImproved {
		result.Reasons = append(result.Reasons, "failure-counted mean guesses did not strictly improve")
	}
	if !result.NoAcceptedIllegalOrRepeated {
		result.Reasons = append(result.Reasons, "candidate accepted illegal or repeated actions")
	}
	if !result.NumericallyStable {
		result.Reasons = append(result.Reasons, "candidate reported numerical instability")
	}
	if !result.OldPolicyKLControlled {
		result.Reasons = append(result.Reasons, "old-policy KL exceeded configured limit")
	}
	if !result.SupervisedReferenceKLControlled {
		result.Reasons = append(result.Reasons, "supervised-reference KL exceeded configured limit")
	}
	if !result.EntropyNotCollapsed {
		result.Reasons = append(result.Reasons, "policy entropy fell below configured floor")
	}
	result.Accepted = result.SolvedCountNonDecreasing && result.FailureCountedMeanImproved &&
		result.NoAcceptedIllegalOrRepeated && result.NumericallyStable && result.OldPolicyKLControlled &&
		result.SupervisedReferenceKLControlled && result.EntropyNotCollapsed
	return result
}

func classify(comparison Comparison) Classification {
	// Safety/integrity violations and directly worse deployment metrics are
	// rejected rather than described as statistical uncertainty.
	if !comparison.Acceptance.NoAcceptedIllegalOrRepeated || !comparison.Acceptance.NumericallyStable ||
		!comparison.Acceptance.OldPolicyKLControlled || !comparison.Acceptance.SupervisedReferenceKLControlled ||
		!comparison.Acceptance.EntropyNotCollapsed || !comparison.Acceptance.SolvedCountNonDecreasing ||
		comparison.PairedMeanDifference > 0 {
		return Rejected
	}
	if !comparison.Acceptance.FailureCountedMeanImproved {
		return Inconclusive
	}
	if comparison.PairedBootstrap95.Upper < 0 {
		return ConvincinglyImproved
	}
	if comparison.PairedBootstrap95.Lower < 0 {
		return Promising
	}
	return Inconclusive
}

func mean(values []float64) float64 {
	var sum float64
	for _, value := range values {
		sum += value
	}
	return sum / float64(len(values))
}

func percentile(sorted []float64, fraction float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	index := int(math.Ceil(fraction*float64(len(sorted)))) - 1
	if index < 0 {
		index = 0
	}
	if index >= len(sorted) {
		index = len(sorted) - 1
	}
	return sorted[index]
}

func splitMix64(value uint64) uint64 {
	value += 0x9e3779b97f4a7c15
	value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9
	value = (value ^ (value >> 27)) * 0x94d049bb133111eb
	return value ^ (value >> 31)
}

func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }
