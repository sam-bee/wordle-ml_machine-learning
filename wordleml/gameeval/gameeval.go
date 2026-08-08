// Package gameeval evaluates a scored policy through the authoritative Wordle
// game engine.
package gameeval

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/sam-bee/wordle-ml_game-engine/game"
	"github.com/sam-bee/wordle-ml_game-engine/words"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

var (
	// ErrInvalidScores indicates that a scorer did not return one finite score
	// for each word in the fixed action vocabulary.
	ErrInvalidScores = errors.New("invalid policy scores")
	// ErrNoLegalAction indicates that a game position has no legal action. This
	// should be impossible for the fixed action vocabulary and game engine.
	ErrNoLegalAction = errors.New("no legal action")
)

// ScoreFunc returns unmasked logits in canonical vocabulary action-ID order.
// The evaluator applies the game legality mask and performs deterministic
// greedy selection itself. Equal scores select the lower action ID.
type ScoreFunc func(context.Context, Position) ([]float32, error)

// Position is the production model input for one game position. Its slices are
// private copies for a scorer; mutating them cannot alter the evaluator.
type Position struct {
	Inputs              modelstate.Inputs
	AvailableActionMask []float32
	Turn                int
	CandidateSolutions  []string
	History             []HistoryTurn
}

// HistoryTurn is one prior authoritative game transition exposed to a scorer.
type HistoryTurn struct {
	Guess    string `json:"guess"`
	Feedback string `json:"feedback"`
}

// Config defines a fixed evaluation population and its scorer. Omitted valid
// guesses and candidate solutions default to the complete frozen vocabulary.
type Config struct {
	Vocabulary         *vocabulary.Vocabulary
	Score              ScoreFunc
	ValidGuesses       []string
	CandidateSolutions []string
}

// Evaluator applies an injected policy scorer to authoritative games.
type Evaluator struct {
	vocabulary *vocabulary.Vocabulary
	encoder    *modelstate.Encoder
	score      ScoreFunc
	valid      []words.Word
	candidates []words.Word
	validIDs   map[int]struct{}
}

// TurnResult is one JSONL-ready transition in a completed game trajectory.
type TurnResult struct {
	Turn                int    `json:"turn"`
	RawTopActionID      int    `json:"raw_top_action_id"`
	RawTopGuess         string `json:"raw_top_guess"`
	Guess               string `json:"guess"`
	Feedback            string `json:"feedback"`
	ShortlistSizeBefore int    `json:"shortlist_size_before"`
	ShortlistSizeAfter  int    `json:"shortlist_size_after"`
}

// GameResult is one JSONL-ready complete game result. The selection counters
// describe raw unmasked score preferences which the evaluator suppressed before
// calling the game engine; an engine-invalid accepted move is always an error.
type GameResult struct {
	Solution                   string       `json:"solution"`
	Solved                     bool         `json:"solved"`
	Guesses                    int          `json:"guesses"`
	Failure                    string       `json:"failure,omitempty"`
	InvalidSelections          int          `json:"invalid_selections"`
	SuppressedRawTopSelections int          `json:"suppressed_raw_top_selections"`
	RepeatedSelections         int          `json:"repeated_selections"`
	Turns                      []TurnResult `json:"turns"`
}

// Summary aggregates deterministic full-game evaluation results. Element i of
// GuessCountDistribution is the number of games that took i+1 guesses. Failed
// games use their sixth accepted guess and therefore contribute to element 5.
type Summary struct {
	Games                      int                `json:"games"`
	Solved                     int                `json:"solved"`
	SolvedFraction             float64            `json:"solved_fraction"`
	MeanGuesses                float64            `json:"mean_guesses"`
	GuessCountDistribution     [game.MaxTurns]int `json:"guess_count_distribution"`
	Failures                   int                `json:"failures"`
	FailedSolutions            []string           `json:"failed_solutions"`
	InvalidSelections          int                `json:"invalid_selections"`
	SuppressedRawTopSelections int                `json:"suppressed_raw_top_selections"`
	RepeatedSelections         int                `json:"repeated_selections"`
}

// Evaluation is a deterministic result suitable for writing one GameResult per
// line to validation-games.jsonl plus its Summary to final metrics.
type Evaluation struct {
	Summary Summary      `json:"summary"`
	Games   []GameResult `json:"games"`
}

// New validates a fixed evaluation configuration.
func New(config Config) (*Evaluator, error) {
	if config.Vocabulary == nil {
		return nil, errors.New("vocabulary must not be nil")
	}
	if config.Score == nil {
		return nil, errors.New("score function must not be nil")
	}
	encoder, err := modelstate.NewEncoder(config.Vocabulary)
	if err != nil {
		return nil, err
	}
	validWords := config.ValidGuesses
	if len(validWords) == 0 {
		validWords = config.Vocabulary.Actions()
	}
	valid, validIDs, err := actionWords(config.Vocabulary, validWords, "valid guesses")
	if err != nil {
		return nil, err
	}
	candidateWords := config.CandidateSolutions
	if len(candidateWords) == 0 {
		candidateWords = config.Vocabulary.Solutions()
	}
	candidates, err := solutionWords(config.Vocabulary, candidateWords, validIDs)
	if err != nil {
		return nil, err
	}
	return &Evaluator{
		vocabulary: config.Vocabulary,
		encoder:    encoder,
		score:      config.Score,
		valid:      valid,
		candidates: candidates,
		validIDs:   validIDs,
	}, nil
}

// Evaluate plays the supplied hidden solutions in their supplied stable order.
// They must belong to this evaluator's candidate population; this prevents a
// caller from accidentally evaluating a mismatched split or vocabulary.
func (e *Evaluator) Evaluate(ctx context.Context, solutions []string) (Evaluation, error) {
	if len(solutions) == 0 {
		return Evaluation{}, errors.New("evaluation solutions must not be empty")
	}
	result := Evaluation{Games: make([]GameResult, 0, len(solutions))}
	for gameIndex, solution := range solutions {
		if err := ctx.Err(); err != nil {
			return Evaluation{}, err
		}
		gameResult, err := e.play(ctx, solution)
		if err != nil {
			return Evaluation{}, fmt.Errorf("evaluate game %d solution %q: %w", gameIndex, solution, err)
		}
		result.Games = append(result.Games, gameResult)
		accumulate(&result.Summary, gameResult)
	}
	result.Summary.Games = len(result.Games)
	result.Summary.SolvedFraction = float64(result.Summary.Solved) / float64(result.Summary.Games)
	var totalGuesses int
	for _, gameResult := range result.Games {
		totalGuesses += gameResult.Guesses
	}
	result.Summary.MeanGuesses = float64(totalGuesses) / float64(result.Summary.Games)
	return result, nil
}

func (e *Evaluator) play(ctx context.Context, solutionText string) (GameResult, error) {
	solutionID, found := e.vocabulary.SolutionID(solutionText)
	if !found {
		return GameResult{}, fmt.Errorf("solution is not in the frozen solution vocabulary")
	}
	solutionActionID, _ := e.vocabulary.SolutionActionID(solutionID)
	if _, found := e.validIDs[solutionActionID]; !found {
		return GameResult{}, fmt.Errorf("solution is not a valid guess")
	}
	if !containsWord(e.candidates, words.Word(solutionText)) {
		return GameResult{}, fmt.Errorf("solution is not in the evaluator candidate population")
	}
	state, err := game.NewState(words.Word(solutionText), e.valid, e.candidates)
	if err != nil {
		return GameResult{}, err
	}
	result := GameResult{Solution: solutionText, Turns: make([]TurnResult, 0, game.MaxTurns)}
	for !state.Finished() {
		if err := ctx.Err(); err != nil {
			return GameResult{}, err
		}
		position, err := e.position(state)
		if err != nil {
			return GameResult{}, err
		}
		// Give the scorer a private copy: policy inference must not be able to
		// alter the legality mask that the evaluator applies afterwards.
		scores, err := e.score(ctx, clonePosition(position))
		if err != nil {
			return GameResult{}, fmt.Errorf("score turn %d: %w", state.TurnCount()+1, err)
		}
		rawTopID, selectedID, err := selectAction(scores, position.AvailableActionMask)
		if err != nil {
			return GameResult{}, err
		}
		if position.AvailableActionMask[rawTopID] == 0 {
			result.SuppressedRawTopSelections++
			if _, valid := e.validIDs[rawTopID]; !valid {
				result.InvalidSelections++
			} else {
				result.RepeatedSelections++
			}
		}
		rawWord, _ := e.vocabulary.ActionWord(rawTopID)
		guess, _ := e.vocabulary.ActionWord(selectedID)
		shortlistBefore := len(state.CandidateSolutions())
		feedback, err := state.ApplyGuess(words.Word(guess))
		if err != nil {
			return GameResult{}, fmt.Errorf("selected action %d %q was not legal: %w", selectedID, guess, err)
		}
		result.Turns = append(result.Turns, TurnResult{
			Turn:                state.TurnCount(),
			RawTopActionID:      rawTopID,
			RawTopGuess:         rawWord,
			Guess:               guess,
			Feedback:            feedback.String(),
			ShortlistSizeBefore: shortlistBefore,
			ShortlistSizeAfter:  len(state.CandidateSolutions()),
		})
	}
	result.Solved = state.Solved()
	result.Guesses = state.TurnCount()
	if !result.Solved {
		result.Failure = "unsolved_after_six_guesses"
	}
	return result, nil
}

func (e *Evaluator) position(state *game.State) (Position, error) {
	candidates := state.CandidateSolutions()
	bits := make([]byte, modelstate.CandidateBitsetBytes)
	candidateTexts := make([]string, len(candidates))
	for index, candidate := range candidates {
		solutionID, found := e.vocabulary.SolutionID(string(candidate))
		if !found {
			return Position{}, fmt.Errorf("engine candidate %q is not a solution vocabulary word", candidate)
		}
		bits[solutionID/8] |= 1 << uint(solutionID%8)
		candidateTexts[index] = string(candidate)
	}
	inputs, err := e.encoder.Encode(bits, state.TurnCount())
	if err != nil {
		return Position{}, err
	}
	available := make([]float32, vocabulary.NumActions)
	for actionID := range e.validIDs {
		available[actionID] = 1
	}
	history := state.History()
	historyTurns := make([]HistoryTurn, len(history))
	for index, prior := range history {
		actionID, found := e.vocabulary.ActionID(string(prior.Guess))
		if !found {
			return Position{}, fmt.Errorf("engine history guess %q is not an action vocabulary word", prior.Guess)
		}
		available[actionID] = 0
		historyTurns[index] = HistoryTurn{Guess: string(prior.Guess), Feedback: prior.Feedback.String()}
	}
	return Position{
		Inputs:              inputs,
		AvailableActionMask: available,
		Turn:                state.TurnCount(),
		CandidateSolutions:  candidateTexts,
		History:             historyTurns,
	}, nil
}

func selectAction(scores, available []float32) (rawTopID, selectedID int, err error) {
	if len(scores) != vocabulary.NumActions {
		return 0, 0, fmt.Errorf("%w: got %d scores, want %d", ErrInvalidScores, len(scores), vocabulary.NumActions)
	}
	if len(available) != vocabulary.NumActions {
		return 0, 0, fmt.Errorf("available action mask has %d values, want %d", len(available), vocabulary.NumActions)
	}
	rawTopID, selectedID = -1, -1
	var rawTop, selected float32
	for actionID, score := range scores {
		if math.IsNaN(float64(score)) || math.IsInf(float64(score), 0) {
			return 0, 0, fmt.Errorf("%w: score at action %d is not finite", ErrInvalidScores, actionID)
		}
		if rawTopID < 0 || score > rawTop {
			rawTopID, rawTop = actionID, score
		}
		if available[actionID] != 0 && (selectedID < 0 || score > selected) {
			selectedID, selected = actionID, score
		}
	}
	if selectedID < 0 {
		return 0, 0, ErrNoLegalAction
	}
	return rawTopID, selectedID, nil
}

func clonePosition(position Position) Position {
	copyInputs := position.Inputs
	copyInputs.CandidateMask = append([]float32(nil), position.Inputs.CandidateMask...)
	copyInputs.CandidateStats = append([]float32(nil), position.Inputs.CandidateStats...)
	copyInputs.RemainingActionMask = append([]float32(nil), position.Inputs.RemainingActionMask...)
	return Position{
		Inputs:              copyInputs,
		AvailableActionMask: append([]float32(nil), position.AvailableActionMask...),
		Turn:                position.Turn,
		CandidateSolutions:  append([]string(nil), position.CandidateSolutions...),
		History:             append([]HistoryTurn(nil), position.History...),
	}
}

func actionWords(v *vocabulary.Vocabulary, values []string, name string) ([]words.Word, map[int]struct{}, error) {
	if len(values) == 0 {
		return nil, nil, fmt.Errorf("%s must not be empty", name)
	}
	result := make([]words.Word, len(values))
	ids := make(map[int]struct{}, len(values))
	for index, value := range values {
		actionID, found := v.ActionID(value)
		if !found {
			return nil, nil, fmt.Errorf("%s[%d] %q is not an action vocabulary word", name, index, value)
		}
		if _, duplicate := ids[actionID]; duplicate {
			return nil, nil, fmt.Errorf("%s contains duplicate %q", name, value)
		}
		ids[actionID] = struct{}{}
		result[index] = words.Word(value)
	}
	return result, ids, nil
}

func solutionWords(v *vocabulary.Vocabulary, values []string, validIDs map[int]struct{}) ([]words.Word, error) {
	if len(values) == 0 {
		return nil, errors.New("candidate solutions must not be empty")
	}
	result := make([]words.Word, len(values))
	seen := make(map[int]struct{}, len(values))
	for index, value := range values {
		solutionID, found := v.SolutionID(value)
		if !found {
			return nil, fmt.Errorf("candidate solutions[%d] %q is not a solution vocabulary word", index, value)
		}
		if _, duplicate := seen[solutionID]; duplicate {
			return nil, fmt.Errorf("candidate solutions contains duplicate %q", value)
		}
		actionID, _ := v.SolutionActionID(solutionID)
		if _, valid := validIDs[actionID]; !valid {
			return nil, fmt.Errorf("candidate solution %q is not a valid guess", value)
		}
		seen[solutionID] = struct{}{}
		result[index] = words.Word(value)
	}
	return result, nil
}

func containsWord(values []words.Word, target words.Word) bool {
	for _, value := range values {
		if value == target {
			return true
		}
	}
	return false
}

func accumulate(summary *Summary, result GameResult) {
	if result.Solved {
		summary.Solved++
	} else {
		summary.Failures++
		summary.FailedSolutions = append(summary.FailedSolutions, result.Solution)
	}
	if result.Guesses > 0 && result.Guesses <= game.MaxTurns {
		summary.GuessCountDistribution[result.Guesses-1]++
	}
	summary.InvalidSelections += result.InvalidSelections
	summary.SuppressedRawTopSelections += result.SuppressedRawTopSelections
	summary.RepeatedSelections += result.RepeatedSelections
}
