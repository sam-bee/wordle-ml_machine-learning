package rl

import (
	"errors"
	"fmt"

	"github.com/sam-bee/wordle-ml_game-engine/game"
	"github.com/sam-bee/wordle-ml_game-engine/words"
	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

var (
	// ErrInvalidActionID reports an action outside the frozen actor vocabulary.
	ErrInvalidActionID = errors.New("invalid action ID")
	// ErrActionUnavailable reports an action excluded by the exact mask returned
	// in Observation. In this environment all vocabulary actions are legal to
	// the engine initially; an unavailable action is therefore a prior accepted
	// guess.
	ErrActionUnavailable = errors.New("action is unavailable")
)

// Observation is exactly the information made available to the actor for one
// environment position. It deliberately has no hidden-answer field. The
// availability mask is distinct from Inputs.RemainingActionMask: the latter is
// a learned candidate feature, while this mask is applied to policy logits to
// prevent unavailable actions from receiving probability mass.
type Observation struct {
	Inputs              modelstate.Inputs
	AvailableActionMask []float32
	Turn                int
}

// EpisodeMetadata is environment-only trajectory metadata. It must not be
// used to construct model inputs. SolutionID is the canonical frozen solution
// identifier, never its hidden word.
type EpisodeMetadata struct {
	EpisodeID  int64 `json:"episode_id"`
	SolutionID int   `json:"solution_id"`
}

// Transition describes the authoritative result of one accepted game-engine
// guess. Observation is the pre-action actor input and exact availability
// mask. Turn is zero-based, matching Inputs.Turn and PPO's transition order.
type Transition struct {
	Observation Observation     `json:"-"`
	ActionID    int             `json:"action_id"`
	Reward      float32         `json:"reward"`
	Terminal    bool            `json:"terminal"`
	Solved      bool            `json:"solved"`
	Turn        int             `json:"turn"`
	Feedback    string          `json:"feedback"`
	Metadata    EpisodeMetadata `json:"metadata"`
}

// Environment adapts the authoritative game engine to PPO. It always creates
// the engine as game.NewState(hidden, vocab.Actions(), vocab.Solutions()), so
// Wordle transition, feedback, termination, and valid-guess legality come from
// the existing engine rather than this package.
type Environment struct {
	vocabulary *vocabulary.Vocabulary
	encoder    *modelstate.Encoder
	state      *game.State
	metadata   EpisodeMetadata
}

// NewEnvironment creates a single six-turn environment episode. hiddenSolution
// is environment-side only: it is passed directly to the game engine and is
// never included in Observation.
func NewEnvironment(v *vocabulary.Vocabulary, hiddenSolution string, episodeID int64) (*Environment, error) {
	if v == nil {
		return nil, errors.New("vocabulary must not be nil")
	}
	solutionID, found := v.SolutionID(hiddenSolution)
	if !found {
		return nil, fmt.Errorf("hidden solution %q is not in the frozen solution vocabulary", hiddenSolution)
	}
	encoder, err := modelstate.NewEncoder(v)
	if err != nil {
		return nil, err
	}
	valid := make([]words.Word, vocabulary.NumActions)
	for actionID, word := range v.Actions() {
		valid[actionID] = words.Word(word)
	}
	candidates := make([]words.Word, vocabulary.NumSolutions)
	for solutionID, word := range v.Solutions() {
		candidates[solutionID] = words.Word(word)
	}
	state, err := game.NewState(words.Word(hiddenSolution), valid, candidates)
	if err != nil {
		return nil, fmt.Errorf("create authoritative Wordle state: %w", err)
	}
	return &Environment{
		vocabulary: v,
		encoder:    encoder,
		state:      state,
		metadata: EpisodeMetadata{
			EpisodeID:  episodeID,
			SolutionID: solutionID,
		},
	}, nil
}

// Metadata returns the environment-side episode identifiers. Callers that
// feed the actor must use Observation, not this metadata.
func (e *Environment) Metadata() EpisodeMetadata { return e.metadata }

// Finished reports the game engine's authoritative terminal state.
func (e *Environment) Finished() bool { return e.state.Finished() }

// Observation derives the actor input and exact action mask from the current
// game-engine state. Observations are only available before a legal turn; a
// terminal six-turn state cannot be encoded because the actor has turns 0..5.
func (e *Environment) Observation() (Observation, error) {
	if e.state.Finished() {
		return Observation{}, game.ErrFinished
	}
	candidates := e.state.CandidateSolutions()
	bits := make([]byte, modelstate.CandidateBitsetBytes)
	for _, candidate := range candidates {
		solutionID, found := e.vocabulary.SolutionID(string(candidate))
		if !found {
			return Observation{}, fmt.Errorf("engine candidate %q is not in the frozen solution vocabulary", candidate)
		}
		bits[solutionID/8] |= 1 << uint(solutionID%8)
	}
	inputs, err := e.encoder.Encode(bits, e.state.TurnCount())
	if err != nil {
		return Observation{}, err
	}
	available := make([]float32, vocabulary.NumActions)
	for actionID := range available {
		available[actionID] = 1
	}
	// The game is initialized with every frozen action as a valid guess. The
	// only dynamically unavailable actions are exactly the accepted history.
	for _, prior := range e.state.History() {
		actionID, found := e.vocabulary.ActionID(string(prior.Guess))
		if !found {
			return Observation{}, fmt.Errorf("engine history guess %q is not in the frozen action vocabulary", prior.Guess)
		}
		available[actionID] = 0
	}
	return Observation{Inputs: cloneInputs(inputs), AvailableActionMask: available, Turn: e.state.TurnCount()}, nil
}

// Step applies one available action through the authoritative game engine and
// returns its exact reward: -0.05 for a non-terminal guess, +1.00 for a solve,
// and -1.00 for an unsolved sixth guess.
func (e *Environment) Step(actionID int) (Transition, error) {
	if e.state.Finished() {
		return Transition{}, game.ErrFinished
	}
	observation, err := e.Observation()
	if err != nil {
		return Transition{}, err
	}
	if actionID < 0 || actionID >= vocabulary.NumActions {
		return Transition{}, fmt.Errorf("%w: %d", ErrInvalidActionID, actionID)
	}
	if observation.AvailableActionMask[actionID] == 0 {
		return Transition{}, fmt.Errorf("%w: %d", ErrActionUnavailable, actionID)
	}
	guess, _ := e.vocabulary.ActionWord(actionID)
	feedback, err := e.state.ApplyGuess(words.Word(guess))
	if err != nil {
		// Engine ApplyGuess remains authoritative for accepted actions and all
		// transition details. A failure here is an invariant violation, not a
		// replacement implementation of the engine's rules.
		return Transition{}, fmt.Errorf("authoritative engine apply action %d %q: %w", actionID, guess, err)
	}
	transition := Transition{
		Observation: observation,
		ActionID:    actionID,
		Terminal:    e.state.Finished(),
		Solved:      e.state.Solved(),
		Turn:        observation.Turn,
		Feedback:    feedback.String(),
		Metadata:    e.metadata,
	}
	switch {
	case transition.Solved:
		transition.Reward = 1.00
	case transition.Terminal:
		transition.Reward = -1.00
	default:
		transition.Reward = -0.05
	}
	return transition, nil
}

func cloneInputs(inputs modelstate.Inputs) modelstate.Inputs {
	return modelstate.Inputs{
		CandidateMask:       append([]float32(nil), inputs.CandidateMask...),
		CandidateStats:      append([]float32(nil), inputs.CandidateStats...),
		Turn:                inputs.Turn,
		RemainingActionMask: append([]float32(nil), inputs.RemainingActionMask...),
	}
}
