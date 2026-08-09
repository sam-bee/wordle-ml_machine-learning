package ppo

import (
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sort"

	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	rlenv "github.com/sam-bee/wordle-ml_machine-learning/rl"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

// CollectorConfig controls one disposable on-policy rollout batch.
type CollectorConfig struct {
	AnswerPool    []string
	Games         int
	Balanced      bool
	ParallelGames int
	Seed          int64
	Gamma         float64
	GAELambda     float64
}

// TrajectoryTransition retains every quantity needed to recompute the PPO
// actor ratio with the exact policy mask. ModelInputs are also the critic's
// complete four-tensor state representation; no reduced critic-only encoding
// is retained.
type TrajectoryTransition struct {
	ModelInputs  modelstate.Inputs `json:"-"`
	Availability []float32         `json:"-"`
	ActionID     int32             `json:"action_id"`
	OldLogProb   float64           `json:"old_action_log_probability"`
	OldValue     float64           `json:"old_value"`
	Reward       float64           `json:"reward"`
	Return       float64           `json:"return"`
	Advantage    float64           `json:"advantage"`
	Terminal     bool              `json:"terminal"`
	EpisodeID    int64             `json:"episode_id"`
	Turn         int               `json:"turn"`
	SolutionID   int               `json:"solution_id"`
	Solved       bool              `json:"solved"`
}

// RolloutEpisode is one completed on-policy Wordle game.
type RolloutEpisode struct {
	EpisodeID          int64   `json:"episode_id"`
	SolutionID         int     `json:"solution_id"`
	Solved             bool    `json:"solved"`
	Guesses            int     `json:"guesses"`
	Return             float64 `json:"return"`
	OpeningAction      int     `json:"opening_action"`
	OpeningProbability float64 `json:"opening_action_probability"`
}

// RolloutMetrics are the scalar telemetry derived before optimisation.
type RolloutMetrics struct {
	EpisodeReturnMean            float64 `json:"episode_return_mean"`
	SolveRate                    float64 `json:"solve_rate"`
	MeanGuessesSolved            float64 `json:"mean_guesses_solved"`
	FailureCountedMeanGuesses    float64 `json:"failure_counted_mean_guesses"`
	PolicyEntropy                float64 `json:"policy_entropy"`
	SupervisedReferenceKL        float64 `json:"supervised_reference_kl"`
	AdvantageMeanBeforeNormalize float64 `json:"advantage_mean_before_normalize"`
	AdvantageStdBeforeNormalize  float64 `json:"advantage_std_before_normalize"`
	AdvantageMean                float64 `json:"advantage_mean"`
	AdvantageStd                 float64 `json:"advantage_std"`
	ReturnMean                   float64 `json:"return_mean"`
	StepRewardMean               float64 `json:"step_reward_mean"`
	RolloutGames                 int     `json:"rollout_games"`
	RolloutSteps                 int     `json:"rollout_steps"`
	IllegalOrRepeatedActionCount int     `json:"illegal_or_repeated_action_count"`
	OpeningAction                int     `json:"opening_action"`
	OpeningActionProbability     float64 `json:"opening_action_probability"`
}

// Rollout is discarded after exactly one PPO iteration. Critic warm-up uses
// the same complete transition contract, including encoded inputs and masks.
type Rollout struct {
	Transitions []TrajectoryTransition `json:"-"`
	Episodes    []RolloutEpisode       `json:"episodes"`
	Metrics     RolloutMetrics         `json:"metrics"`
}

// CollectRollout freezes oldPolicy by convention: this function performs only
// inference on it. The caller creates a separate candidate session for all
// updates, so every action is genuinely on-policy with these stored stats.
func CollectRollout(vocab *vocabulary.Vocabulary, oldPolicy, critic, supervisedReference interface{}, config CollectorConfig) (Rollout, error) {
	actor, ok := oldPolicy.(*ActorSession)
	if !ok || actor == nil {
		return Rollout{}, errors.New("collector requires an old-policy ActorSession")
	}
	valueModel, ok := critic.(*CriticSession)
	if !ok || valueModel == nil {
		return Rollout{}, errors.New("collector requires a CriticSession")
	}
	var reference *ActorSession
	if supervisedReference != nil {
		reference, ok = supervisedReference.(*ActorSession)
		if !ok {
			return Rollout{}, errors.New("collector supervised reference is not an ActorSession")
		}
	}
	if err := validateCollectorConfig(config); err != nil {
		return Rollout{}, err
	}
	answers := rolloutAnswers(config)
	actionRNG := rand.New(rand.NewSource(config.Seed))
	parallel := config.ParallelGames
	if parallel <= 0 {
		parallel = 256
	}
	if parallel > len(answers) {
		parallel = len(answers)
	}
	result := Rollout{Episodes: make([]RolloutEpisode, 0, len(answers))}
	var entropySum, referenceKLSum float64
	for start := 0; start < len(answers); start += parallel {
		end := min(start+parallel, len(answers))
		chunk, entropy, referenceKL, err := collectChunk(vocab, actor, valueModel, reference, answers[start:end], int64(start), actionRNG)
		if err != nil {
			return Rollout{}, fmt.Errorf("collect games %d..%d: %w", start, end-1, err)
		}
		result.Transitions = append(result.Transitions, chunk.Transitions...)
		result.Episodes = append(result.Episodes, chunk.Episodes...)
		entropySum += entropy
		referenceKLSum += referenceKL
	}
	if err := finishRollout(&result, config.Gamma, config.GAELambda); err != nil {
		return Rollout{}, err
	}
	if len(result.Transitions) > 0 {
		result.Metrics.PolicyEntropy = entropySum / float64(len(result.Transitions))
		result.Metrics.SupervisedReferenceKL = referenceKLSum / float64(len(result.Transitions))
	}
	return result, nil
}

func validateCollectorConfig(config CollectorConfig) error {
	if len(config.AnswerPool) == 0 || config.Games <= 0 || config.Seed == 0 {
		return errors.New("collector answer pool, positive game count, and non-zero seed are required")
	}
	if config.Balanced && config.Games%len(config.AnswerPool) != 0 {
		return fmt.Errorf("balanced rollout games %d must be a multiple of answer pool %d", config.Games, len(config.AnswerPool))
	}
	if config.Gamma < 0 || config.Gamma > 1 || config.GAELambda < 0 || config.GAELambda > 1 || !finite64(config.Gamma) || !finite64(config.GAELambda) {
		return errors.New("collector gamma and GAE lambda must be finite in [0,1]")
	}
	return nil
}

func rolloutAnswers(config CollectorConfig) []string {
	rng := rand.New(rand.NewSource(config.Seed ^ 0x5deece66d))
	answers := make([]string, config.Games)
	if config.Balanced {
		for index := range answers {
			answers[index] = config.AnswerPool[index%len(config.AnswerPool)]
		}
		rng.Shuffle(len(answers), func(i, j int) { answers[i], answers[j] = answers[j], answers[i] })
		return answers
	}
	for index := range answers {
		answers[index] = config.AnswerPool[rng.Intn(len(config.AnswerPool))]
	}
	return answers
}

func collectChunk(vocab *vocabulary.Vocabulary, actor *ActorSession, critic *CriticSession, reference *ActorSession, answers []string, episodeOffset int64, rng *rand.Rand) (Rollout, float64, float64, error) {
	environments := make([]*rlenv.Environment, len(answers))
	episodes := make([]RolloutEpisode, len(answers))
	for index, answer := range answers {
		environment, err := rlenv.NewEnvironment(vocab, answer, episodeOffset+int64(index))
		if err != nil {
			return Rollout{}, 0, 0, err
		}
		environments[index] = environment
		metadata := environment.Metadata()
		episodes[index] = RolloutEpisode{EpisodeID: metadata.EpisodeID, SolutionID: metadata.SolutionID, OpeningAction: -1}
	}
	result := Rollout{Episodes: episodes}
	var entropySum, referenceKLSum float64
	for turn := 0; turn < 6; turn++ {
		activeIndices := make([]int, 0, len(environments))
		observations := make([]rlenv.Observation, 0, len(environments))
		inputs := make([]modelstate.Inputs, 0, len(environments))
		for index, environment := range environments {
			if environment.Finished() {
				continue
			}
			observation, err := environment.Observation()
			if err != nil {
				return Rollout{}, 0, 0, err
			}
			activeIndices = append(activeIndices, index)
			observations = append(observations, observation)
			inputs = append(inputs, observation.Inputs)
		}
		if len(activeIndices) == 0 {
			break
		}
		// Keep one fixed batch shape for all six turns in this chunk. This avoids
		// recompiling the large CUDA actor graph for every random active-game
		// count; padded rows are exact observation duplicates and are ignored.
		activeCount := len(activeIndices)
		for len(inputs) < len(environments) {
			inputs = append(inputs, inputs[0])
		}
		logits, err := actor.Predict(inputs)
		if err != nil {
			return Rollout{}, 0, 0, err
		}
		values, err := critic.Predict(inputs)
		if err != nil {
			return Rollout{}, 0, 0, err
		}
		var referenceLogits [][]float32
		if reference != nil {
			referenceLogits, err = reference.Predict(inputs)
			if err != nil {
				return Rollout{}, 0, 0, err
			}
		}
		for batchIndex, environmentIndex := range activeIndices[:activeCount] {
			distribution, err := NewCategorical(logits[batchIndex], observations[batchIndex].AvailableActionMask)
			if err != nil {
				return Rollout{}, 0, 0, err
			}
			action, err := distribution.Sample(rng)
			if err != nil {
				return Rollout{}, 0, 0, err
			}
			transition, err := environments[environmentIndex].Step(action)
			if err != nil {
				return Rollout{}, 0, 0, fmt.Errorf("sampled action %d rejected by authoritative engine: %w", action, err)
			}
			if transition.Turn != turn || transition.Observation.AvailableActionMask[action] != 1 {
				return Rollout{}, 0, 0, errors.New("collector transition turn/mask invariant failed")
			}
			stored := TrajectoryTransition{
				ModelInputs: transition.Observation.Inputs, Availability: transition.Observation.AvailableActionMask,
				ActionID: int32(action), OldLogProb: distribution.LogProbability(action), OldValue: float64(values[batchIndex]),
				Reward: float64(transition.Reward), Terminal: transition.Terminal, EpisodeID: transition.Metadata.EpisodeID,
				Turn: transition.Turn, SolutionID: transition.Metadata.SolutionID, Solved: transition.Solved,
			}
			result.Transitions = append(result.Transitions, stored)
			entropySum += distribution.Entropy()
			if reference != nil {
				referenceDistribution, err := NewCategorical(referenceLogits[batchIndex], observations[batchIndex].AvailableActionMask)
				if err != nil {
					return Rollout{}, 0, 0, err
				}
				kl, err := ReferenceKL(referenceDistribution, distribution)
				if err != nil {
					return Rollout{}, 0, 0, err
				}
				referenceKLSum += kl
			}
			episode := &result.Episodes[environmentIndex]
			if turn == 0 {
				episode.OpeningAction = action
				episode.OpeningProbability = distribution.Probability(action)
			}
			if transition.Terminal {
				episode.Solved = transition.Solved
				episode.Guesses = turn + 1
			}
		}
	}
	for _, episode := range result.Episodes {
		if episode.Guesses < 1 || episode.Guesses > 6 {
			return Rollout{}, 0, 0, fmt.Errorf("episode %d did not finish", episode.EpisodeID)
		}
	}
	return result, entropySum, referenceKLSum, nil
}

func finishRollout(rollout *Rollout, gamma, lambda float64) error {
	if rollout == nil || len(rollout.Transitions) == 0 || len(rollout.Episodes) == 0 {
		return errors.New("cannot finish an empty rollout")
	}
	indicesByEpisode := make(map[int64][]int, len(rollout.Episodes))
	for index, transition := range rollout.Transitions {
		indicesByEpisode[transition.EpisodeID] = append(indicesByEpisode[transition.EpisodeID], index)
	}
	rawAdvantages := make([]float64, len(rollout.Transitions))
	var episodeReturnSum, solvedGuessSum, failureCountedSum float64
	var solvedCount int
	for episodeIndex := range rollout.Episodes {
		episode := &rollout.Episodes[episodeIndex]
		indices := indicesByEpisode[episode.EpisodeID]
		sort.Slice(indices, func(i, j int) bool {
			return rollout.Transitions[indices[i]].Turn < rollout.Transitions[indices[j]].Turn
		})
		if len(indices) != episode.Guesses {
			return fmt.Errorf("episode %d has %d transitions, want %d", episode.EpisodeID, len(indices), episode.Guesses)
		}
		rewards := make([]float64, len(indices))
		values := make([]float64, len(indices)+1)
		terminals := make([]bool, len(indices))
		for position, index := range indices {
			transition := &rollout.Transitions[index]
			rewards[position] = transition.Reward
			values[position] = transition.OldValue
			terminals[position] = transition.Terminal
		}
		returns, err := DiscountedReturns(rewards, gamma)
		if err != nil {
			return err
		}
		advantages, err := GeneralizedAdvantages(rewards, values, terminals, gamma, lambda)
		if err != nil {
			return err
		}
		for position, index := range indices {
			rollout.Transitions[index].Return = returns[position]
			rawAdvantages[index] = advantages[position]
			rollout.Transitions[index].Solved = episode.Solved
		}
		episode.Return = returns[0]
		episodeReturnSum += episode.Return
		if episode.Solved {
			solvedCount++
			solvedGuessSum += float64(episode.Guesses)
			failureCountedSum += float64(episode.Guesses)
		} else {
			failureCountedSum += 7
		}
	}
	normalized, rawMean, rawStd, err := NormalizeAdvantages(rawAdvantages)
	if err != nil {
		return err
	}
	for index := range rollout.Transitions {
		rollout.Transitions[index].Advantage = normalized[index]
	}
	var normalizedMean, normalizedSquares, returnSum, rewardSum float64
	for _, transition := range rollout.Transitions {
		normalizedMean += transition.Advantage
		returnSum += transition.Return
		rewardSum += transition.Reward
	}
	normalizedMean /= float64(len(rollout.Transitions))
	for _, transition := range rollout.Transitions {
		delta := transition.Advantage - normalizedMean
		normalizedSquares += delta * delta
	}
	metrics := &rollout.Metrics
	metrics.EpisodeReturnMean = episodeReturnSum / float64(len(rollout.Episodes))
	metrics.SolveRate = float64(solvedCount) / float64(len(rollout.Episodes))
	if solvedCount > 0 {
		metrics.MeanGuessesSolved = solvedGuessSum / float64(solvedCount)
	}
	metrics.FailureCountedMeanGuesses = failureCountedSum / float64(len(rollout.Episodes))
	metrics.AdvantageMeanBeforeNormalize = rawMean
	metrics.AdvantageStdBeforeNormalize = rawStd
	metrics.AdvantageMean = normalizedMean
	metrics.AdvantageStd = math.Sqrt(normalizedSquares / float64(len(rollout.Transitions)))
	metrics.ReturnMean = returnSum / float64(len(rollout.Transitions))
	metrics.StepRewardMean = rewardSum / float64(len(rollout.Transitions))
	metrics.RolloutGames = len(rollout.Episodes)
	metrics.RolloutSteps = len(rollout.Transitions)
	openingCounts := make(map[int]int)
	openingProbabilitySums := make(map[int]float64)
	for _, episode := range rollout.Episodes {
		openingCounts[episode.OpeningAction]++
		openingProbabilitySums[episode.OpeningAction] += episode.OpeningProbability
	}
	metrics.OpeningAction = -1
	for action, count := range openingCounts {
		if metrics.OpeningAction < 0 || count > openingCounts[metrics.OpeningAction] || count == openingCounts[metrics.OpeningAction] && action < metrics.OpeningAction {
			metrics.OpeningAction = action
		}
	}
	metrics.OpeningActionProbability = openingProbabilitySums[metrics.OpeningAction] / float64(openingCounts[metrics.OpeningAction])
	return nil
}
