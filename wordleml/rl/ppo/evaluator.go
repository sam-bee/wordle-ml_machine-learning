package ppo

import (
	"errors"
	"fmt"

	"github.com/sam-bee/wordle-ml_machine-learning/modelstate"
	rlenv "github.com/sam-bee/wordle-ml_machine-learning/rl"
	"github.com/sam-bee/wordle-ml_machine-learning/rl/evaluation"
	"github.com/sam-bee/wordle-ml_machine-learning/vocabulary"
)

// EvaluateGreedy plays the fixed development solutions with deterministic
// deployment actions. Every transition still passes through the authoritative
// engine; old/reference actors are read-only diagnostics on the candidate's
// visited states.
func EvaluateGreedy(vocab *vocabulary.Vocabulary, solutions []string, current, oldPolicy, supervisedReference *ActorSession, critic *CriticSession, gamma, clipRange float64) (evaluation.Evaluation, ValueStatistics, error) {
	if vocab == nil || len(solutions) == 0 || current == nil || oldPolicy == nil || supervisedReference == nil || critic == nil {
		return evaluation.Evaluation{}, ValueStatistics{}, errors.New("greedy evaluation requires vocabulary, solutions, three actors, and critic")
	}
	environments := make([]*rlenv.Environment, len(solutions))
	games := make([]evaluation.GameResult, len(solutions))
	valuePredictionsByGame := make([][]float64, len(solutions))
	for index, solution := range solutions {
		environment, err := rlenv.NewEnvironment(vocab, solution, int64(index))
		if err != nil {
			return evaluation.Evaluation{}, ValueStatistics{}, err
		}
		environments[index] = environment
		games[index] = evaluation.GameResult{SolutionID: solution, Turns: make([]evaluation.TurnRecord, 0, 6)}
	}
	var currentSelectedLogProbs, oldSelectedLogProbs []float64
	var entropySum, oldKLSum, referenceKLSum float64
	var diagnosticStates int
	for turn := 0; turn < 6; turn++ {
		indices := make([]int, 0, len(solutions))
		observations := make([]rlenv.Observation, 0, len(solutions))
		inputs := make([]modelstate.Inputs, 0, len(solutions))
		for index, environment := range environments {
			if environment.Finished() {
				continue
			}
			observation, err := environment.Observation()
			if err != nil {
				return evaluation.Evaluation{}, ValueStatistics{}, err
			}
			indices = append(indices, index)
			observations = append(observations, observation)
			inputs = append(inputs, observation.Inputs)
		}
		if len(indices) == 0 {
			break
		}
		activeCount := len(indices)
		for len(inputs) < len(environments) {
			inputs = append(inputs, inputs[0])
		}
		currentLogits, err := current.Predict(inputs)
		if err != nil {
			return evaluation.Evaluation{}, ValueStatistics{}, err
		}
		oldLogits, err := oldPolicy.Predict(inputs)
		if err != nil {
			return evaluation.Evaluation{}, ValueStatistics{}, err
		}
		referenceLogits, err := supervisedReference.Predict(inputs)
		if err != nil {
			return evaluation.Evaluation{}, ValueStatistics{}, err
		}
		values, err := critic.Predict(inputs)
		if err != nil {
			return evaluation.Evaluation{}, ValueStatistics{}, err
		}
		for offset, gameIndex := range indices[:activeCount] {
			mask := observations[offset].AvailableActionMask
			currentDistribution, err := NewCategorical(currentLogits[offset], mask)
			if err != nil {
				return evaluation.Evaluation{}, ValueStatistics{}, err
			}
			oldDistribution, err := NewCategorical(oldLogits[offset], mask)
			if err != nil {
				return evaluation.Evaluation{}, ValueStatistics{}, err
			}
			referenceDistribution, err := NewCategorical(referenceLogits[offset], mask)
			if err != nil {
				return evaluation.Evaluation{}, ValueStatistics{}, err
			}
			action, err := currentDistribution.Greedy()
			if err != nil {
				return evaluation.Evaluation{}, ValueStatistics{}, err
			}
			transition, err := environments[gameIndex].Step(action)
			if err != nil {
				return evaluation.Evaluation{}, ValueStatistics{}, fmt.Errorf("greedy action %d was rejected by engine: %w", action, err)
			}
			guess, _ := vocab.ActionWord(action)
			games[gameIndex].Turns = append(games[gameIndex].Turns, evaluation.TurnRecord{
				Turn: turn + 1, Guess: guess, ActionID: action,
				ActionProbability: currentDistribution.Probability(action), PolicyEntropy: currentDistribution.Entropy(),
				Reward: float64(transition.Reward), Terminal: transition.Terminal,
			})
			valuePredictionsByGame[gameIndex] = append(valuePredictionsByGame[gameIndex], float64(values[offset]))
			currentSelectedLogProbs = append(currentSelectedLogProbs, currentDistribution.LogProbability(action))
			oldSelectedLogProbs = append(oldSelectedLogProbs, oldDistribution.LogProbability(action))
			entropySum += currentDistribution.Entropy()
			oldKL, err := ReferenceKL(oldDistribution, currentDistribution)
			if err != nil {
				return evaluation.Evaluation{}, ValueStatistics{}, err
			}
			oldKLSum += oldKL
			kl, err := ReferenceKL(referenceDistribution, currentDistribution)
			if err != nil {
				return evaluation.Evaluation{}, ValueStatistics{}, err
			}
			referenceKLSum += kl
			diagnosticStates++
			if transition.Terminal {
				games[gameIndex].Solved = transition.Solved
				games[gameIndex].Guesses = turn + 1
			}
		}
	}
	for index, game := range games {
		if game.Guesses < 1 || game.Guesses > 6 {
			return evaluation.Evaluation{}, ValueStatistics{}, fmt.Errorf("development game %s did not terminate", solutions[index])
		}
	}
	clipFraction, err := ClipFraction(currentSelectedLogProbs, oldSelectedLogProbs, clipRange)
	if err != nil {
		return evaluation.Evaluation{}, ValueStatistics{}, err
	}
	var valuePredictions, realisedReturns []float64
	var valueTurns []int
	var valueSolved []bool
	for gameIndex, game := range games {
		rewards := make([]float64, len(game.Turns))
		for index, turn := range game.Turns {
			rewards[index] = turn.Reward
		}
		returns, err := DiscountedReturns(rewards, gamma)
		if err != nil {
			return evaluation.Evaluation{}, ValueStatistics{}, err
		}
		for index, prediction := range valuePredictionsByGame[gameIndex] {
			valuePredictions = append(valuePredictions, prediction)
			realisedReturns = append(realisedReturns, returns[index])
			valueTurns = append(valueTurns, index)
			valueSolved = append(valueSolved, game.Solved)
		}
	}
	valueStats, err := CalculateValueStatistics(valuePredictions, realisedReturns, valueTurns, valueSolved)
	if err != nil {
		return evaluation.Evaluation{}, ValueStatistics{}, err
	}
	if err := ParametersFinite(current.Store); err != nil {
		return evaluation.Evaluation{}, ValueStatistics{}, err
	}
	if err := ParametersFinite(critic.Store); err != nil {
		return evaluation.Evaluation{}, ValueStatistics{}, err
	}
	result := evaluation.Evaluation{
		Games: games,
		Diagnostics: evaluation.Diagnostics{
			PolicyEntropy: entropySum / float64(diagnosticStates), ApproxOldPolicyKL: oldKLSum / float64(diagnosticStates),
			SupervisedReferenceKL: referenceKLSum / float64(diagnosticStates), ClipFraction: clipFraction,
			CriticExplainedVar: valueStats.ExplainedVariance, NumericallyStable: true,
		},
	}
	return result, valueStats, nil
}
