package ppo

import (
	"errors"
	"fmt"
	"math"
)

// ValueStatistics summarize critic predictions against realised returns.
type ValueStatistics struct {
	Count             int                     `json:"count"`
	ValueLoss         float64                 `json:"value_loss"`
	ExplainedVariance float64                 `json:"explained_variance"`
	PredictionBias    float64                 `json:"prediction_bias"`
	OpeningPrediction float64                 `json:"opening_state_prediction"`
	ByTurn            map[int]PredictionGroup `json:"prediction_by_turn"`
	Solved            PredictionGroup         `json:"solved_episodes"`
	Failed            PredictionGroup         `json:"failed_episodes"`
}

// PredictionGroup gives prediction/target means for one requested slice.
type PredictionGroup struct {
	Count          int     `json:"count"`
	PredictionMean float64 `json:"prediction_mean"`
	ReturnMean     float64 `json:"return_mean"`
	Bias           float64 `json:"bias"`
}

// CalculateValueStatistics computes critic warm-up evidence. solved flags are
// transition-level copies of their complete episode outcome.
func CalculateValueStatistics(predictions, returns []float64, turns []int, solved []bool) (ValueStatistics, error) {
	if len(predictions) == 0 || len(predictions) != len(returns) || len(predictions) != len(turns) || len(predictions) != len(solved) {
		return ValueStatistics{}, fmt.Errorf("value statistic lengths predictions=%d returns=%d turns=%d solved=%d", len(predictions), len(returns), len(turns), len(solved))
	}
	if err := CheckFinite("value predictions", predictions); err != nil {
		return ValueStatistics{}, err
	}
	if err := CheckFinite("value returns", returns); err != nil {
		return ValueStatistics{}, err
	}
	result := ValueStatistics{Count: len(predictions), ByTurn: make(map[int]PredictionGroup)}
	var predictionMean, returnMean float64
	for index := range predictions {
		delta := predictions[index] - returns[index]
		result.ValueLoss += delta * delta
		predictionMean += predictions[index]
		returnMean += returns[index]
		if turns[index] < 0 || turns[index] >= 6 {
			return ValueStatistics{}, fmt.Errorf("turn %d at index %d is outside 0..5", turns[index], index)
		}
	}
	predictionMean /= float64(len(predictions))
	returnMean /= float64(len(returns))
	result.ValueLoss /= float64(len(predictions))
	result.PredictionBias = predictionMean - returnMean
	var targetVariance, residualVariance float64
	residualMean := result.PredictionBias
	for index := range predictions {
		targetDelta := returns[index] - returnMean
		residualDelta := (predictions[index] - returns[index]) - residualMean
		targetVariance += targetDelta * targetDelta
		residualVariance += residualDelta * residualDelta
	}
	targetVariance /= float64(len(returns))
	residualVariance /= float64(len(returns))
	if targetVariance <= 1e-15 {
		result.ExplainedVariance = 0
	} else {
		result.ExplainedVariance = 1 - residualVariance/targetVariance
	}

	turnPredictions := make(map[int][]float64)
	turnReturns := make(map[int][]float64)
	var openingSum float64
	var openingCount int
	var solvedPredictions, solvedReturns, failedPredictions, failedReturns []float64
	for index := range predictions {
		turn := turns[index]
		turnPredictions[turn] = append(turnPredictions[turn], predictions[index])
		turnReturns[turn] = append(turnReturns[turn], returns[index])
		if turn == 0 {
			openingSum += predictions[index]
			openingCount++
		}
		if solved[index] {
			solvedPredictions = append(solvedPredictions, predictions[index])
			solvedReturns = append(solvedReturns, returns[index])
		} else {
			failedPredictions = append(failedPredictions, predictions[index])
			failedReturns = append(failedReturns, returns[index])
		}
	}
	if openingCount == 0 {
		return ValueStatistics{}, errors.New("value statistics contain no opening states")
	}
	result.OpeningPrediction = openingSum / float64(openingCount)
	for turn, values := range turnPredictions {
		result.ByTurn[turn+1] = predictionGroup(values, turnReturns[turn])
	}
	result.Solved = predictionGroup(solvedPredictions, solvedReturns)
	result.Failed = predictionGroup(failedPredictions, failedReturns)
	for _, value := range []float64{result.ValueLoss, result.ExplainedVariance, result.PredictionBias, result.OpeningPrediction} {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return ValueStatistics{}, errors.New("value statistics contain a non-finite result")
		}
	}
	return result, nil
}

func predictionGroup(predictions, returns []float64) PredictionGroup {
	group := PredictionGroup{Count: len(predictions)}
	if len(predictions) == 0 {
		return group
	}
	for index := range predictions {
		group.PredictionMean += predictions[index]
		group.ReturnMean += returns[index]
	}
	group.PredictionMean /= float64(len(predictions))
	group.ReturnMean /= float64(len(returns))
	group.Bias = group.PredictionMean - group.ReturnMean
	return group
}
