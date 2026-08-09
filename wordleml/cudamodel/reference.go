package cudamodel

import (
	"fmt"
)

// Forward evaluates the complete exported policy using straightforward FP32
// loops and returns all unmasked raw action logits. It is intentionally only a
// portable verification evaluator; production inference belongs to CUDA.
func (m *Model) Forward(inputs Inputs) ([]float32, error) {
	logits, _, err := m.forward(inputs, false)
	return logits, err
}

// ForwardWithActivations evaluates the policy and additionally returns named
// copies of the layer outputs that make export/CUDA parity failures localizable.
// The map contains candidate_projection_relu, stats_projection_relu,
// turn_embedding, residual_in_relu, residual_out, hidden, base_logits, and
// candidate_bonus. The candidate-bonus activation is a one-element slice.
func (m *Model) ForwardWithActivations(inputs Inputs) ([]float32, map[string][]float32, error) {
	return m.forward(inputs, true)
}

func (m *Model) forward(inputs Inputs, includeActivations bool) ([]float32, map[string][]float32, error) {
	if m == nil {
		return nil, nil, fmt.Errorf("model is nil")
	}
	if len(m.weights) != ParameterCount {
		return nil, nil, fmt.Errorf("model has %d weights, want %d", len(m.weights), ParameterCount)
	}
	candidateSum, err := ValidateInputs(inputs)
	if err != nil {
		return nil, nil, err
	}
	for index, weight := range m.weights {
		if !isFinite(weight) {
			return nil, nil, fmt.Errorf("weight %d is not finite", index)
		}
	}

	candidateFeatures := denseRelu(m.weights, tensorOffset(CandidateProjectionWeight), tensorOffset(CandidateProjectionBias), CandidateProjectionSize, NumSolutions, inputs.CandidateMask, candidateSum)
	if err := validateFiniteSlice("candidate_projection_relu", candidateFeatures); err != nil {
		return nil, nil, err
	}
	statsFeatures := denseRelu(m.weights, tensorOffset(StatsProjectionWeight), tensorOffset(StatsProjectionBias), StatsProjectionSize, CandidateStatsSize, inputs.CandidateStats, 1)
	if err := validateFiniteSlice("stats_projection_relu", statsFeatures); err != nil {
		return nil, nil, err
	}
	turnFeatures := append([]float32(nil), m.weights[tensorOffset(TurnEmbedding)+int(inputs.Turn)*TurnEmbeddingSize:tensorOffset(TurnEmbedding)+(int(inputs.Turn)+1)*TurnEmbeddingSize]...)
	if err := validateFiniteSlice("turn_embedding", turnFeatures); err != nil {
		return nil, nil, err
	}

	h := make([]float32, TrunkSize)
	copy(h, candidateFeatures)
	copy(h[CandidateProjectionSize:], statsFeatures)
	copy(h[CandidateProjectionSize+StatsProjectionSize:], turnFeatures)

	residualIn := denseRelu(m.weights, tensorOffset(ResidualInWeight), tensorOffset(ResidualInBias), TrunkSize, TrunkSize, h, 1)
	if err := validateFiniteSlice("residual_in_relu", residualIn); err != nil {
		return nil, nil, err
	}
	residualOut := dense(m.weights, tensorOffset(ResidualOutWeight), tensorOffset(ResidualOutBias), TrunkSize, TrunkSize, residualIn)
	if err := validateFiniteSlice("residual_out", residualOut); err != nil {
		return nil, nil, err
	}
	for index := range h {
		h[index] = relu(h[index] + residualOut[index])
	}
	if err := validateFiniteSlice("hidden", h); err != nil {
		return nil, nil, err
	}

	baseLogits := dense(m.weights, tensorOffset(BaseLogitsWeight), tensorOffset(BaseLogitsBias), NumActions, TrunkSize, h)
	if err := validateFiniteSlice("base_logits", baseLogits); err != nil {
		return nil, nil, err
	}
	beta := dense(m.weights, tensorOffset(CandidateBonusWeight), tensorOffset(CandidateBonusBias), 1, TrunkSize, h)[0]
	if !isFinite(beta) {
		return nil, nil, fmt.Errorf("candidate_bonus is not finite")
	}

	logits := make([]float32, NumActions)
	for index := range logits {
		logits[index] = baseLogits[index] + beta*inputs.RemainingActionMask[index]
	}
	if err := validateFiniteSlice("logits", logits); err != nil {
		return nil, nil, err
	}
	if !includeActivations {
		return logits, nil, nil
	}
	return logits, map[string][]float32{
		"candidate_projection_relu": candidateFeatures,
		"stats_projection_relu":     statsFeatures,
		"turn_embedding":            turnFeatures,
		"residual_in_relu":          residualIn,
		"residual_out":              residualOut,
		"hidden":                    append([]float32(nil), h...),
		"base_logits":               baseLogits,
		"candidate_bonus":           []float32{beta},
	}, nil
}

// ValidateInputs checks the contract of one non-batched policy evaluation and
// returns the positive FP32 candidate-mask sum used for mean pooling.
func ValidateInputs(inputs Inputs) (float32, error) {
	if len(inputs.CandidateMask) != NumSolutions {
		return 0, fmt.Errorf("candidate mask has length %d, want %d", len(inputs.CandidateMask), NumSolutions)
	}
	if len(inputs.CandidateStats) != CandidateStatsSize {
		return 0, fmt.Errorf("candidate stats has length %d, want %d", len(inputs.CandidateStats), CandidateStatsSize)
	}
	if len(inputs.RemainingActionMask) != NumActions {
		return 0, fmt.Errorf("remaining action mask has length %d, want %d", len(inputs.RemainingActionMask), NumActions)
	}
	if inputs.Turn < 0 || inputs.Turn >= NumTurns {
		return 0, fmt.Errorf("turn is %d, want 0 through %d", inputs.Turn, NumTurns-1)
	}

	var candidateSum float32
	for index, value := range inputs.CandidateMask {
		if !isFinite(value) {
			return 0, fmt.Errorf("candidate mask value %d is not finite", index)
		}
		candidateSum += value
	}
	if !isFinite(candidateSum) || candidateSum <= 0 {
		return 0, fmt.Errorf("candidate mask sum is %g, want a positive finite value", candidateSum)
	}
	if err := validateFiniteSlice("candidate stats", inputs.CandidateStats); err != nil {
		return 0, err
	}
	if err := validateFiniteSlice("remaining action mask", inputs.RemainingActionMask); err != nil {
		return 0, err
	}
	return candidateSum, nil
}

func denseRelu(weights []float32, weightOffset, biasOffset, outputs, inputs int, input []float32, inputScale float32) []float32 {
	values := make([]float32, outputs)
	for output := range values {
		sum := weights[biasOffset+output]
		rowOffset := weightOffset + output*inputs
		for inputIndex := 0; inputIndex < inputs; inputIndex++ {
			inputValue := input[inputIndex]
			if inputScale != 1 {
				inputValue /= inputScale
			}
			sum += weights[rowOffset+inputIndex] * inputValue
		}
		values[output] = relu(sum)
	}
	return values
}

func dense(weights []float32, weightOffset, biasOffset, outputs, inputs int, input []float32) []float32 {
	values := make([]float32, outputs)
	for output := range values {
		sum := weights[biasOffset+output]
		rowOffset := weightOffset + output*inputs
		for inputIndex := 0; inputIndex < inputs; inputIndex++ {
			sum += weights[rowOffset+inputIndex] * input[inputIndex]
		}
		values[output] = sum
	}
	return values
}

func relu(value float32) float32 {
	if value > 0 {
		return value
	}
	return 0
}

func validateFiniteSlice(name string, values []float32) error {
	for index, value := range values {
		if !isFinite(value) {
			return fmt.Errorf("%s value %d is not finite", name, index)
		}
	}
	return nil
}

func tensorOffset(name string) int {
	for _, tensor := range expectedTensors {
		if tensor.Name == name {
			return tensor.Offset
		}
	}
	panic(fmt.Sprintf("unknown fixed tensor %q", name))
}
