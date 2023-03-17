package elman

import (
	"math"

	"github.com/mtricolici/ai-study-2023/golibs/elman-network/emath"
)

func calculate_gradient(output, errors []float64) []float64 {
	size := len(output)
	gradient := make([]float64, size)
	for i := 0; i < size; i++ {
		gradient[i] = errors[i] * emath.SigmoidDerivative(output[i])
	}
	return gradient
}

func calculate_hidden_error(currentLayer, nextLayer *Layer, nextErrors []float64) []float64 {
	errors := make([]float64, currentLayer.NumNeurons)

	for i := 0; i < currentLayer.NumNeurons; i++ {
		errors[i] = 0.0
		for j := 0; j < nextLayer.NumNeurons; j++ {
			errors[i] += nextLayer.Neurons[j].Weights[i] * nextErrors[j]
		}
	}

	return errors
}

func calculate_error(target, output []float64) []float64 {
	size := len(output)
	errors := make([]float64, size)

	for i := 0; i < size; i++ {
		errors[i] = target[i] - output[i]
	}

	return errors
}

func calculate_avg_error(target, output []float64) float64 {
	error_sum := 0.0

	for i := 0; i < len(target); i++ {
		diff := target[i] - output[i]
		error_sum += math.Abs(diff)

	}

	return error_sum / float64(len(target))
}
