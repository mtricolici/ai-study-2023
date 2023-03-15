package backpropagation

import (
	"math"

	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/neural_math"
)

func array_minus_array(arr1, arr2 []float64) []float64 {
	result := make([]float64, len(arr1))
	for i := range arr1 {
		result[i] = arr1[i] - arr2[i]
	}
	return result
}

func calculate_delta(errors, outputs []float64) []float64 {
	delta := make([]float64, len(outputs))

	for j, output := range outputs {
		delta[j] = errors[j] * neural_math.SigmoidDerivative(output)
	}

	return delta
}

func calcualte_error_sum(errors []float64) float64 {
	error := 0.0

	for _, err := range errors {
		error += math.Abs(err)
	}

	return error
}
