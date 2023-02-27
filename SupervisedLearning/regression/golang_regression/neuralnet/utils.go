package neuralnet

import "math"

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

func Array_minus_array(arr1, arr2 []float64) []float64 {
	result := make([]float64, len(arr1))
	for i := range arr1 {
		result[i] = arr1[i] - arr2[i]
	}
	return result
}

func Calculate_delta(errors, outputs []float64) []float64 {
	delta := make([]float64, len(outputs))

	for j, output := range outputs {
		delta[j] = errors[j] * SigmoidDerivative(output)
	}

	return delta
}

func Calcualte_error_sum(errors []float64) float64 {
	error := 0.0

	for _, err := range errors {
		error += math.Abs(err)
	}

	return error
}
