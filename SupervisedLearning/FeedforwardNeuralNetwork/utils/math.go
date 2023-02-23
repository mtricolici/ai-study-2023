package utils

import "math"

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

func Relu(x float64) float64 {
	if x < 0 {
		return 0
	}

	return x
}

func Tanh(x float64) float64 {
	return math.Tanh(x)
}

func Elu(x float64) float64 {
	if x < 0 {
		return math.Exp(x) - 1
	}

	return x
}

func Softmax(x []float64) []float64 {
	y := make([]float64, len(x))
	var sum float64
	for i := 0; i < len(x); i++ {
		y[i] = math.Exp(x[i])
		sum += y[i]
	}
	for i := 0; i < len(x); i++ {
		y[i] /= sum
	}
	return y
}
