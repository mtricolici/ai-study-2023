package emath

import "math"

const (
	sigmoid_threshold = 50.0
)

// "clamped" sigmoid function with threshold
func Sigmoid(x float64) float64 {
	if x > sigmoid_threshold {
		return 1.0
	}

	if x < -sigmoid_threshold {
		return 0.0
	}

	return 1 / (1 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}
