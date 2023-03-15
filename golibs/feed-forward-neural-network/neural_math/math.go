package neural_math

import "math"

// "clamped" sigmoid function with simple thresholding
func Sigmoid(x float64) float64 {
	if x > 100 {
		return 1.0
	}

	if x < -100 {
		return 0.0
	}

	return 1 / (1 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}
