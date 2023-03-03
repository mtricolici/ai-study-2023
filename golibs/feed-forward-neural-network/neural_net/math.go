package neural_net

import "math"

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}
