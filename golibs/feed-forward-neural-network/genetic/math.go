package genetic

import "math/rand"

func randGaussianDistribution(value float64) float64 {
	newValue := rand.NormFloat64()*0.05 + value
	//newValue := rand.Float64()*0.1 + value
	// it converges very fast with NormFloat64() instead of Float64()
	// I don't know why :D

	if newValue < 0 {
		return 0.0
	}

	if newValue > 1 {
		return 1.0
	}

	return newValue
}
