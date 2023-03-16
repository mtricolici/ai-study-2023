package random

import "math/rand"

func Int(max int) int {
	return rand.Intn(max)
}

func Float64() float64 {
	return rand.Float64()
}

func NormFloat64() float64 {
	return rand.NormFloat64()
}

func Float64Range(min, max float64) float64 {
	return min + Float64()*(max-min)
}

func NormFloat64Range(min, max float64) float64 {
	value := min + Float64()*(max-min)
	if value < min {
		return min
	}
	if value > max {
		return max
	}

	return value
}
