package genetic

import (
	"math/rand"
	"strconv"
)

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

func groupDigits(num int) string {
	str := strconv.Itoa(num)
	n := len(str)
	if n <= 3 {
		return str
	}
	var result string
	for i := 0; i < n; i++ {
		result += string(str[i])
		if (n-i-1)%3 == 0 && i != n-1 {
			result += "."
		}
	}
	return result
}
