package genetic

import (
	"math/rand"
	"strconv"
)

func randGaussianDistribution(value float64) float64 {
	return rand.NormFloat64()*0.1 + value
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
