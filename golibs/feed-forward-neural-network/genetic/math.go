package genetic

import (
	"strconv"

	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/utils"
)

func randGaussianDistribution(value float64) float64 {
	//newValue := rand.NormFloat64()*0.1 + value
	newValue := utils.CryptoRandomFloatRange(-1.0, 1.0)*0.1 + value

	if newValue < -1.0 {
		return -1.0
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
