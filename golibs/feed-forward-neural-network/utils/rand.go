package utils

import (
	cryptoRandom "crypto/rand"
	"math/big"
	simpleRandom "math/rand"
)

var (
	useCryptoRandomGenerator = false
)

func EnableCryptoRandomGenerator(enable bool) {
	useCryptoRandomGenerator = enable
}

func cryptoRandomInt64(max int64) int64 {
	nBig, err := cryptoRandom.Int(cryptoRandom.Reader, big.NewInt(max))
	if err != nil {
		panic(err)
	}
	return nBig.Int64()
}

func CryptoRandomInt(max int) int {
	if useCryptoRandomGenerator {
		return int(cryptoRandomInt64(int64(max)))
	}

	return simpleRandom.Intn(max)
}

// Returns a random number between 0 and 1
func CryptoRandomFloat() float64 {
	if useCryptoRandomGenerator {
		// generate a random number from 0  2^53 . i.e. math.pow(2,53)
		// then devide it by 2^53
		// So we get a random between 0 and 1
		return float64(cryptoRandomInt64(1<<53)) / (1 << 53)
	}

	return simpleRandom.Float64()
}

func CryptoRandomFloatRange(min, max float64) float64 {
	if useCryptoRandomGenerator {
		return min + CryptoRandomFloat()*(max-min)
	}

	value := simpleRandom.NormFloat64()
	if value < min {
		return min
	}

	if value > max {
		return max
	}

	return value
}
