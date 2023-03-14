package utils

import (
	"crypto/rand"
	"math/big"
)

func CryptoRandomInt64(max int64) int64 {
	nBig, err := rand.Int(rand.Reader, big.NewInt(max))
	if err != nil {
		panic(err)
	}
	return nBig.Int64()
}

func CryptoRandomInt(max int) int {
	return int(CryptoRandomInt64(int64(max)))
}

// Returns a random number between 0 and 1
func CryptoRandomFloat() float64 {
	// generate a random number from 0  2^53 . i.e. math.pow(2,53)
	// then devide it by 2^53
	// So we get a random between 0 and 1
	return float64(CryptoRandomInt64(1<<53)) / (1 << 53)
}

func CryptoRandomFloatRange(min, max float64) float64 {
	return min + CryptoRandomFloat()*(max-min)
}
