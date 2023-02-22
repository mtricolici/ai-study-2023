package utils

func SumFloatsArray(floats []float64) float64 {
	var sum float64
	for _, value := range floats {
		sum += value
	}

	return sum
}
