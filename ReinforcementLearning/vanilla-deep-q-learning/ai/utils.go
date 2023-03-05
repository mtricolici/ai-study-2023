package ai

func args_max_index(slice []float64) int {
	maxIndex := 0
	maxValue := slice[0]
	for i := 1; i < len(slice); i++ {
		if slice[i] > maxValue {
			maxIndex = i
			maxValue = slice[i]
		}
	}
	return maxIndex
}

func args_max(slice []float64) float64 {
	max := slice[0]
	for i := 1; i < len(slice); i++ {
		if slice[i] > max {
			max = slice[i]
		}
	}
	return max
}
