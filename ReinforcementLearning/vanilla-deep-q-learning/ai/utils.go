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
