package utils

func TrainLinearRegression(x []float64, y []float64) (slope float64, intercept float64) {
	n := len(x)

	// Calculate the mean of x and y
	xMean := mean(x)
	yMean := mean(y)

	// Calculate the slope and intercept
	var numerator float64
	var denominator float64
	for i := 0; i < n; i++ {
		numerator += (x[i] - xMean) * (y[i] - yMean)
		denominator += (x[i] - xMean) * (x[i] - xMean)
	}
	slope = numerator / denominator
	intercept = yMean - slope*xMean

	return slope, intercept
}

func PredictLinearRegression(x float64, slope float64, intercept float64) float64 {
	return slope*x + intercept
}

func mean(values []float64) float64 {
	sum := 0.0
	n := len(values)
	for i := 0; i < n; i++ {
		sum += values[i]
	}
	return sum / float64(n)
}
