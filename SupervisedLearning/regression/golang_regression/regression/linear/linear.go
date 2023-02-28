package linear

import (
	"fmt"
	"regression_sample1/utils"
	"time"
)

type LinearRegression struct {
	slope     float64
	intercept float64
}

func NewLinearRegression() *LinearRegression {
	return &LinearRegression{}
}

func (lr *LinearRegression) Train(x []float64, y []float64) {

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

	lr.slope = numerator / denominator
	lr.intercept = yMean - lr.slope*xMean
}

func (lr *LinearRegression) Predict(x float64) float64 {
	return lr.slope*x + lr.intercept
}

func mean(values []float64) float64 {
	sum := 0.0
	n := len(values)
	for i := 0; i < n; i++ {
		sum += values[i]
	}
	return sum / float64(n)
}

func (lr *LinearRegression) TrainAndTest(inputDates, inputValues []float64) {
	fmt.Println("Linear regression in progress ..")

	lr.Train(inputDates, inputValues)

	predictedValuesCount := (len(inputValues) - 1) / 2

	// Predict next X values
	var nextPredictedValues []float64

	lastDate := time.Unix(int64(inputDates[len(inputDates)-1]), 0)

	for i := 0; i < predictedValuesCount; i++ {
		nextDate := lastDate.AddDate(0, 0, 1) // Add 1 day
		lastDate = nextDate                   //save for next iteration

		nextInput := float64(nextDate.Unix())
		nextValue := lr.Predict(nextInput)

		//fmt.Printf("Predicted next value: %f\n", nextValue)

		nextPredictedValues = append(nextPredictedValues, nextValue)
	}

	// Save results to a PNG file for better visibility!
	utils.DrawRegressionToPngFile(
		inputValues, nextPredictedValues, "Linear regression", "linear.png")
}
