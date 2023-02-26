package main

import (
	"fmt"
	"regression_sample1/utils"
	"time"
)

func main() {
	fmt.Println("Example of regression in golang!")

	// inputDates - dates converted to float64
	// inputValues - euro/usd exchange rate for that date
	inputDates, inputValues := utils.Read_dataset()

	// Train the linear regression model
	slope, intercept := utils.TrainLinearRegression(inputDates, inputValues)

	// Predict next X values
	var nextPredictedValues []float64

	lastDate := time.Unix(int64(inputDates[len(inputDates)-1]), 0)

	predictedValuesCount := (len(inputValues) - 1) / 2

	for i := 0; i < predictedValuesCount; i++ {
		nextDate := lastDate.AddDate(0, 0, 1) // Add 1 day
		lastDate = nextDate                   //save for next iteration

		nextInput := float64(nextDate.Unix())
		nextValue := utils.PredictLinearRegression(nextInput, slope, intercept)

		fmt.Printf("Predicted next value: %f\n", nextValue)

		nextPredictedValues = append(nextPredictedValues, nextValue)
	}

	// Save results to a PNG file for better visibility!
	utils.DrawRegressionToPngFile(
		inputValues, nextPredictedValues, "output.png")
}
