package main

import (
	"fmt"
	"regression_sample1/regression/linear"
	"regression_sample1/utils"
)

func main() {
	fmt.Println("Example of regression in golang!")

	// inputDates - dates converted to float64
	// inputValues - euro/usd exchange rate for that date
	inputDates, inputValues := utils.Read_dataset()

	// Invoke Linear Regression POC
	lr := linear.NewLinearRegression()
	lr.TrainAndTest(inputDates, inputValues)

	fmt.Println("Done!")
}
