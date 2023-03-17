package main

import (
	"fmt"
	"regression_sample1/regression/elmanr"
	"regression_sample1/regression/linear"
	"regression_sample1/regression/polynomial"
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

	// nr := neural.NewNeuralRegression(300, 300)
	// nr.TrainAndTest(inputValues, 500)

	pl := polynomial.NewPolynomialRegression(2)
	pl.TrainAndTest(inputDates, inputValues)

	//rr := recurrentnn.NewRecurrentNNRegression(10, 64)
	//rr.TrainAndTest(inputValues)

	er := elmanr.NewElmanRegression(30, 100)
	er.NumEpochs = 8000
	er.PredictionsCount = 200
	er.Network.LearningRate = 0.08
	er.Network.TrainingReportSeconds = 5 // print training progress every 15 seconds
	er.TrainAndTest(inputValues)

	fmt.Println("Done!")
}
