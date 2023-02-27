package neural

import (
	"regression_sample1/neuralnet"
	"regression_sample1/utils"
)

type NeuralRegression struct {
	batchSize                int
	numEpochs                int
	learningRate             float64
	stopTrainingMaxAvgError  float64
	normalizeValueMultiplier float64
	network                  *neuralnet.FeedForwardNeuralNetwork
}

func NewNeuralRegression(batchSize, hiddenNeurons int) *NeuralRegression {
	network := neuralnet.NewFeedForwardNeuralNetwork(batchSize, hiddenNeurons, 1)
	return &NeuralRegression{
		batchSize:                batchSize,
		numEpochs:                1000,
		learningRate:             0.05,
		stopTrainingMaxAvgError:  0.1,
		normalizeValueMultiplier: 0.5,
		network:                  network,
	}
}

func (nr *NeuralRegression) Train(inputValues []float64) {
	var trainingInputs [][]float64
	var trainingLabels [][]float64

	// need to split inputValues[] into arrays of 'batchSize' + 1.
	// first 'batchSize' elements are input for neural network
	// element 'batchSize' + 1 -> expected output

	inputIndex := 0

	for inputIndex+nr.batchSize < len(inputValues) {

		// next value after them - output value for neural network
		tinput := make([]float64, nr.batchSize)
		tlabel := make([]float64, 1)

		// first 'batchSize' values - input for neural network
		for i := 0; i < nr.batchSize; i++ {
			tinput[i] = inputValues[inputIndex+i] * nr.normalizeValueMultiplier
		}
		// next element - outpot of neural network
		tlabel[0] = inputValues[inputIndex+nr.batchSize] * nr.normalizeValueMultiplier

		trainingInputs = append(trainingInputs, tinput)
		trainingLabels = append(trainingLabels, tlabel)

		inputIndex++
	}

	nr.network.Train(trainingInputs, trainingLabels, nr.numEpochs, nr.learningRate, nr.stopTrainingMaxAvgError)
}

func (nr *NeuralRegression) Predict(inputValues []float64, predictionsCount int) []float64 {

	predicted := make([]float64, predictionsCount)

	// Find last X values in training data to predict next values
	inputs := nr.getLastBatchOfElements(inputValues)

	// normalize inputs
	for i := 0; i < nr.batchSize; i++ {
		inputs[i] = inputs[i] * nr.normalizeValueMultiplier
	}

	for i := 0; i < predictionsCount; i++ {
		nextPredictedValue := nr.network.Predict(inputs)[0]
		predicted[i] = nextPredictedValue / nr.normalizeValueMultiplier

		//move inputs left and add nextPredictedValue to the end
		for k := 0; k < nr.batchSize-1; k++ {
			inputs[k] = inputs[k+1]
		}
		inputs[nr.batchSize-1] = nextPredictedValue
	}

	return predicted
}

func (nr *NeuralRegression) getLastBatchOfElements(arr []float64) []float64 {
	arrLen := len(arr)
	lastBatchSizeElements := arr[arrLen-nr.batchSize:]

	result := make([]float64, nr.batchSize)
	copy(result, lastBatchSizeElements)
	return result
}

func (nr *NeuralRegression) TrainAndTest(inputValues []float64, predictionsCount int) {

	nr.Train(inputValues)

	nextPredictedValues := nr.Predict(inputValues, predictionsCount)

	// Save results to a PNG file for better visibility!
	utils.DrawRegressionToPngFile(
		inputValues, nextPredictedValues, "neural regression", "neural.png")
}
