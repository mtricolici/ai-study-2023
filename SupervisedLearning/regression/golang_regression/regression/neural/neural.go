package neural

import (
	"regression_sample1/utils"

	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/backpropagation"
	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/neural_net"
)

type NeuralRegression struct {
	batchSize                int
	numEpochs                int
	normalizeValueMultiplier float64
	network                  *neural_net.FeedForwardNeuralNetwork
	training                 *backpropagation.BackpropagationTraining
}

func NewNeuralRegression(batchSize, hiddenNeurons int) *NeuralRegression {
	topology := []int{batchSize, hiddenNeurons, 1}
	network := neural_net.NewFeedForwardNeuralNetwork(topology, true)
	training := backpropagation.NewBackpropagationTraining(network)
	training.LearningRate = 0.08
	training.StopTrainingMaxAvgError = 0.001

	return &NeuralRegression{
		batchSize:                batchSize,
		numEpochs:                1000,
		normalizeValueMultiplier: 0.5,
		network:                  network,
		training:                 training,
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

	nr.training.Train(trainingInputs, trainingLabels, nr.numEpochs)
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
