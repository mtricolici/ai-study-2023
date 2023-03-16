package recurrentnn

import (
	"fmt"
	"regression_sample1/utils"

	"github.com/mtricolici/ai-study-2023/golibs/recurrent-neural-network/rnn"
	"github.com/mtricolici/ai-study-2023/golibs/recurrent-neural-network/trunc_backpropagation_tt"
)

type RecurrentNNRegression struct {
	network   *rnn.VanillaRecurrentNetwork
	training  *trunc_backpropagation_tt.TruncatedBackpropagationThroughTime
	timeSteps int
}

func NewRecurrentNNRegression(timeSteps, hiddenNeurons int) *RecurrentNNRegression {
	// Create RNN
	rnnStructure := []int{timeSteps, hiddenNeurons, 1}
	network := rnn.NewVanillaRecurrentNetwork(rnnStructure, true)

	// Create RNN training
	training := trunc_backpropagation_tt.NewTruncBackpropagationTT(network)
	training.LearningRate = 0.01
	training.TimeSteps = timeSteps

	return &RecurrentNNRegression{
		network:   network,
		training:  training,
		timeSteps: timeSteps,
	}
}

func (rr *RecurrentNNRegression) Predict(inputValues []float64, predictionsCount int) []float64 {
	predictions := make([]float64, 0, predictionsCount)

	for i := 0; i < predictionsCount; i++ {
		lastInput := inputValues[len(inputValues)-rr.timeSteps+i:]
		if len(lastInput) < rr.timeSteps { // << TODO: is this need?!
			lastInput = append(lastInput, predictions...)
		}

		// Important: Reset the network state before predicting
		rr.network.ResetState()

		predictedValue := rr.network.Forward(lastInput)[0]

		predictions = append(predictions, predictedValue)
		inputValues = append(inputValues, predictedValue)
	}

	return predictions
}

func (rr *RecurrentNNRegression) Train(inputValues []float64, epochs int) {
	// prepare training data
	inputs := make([][]float64, len(inputValues)-rr.timeSteps)
	targets := make([][]float64, len(inputValues)-rr.timeSteps)

	for i := 0; i < len(inputValues)-rr.timeSteps; i++ {
		inputs[i] = inputValues[i : i+rr.timeSteps]
		targets[i] = []float64{inputValues[i+rr.timeSteps]}
	}

	// Train RNN
	for epoch := 0; epoch < epochs; epoch++ {
		rr.training.Train(inputs, targets)
	}
}

func (rr *RecurrentNNRegression) TrainAndTest(inputValues []float64) {
	fmt.Println("Linear regression in progress ..")

	rr.Train(inputValues, 100)

	predictedValuesCount := (len(inputValues) - 1) / 2
	predictedValues := rr.Predict(inputValues, predictedValuesCount)

	// Save results to a PNG file for better visibility!
	utils.DrawRegressionToPngFile(
		inputValues, predictedValues, "Recurrent NN regression", "recurrent-neuralnet.png")
}
