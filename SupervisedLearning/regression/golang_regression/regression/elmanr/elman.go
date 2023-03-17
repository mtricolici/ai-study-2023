package elmanr

import (
	"regression_sample1/utils"

	"github.com/mtricolici/ai-study-2023/golibs/elman-network/elman"
)

type ElmanRegression struct {
	batchSize        int
	NumEpochs        int
	PredictionsCount int
	Network          *elman.ElmanNetwork
}

func NewElmanRegression(batchSize, hiddenNeurons int) *ElmanRegression {
	network := elman.NewElmanNetwork(batchSize, hiddenNeurons, 1)
	return &ElmanRegression{
		batchSize:        batchSize,
		NumEpochs:        100,
		PredictionsCount: 20,
		Network:          network,
	}
}

func (er *ElmanRegression) multiplyArrayElements(factor float64, data []float64) []float64 {
	// We know from dataset that values are bellow 1.5.
	// to normalize we give factor = 0.5 (i.e. devide by 2)
	// to de-normalize we give factor = 2 (i.e. multiply by 2)
	result := make([]float64, len(data))

	for i := 0; i < len(data); i++ {
		result[i] = data[i] * factor
	}

	return result
}

func (er *ElmanRegression) prepareTrainingData(data []float64) ([][]float64, [][]float64) {
	inputs := make([][]float64, len(data)-er.batchSize)
	targets := make([][]float64, len(data)-er.batchSize)

	for i := 0; i < len(data)-er.batchSize; i++ {
		inputs[i] = er.multiplyArrayElements(0.5, data[i:i+er.batchSize])
		targets[i] = er.multiplyArrayElements(0.5, []float64{data[i+er.batchSize]})
	}

	return inputs, targets
}

func (er *ElmanRegression) TrainAndTest(inputValues []float64) {
	inputs, targets := er.prepareTrainingData(inputValues)

	er.Network.Train(inputs, targets, er.NumEpochs)

	predictions := make([]float64, er.PredictionsCount)

	input := inputValues[len(inputValues)-er.batchSize:]
	// normalize input for predictions
	input = er.multiplyArrayElements(0.5, input)

	// clear training memory context. we need new predictions
	er.Network.ResetContext()

	for i := 0; i < er.PredictionsCount; i++ {
		predicted := er.Network.Predict(input)[0]
		predictions[i] = predicted

		input = input[1:]                // Remove first element
		input = append(input, predicted) // Add predicted to the end
	}

	// de-normalize predicted values
	predictions = er.multiplyArrayElements(2.0, predictions)

	// Save results to a PNG file for better visibility!
	utils.DrawRegressionToPngFile(
		inputValues, predictions, "Elman regression", "elman.png")
}
