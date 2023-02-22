package neuralnetwork

import (
	"MyFeedforwardNeuralNetwork/utils"
	"fmt"
)

func (nn *NeuralNet) Train(inputs [][]float64, labels [][]float64, learningRate float64, numEpochs int) {
	for epoch := 0; epoch < numEpochs; epoch++ {
		avgError := 0.0

		for i, input := range inputs {
			fmt.Println()
			output, activations := nn.ComputeWithActivations(input)
			label := labels[i]

			outputError := nn.calculateOutputError(output, label)
			avgError += utils.SumFloatsArray(outputError)

			hiddenErrors := nn.calculateHiddenErrors(outputError, activations)

			nn.updateWeights(outputError, hiddenErrors, activations, input, learningRate)
		}

		avgError = avgError / float64(len(inputs))

		fmt.Printf("Epoch %d: Error = %.6f\n", epoch+1, avgError)
	}
}

func (nn *NeuralNet) calculateOutputError(output, label []float64) []float64 {
	outputError := make([]float64, nn.numOutputs)
	for j := range output {
		outputError[j] = (label[j] - output[j]) * nn.sigmoidDerivative(output[j])
	}
	return outputError
}

func (nn *NeuralNet) calculateHiddenErrors(outputError []float64, activations [][]float64) [][]float64 {
	hiddenErrors := make([][]float64, nn.numHiddenLayers)

	// Iterate over the hidden layers in reverse order
	for i := len(hiddenErrors) - 1; i >= 0; i-- {
		hiddenErrors[i] = make([]float64, nn.neuronsPerHiddenLayer)

		// Iterate over the neurons in the current hidden layer
		for j := range hiddenErrors[i] {
			// Calculate the error for the current neuron
			var errorSum float64
			for k := 0; k < len(nn.weights[i+1]); k++ {
				if k < len(outputError) {
					errorSum += outputError[k] * nn.weights[i+1][k][j]
				}
			}

			neuronOutput := nn.sigmoidDerivative(activations[i+1][j])
			hiddenErrors[i][j] = errorSum * neuronOutput
		}
	}

	return hiddenErrors
}

func (nn *NeuralNet) updateWeights(
	outputError []float64, hiddenErrors, activations [][]float64, input []float64, learningRate float64) {

	for i := range nn.weights {
		for j, neuronWeights := range nn.weights[i] {
			for k, weight := range neuronWeights {
				var prevOutput float64

				if i == 0 {
					prevOutput = input[k]
				} else {
					prevOutput = activations[i][k]
				}

				var error float64
				if i == nn.numHiddenLayers {
					error = outputError[j]
				} else {
					error = hiddenErrors[i][j]
				}
				nn.weights[i][j][k] = weight + learningRate*error*prevOutput
			}
		}
	}
}
