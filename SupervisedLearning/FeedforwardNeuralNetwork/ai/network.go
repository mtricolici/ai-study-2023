package ai

import (
	"log"
)

type FeedForwardNeuralNetwork struct {
	inputLayer  *Layer
	hiddenLayer *Layer
	outputLayer *Layer
}

func NewFeedForwardNeuralNetwork(numInput, numHidden, numOutput int) *FeedForwardNeuralNetwork {
	inputLayer := NewLayer(numInput, 0)
	hiddenLayer := NewLayer(numHidden, numInput)
	outputLayer := NewLayer(numOutput, numHidden)

	return &FeedForwardNeuralNetwork{
		inputLayer:  inputLayer,
		hiddenLayer: hiddenLayer,
		outputLayer: outputLayer,
	}
}

func (n *FeedForwardNeuralNetwork) Predict(inputs []float64) []float64 {
	hiddenOutputs := n.hiddenLayer.Activate(inputs)
	outputOutputs := n.outputLayer.Activate(hiddenOutputs)
	return outputOutputs
}

func (n *FeedForwardNeuralNetwork) Train(inputs [][]float64, targets [][]float64, numEpochs int, learningRate float64) {
	for epoch := 0; epoch < numEpochs; epoch++ {
		totalError := 0.0
		for i := range inputs {
			// Forward pass
			hiddenOutputs := n.hiddenLayer.Activate(inputs[i])
			outputOutputs := n.outputLayer.Activate(hiddenOutputs)

			// Calculate output layer errors and deltas
			outputErrors := make([]float64, len(outputOutputs))
			for j, output := range outputOutputs {
				outputErrors[j] = targets[i][j] - output
				totalError += outputErrors[j] * outputErrors[j]
			}
			outputDeltas := make([]float64, len(outputOutputs))
			for j, output := range outputOutputs {
				outputDeltas[j] = outputErrors[j] * output * (1 - output)
			}

			// Calculate hidden layer errors and deltas
			hiddenErrors := make([]float64, len(hiddenOutputs))
			for j := range hiddenOutputs {
				errorSum := 0.0
				for k, delta := range outputDeltas {
					errorSum += delta * n.outputLayer.neurons[k].weights[j]
				}
				hiddenErrors[j] = errorSum
			}
			hiddenDeltas := make([]float64, len(hiddenOutputs))
			for j, hiddenOutput := range hiddenOutputs {
				hiddenDeltas[j] = hiddenErrors[j] * hiddenOutput * (1 - hiddenOutput)
			}

			// Update output layer weights and bias
			for j, outputNeuron := range n.outputLayer.neurons {
				for k := range outputNeuron.weights {
					outputNeuron.weights[k] += learningRate * outputDeltas[j] * hiddenOutputs[k]
				}
				outputNeuron.bias += learningRate * outputDeltas[j]
			}

			// Update hidden layer weights and bias
			for j, hiddenNeuron := range n.hiddenLayer.neurons {
				for k := range hiddenNeuron.weights {
					hiddenNeuron.weights[k] += learningRate * hiddenDeltas[j] * inputs[i][k]
				}
				hiddenNeuron.bias += learningRate * hiddenDeltas[j]
			}
		}
		avgError := totalError / float64(len(inputs))
		log.Printf("Epoch %d, Average Error: %f\n", epoch+1, avgError)
	}
}
