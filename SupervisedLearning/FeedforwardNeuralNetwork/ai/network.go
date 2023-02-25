package ai

import (
	"log"
)

type FeedForwardNeuralNetwork struct {
	hiddenLayer *Layer
	outputLayer *Layer
}

func NewFeedForwardNeuralNetwork(numInput, numHidden, numOutput int) *FeedForwardNeuralNetwork {
	hiddenLayer := NewLayer(numHidden, numInput, ActivationSigmoid)
	outputLayer := NewLayer(numOutput, numHidden, ActivationSigmoid)

	return &FeedForwardNeuralNetwork{
		hiddenLayer: hiddenLayer,
		outputLayer: outputLayer,
	}
}

func (n *FeedForwardNeuralNetwork) Predict(inputs []float64) []float64 {
	hiddenOutputs := n.hiddenLayer.Activate(inputs)
	outputOutputs := n.outputLayer.Activate(hiddenOutputs)
	return outputOutputs
}

func (n *FeedForwardNeuralNetwork) Train(
	inputs [][]float64, targets [][]float64,
	numEpochs int, learningRate float64,
	stopTrainingMaxAvgError float64) {

	log.Printf("Max Iterations: %d, Stop when avgError < %f\n",
		numEpochs, stopTrainingMaxAvgError)

	for epoch := 0; epoch < numEpochs; epoch++ {
		avgError := n.train_epoch(inputs, targets, learningRate)

		log.Printf("Epoch %d of %d, Average Error: %f\n", epoch+1, numEpochs, avgError)
		if avgError < stopTrainingMaxAvgError {
			log.Println("Average error is small enough. Stop training")
			break
		}
	}
}

func (n *FeedForwardNeuralNetwork) train_epoch(
	inputs [][]float64, targets [][]float64,
	learningRate float64) float64 {

	totalError := 0.0
	for i := range inputs {
		totalError += n.train_input(inputs[i], targets[i], learningRate)
	}

	return totalError / float64(len(inputs))
}

func (n *FeedForwardNeuralNetwork) train_input(
	input []float64, target []float64,
	learningRate float64) float64 {

	totalError := 0.0

	// Forward pass
	hiddenOutputs := n.hiddenLayer.Activate(input)
	outputOutputs := n.outputLayer.Activate(hiddenOutputs)

	// Calculate output layer errors and deltas
	outputErrors := make([]float64, len(outputOutputs))
	for j, output := range outputOutputs {
		outputErrors[j] = target[j] - output
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
			hiddenNeuron.weights[k] += learningRate * hiddenDeltas[j] * input[k]
		}
		hiddenNeuron.bias += learningRate * hiddenDeltas[j]
	}

	return totalError
}
