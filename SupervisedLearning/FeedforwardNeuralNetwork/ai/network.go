package ai

import (
	"log"
	"math"
)

type FeedForwardNeuralNetwork struct {
	layer1 *Layer
	layer2 *Layer
}

func NewFeedForwardNeuralNetwork(numInput, numHidden, numOutput int) *FeedForwardNeuralNetwork {
	layer1 := NewLayer(numHidden, numInput, ActivationSigmoid)
	layer2 := NewLayer(numOutput, numHidden, ActivationSigmoid)

	return &FeedForwardNeuralNetwork{
		layer1: layer1,
		layer2: layer2,
	}
}

func (n *FeedForwardNeuralNetwork) Predict(inputs []float64) []float64 {
	layer1_outputs := n.layer1.Activate(inputs)
	layer2_outputs := n.layer2.Activate(layer1_outputs)
	return layer2_outputs
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

	// Forward pass
	layer1_outputs := n.layer1.Activate(input)
	layer2_outputs := n.layer2.Activate(layer1_outputs)

	// Calculate output layer errors and deltas
	layer2_errors := make([]float64, len(layer2_outputs))
	for j, output := range layer2_outputs {
		layer2_errors[j] = target[j] - output
	}

	layer2_delta := make([]float64, len(layer2_outputs))
	for j, output := range layer2_outputs {
		layer2_delta[j] = layer2_errors[j] * output * (1 - output)
	}

	// Calculate hidden layer errors and deltas
	layer1_errors := make([]float64, len(layer1_outputs))
	for j := range layer1_outputs {
		errorSum := 0.0
		for k, delta := range layer2_delta {
			errorSum += delta * n.layer2.neurons[k].weights[j]
		}
		layer1_errors[j] = errorSum
	}

	layer1_delta := make([]float64, len(layer1_outputs))
	for j, hiddenOutput := range layer1_outputs {
		layer1_delta[j] = layer1_errors[j] * hiddenOutput * (1 - hiddenOutput)
	}

	// Update output layer weights and bias
	for j, neuron := range n.layer2.neurons {
		for k := range neuron.weights {
			neuron.weights[k] += learningRate * layer2_delta[j] * layer1_outputs[k]
		}
		neuron.bias += learningRate * layer2_delta[j]
	}

	// Update hidden layer weights and bias
	for j, neuron := range n.layer1.neurons {
		for k := range neuron.weights {
			neuron.weights[k] += learningRate * layer1_delta[j] * input[k]
		}
		neuron.bias += learningRate * layer1_delta[j]
	}

	// Calculate total error
	totalError := 0.0

	for _, err := range layer2_errors {
		totalError += math.Abs(err)
	}

	return totalError
}
