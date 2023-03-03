package ai

import (
	"log"
)

type FeedForwardNeuralNetwork struct {
	layer1 *Layer
	layer2 *Layer
}

func NewFeedForwardNeuralNetwork(numInput, numHidden, numOutput int) *FeedForwardNeuralNetwork {
	layer1 := NewLayer(numHidden, numInput)
	layer2 := NewLayer(numOutput, numHidden)

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

	// Calculate Errors and deltas
	layer2_errors := array_minus_array(target, layer2_outputs)
	layer2_delta := calculate_delta(layer2_errors, layer2_outputs)

	layer1_errors := make([]float64, len(layer1_outputs))
	n.layer2.CalculateErrors(&layer1_errors, layer2_delta)

	layer1_delta := calculate_delta(layer1_errors, layer1_outputs)

	// Update weights and bias
	n.layer2.UpdateWeights(layer1_outputs, layer2_delta, learningRate)
	n.layer1.UpdateWeights(input, layer1_delta, learningRate)

	return calcualte_error_sum(layer2_errors)
}
