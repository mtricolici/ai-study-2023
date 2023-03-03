package backpropagation

import (
	"log"

	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/neural_net"
)

type BackpropagationTraining struct {
	network                 *neural_net.FeedForwardNeuralNetwork
	LearningRate            float64
	StopTrainingMaxAvgError float64
}

func NewBackpropagationTraining(network *neural_net.FeedForwardNeuralNetwork) *BackpropagationTraining {
	// create training with some default parameters
	return &BackpropagationTraining{
		network:                 network,
		LearningRate:            0.1,
		StopTrainingMaxAvgError: 0.01,
	}
}

func (t *BackpropagationTraining) Train(
	inputs [][]float64, targets [][]float64,
	numEpochs int) {

	log.Printf("Max Iterations: %d, Stop when avgError < %f\n",
		numEpochs, t.StopTrainingMaxAvgError)

	for epoch := 0; epoch < numEpochs; epoch++ {
		avgError := t.train_epoch(inputs, targets)

		log.Printf("Epoch %d of %d, Average Error: %f\n", epoch+1, numEpochs, avgError)
		if avgError < t.StopTrainingMaxAvgError {
			log.Println("Average error is small enough. Stop training")
			break
		}
	}
}

func (t *BackpropagationTraining) train_epoch(
	inputs [][]float64, targets [][]float64) float64 {

	totalError := 0.0
	for i := range inputs {
		totalError += t.train_input(inputs[i], targets[i])
	}

	return totalError / float64(len(inputs))
}

func (t *BackpropagationTraining) train_input(
	input []float64, target []float64) float64 {

	// Forward pass
	layer1_outputs := t.network.Layer1.Activate(input)
	layer2_outputs := t.network.Layer2.Activate(layer1_outputs)

	// Calculate Errors and deltas for layer2
	layer2_errors := array_minus_array(target, layer2_outputs)
	layer2_delta := calculate_delta(layer2_errors, layer2_outputs)

	// Calculate errors/deltas for layer1
	layer1_errors := t.calculate_layer_errors(len(layer1_outputs), t.network.Layer2, layer2_delta)
	layer1_delta := calculate_delta(layer1_errors, layer1_outputs)

	// Update weights and bias
	t.update_weights(t.network.Layer2, layer1_outputs, layer2_delta)
	t.update_weights(t.network.Layer1, input, layer1_delta)

	return calcualte_error_sum(layer2_errors)
}

func (t *BackpropagationTraining) calculate_layer_errors(size int, layer *neural_net.Layer, delta []float64) []float64 {
	errors := make([]float64, size)

	for i, neuron := range layer.Neurons {
		for e := 0; e < len(neuron.Weights); e++ {
			errors[e] += delta[i] * neuron.Weights[e]
		}
	}

	return errors
}

func (t *BackpropagationTraining) update_weights(layer *neural_net.Layer, inputs, delta []float64) {

	for neuronIndex, neuron := range layer.Neurons {
		for wi := range neuron.Weights {
			neuron.Weights[wi] += t.LearningRate * delta[neuronIndex] * inputs[wi]
		}

		neuron.Bias += t.LearningRate * delta[neuronIndex]
	}
}
