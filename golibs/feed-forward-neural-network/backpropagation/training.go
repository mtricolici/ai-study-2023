package backpropagation

import (
	"log"

	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/neural_net"
)

type BackpropagationTraining struct {
	network                 *neural_net.FeedForwardNeuralNetwork
	LearningRate            float64
	StopTrainingMaxAvgError float64
	Verbose                 bool
}

func NewBackpropagationTraining(network *neural_net.FeedForwardNeuralNetwork) *BackpropagationTraining {
	// create training with some default parameters
	return &BackpropagationTraining{
		network:                 network,
		LearningRate:            0.1,
		StopTrainingMaxAvgError: 0.01,
		Verbose:                 true,
	}
}

func (t *BackpropagationTraining) Train(
	inputs [][]float64, targets [][]float64,
	numEpochs int) {

	t.log("Max Iterations: %d, Stop when avgError < %f\n", numEpochs, t.StopTrainingMaxAvgError)

	for epoch := 0; epoch < numEpochs; epoch++ {
		avgError := t.train_epoch(inputs, targets)

		t.log("Epoch %d of %d, Average Error: %f\n", epoch+1, numEpochs, avgError)

		if avgError < t.StopTrainingMaxAvgError {
			t.log("Average error is small enough. Stop training")
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

func (t *BackpropagationTraining) activate_layers(input []float64) [][]float64 {
	inp := input
	result := make([][]float64, len(t.network.Layers))

	for i, l := range t.network.Layers {
		out := l.Activate(inp)
		result[i] = out
		inp = out // output from layer 1 is input for layer 2 ..
	}

	return result
}

func (t *BackpropagationTraining) train_input(
	input []float64, target []float64) float64 {

	lcount := len(t.network.Layers)

	// Forward pass
	outputs := t.activate_layers(input)

	// Calculate errors and delta for each layer in reverse order

	errors := make([][]float64, len(t.network.Layers))
	delta := make([][]float64, len(t.network.Layers))

	for i := lcount - 1; i >= 0; i-- {
		if i == lcount-1 {
			// this is the output layer !
			errors[i] = array_minus_array(target, outputs[i])
		} else {
			size := len(outputs[i])
			errors[i] = t.calculate_layer_errors(size, &t.network.Layers[i+1], delta[i+1])
		}

		delta[i] = calculate_delta(errors[i], outputs[i])
	}

	// Update Weights and Biases in reverse order
	for i := lcount - 1; i >= 0; i-- {
		if i == 0 {
			t.update_weights(&t.network.Layers[i], input, delta[i])
		} else {
			t.update_weights(&t.network.Layers[i], outputs[i-1], delta[i])
		}
	}

	return calcualte_error_sum(errors[lcount-1])
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

func (t *BackpropagationTraining) log(message string, args ...any) {
	if t.Verbose {
		log.Printf(message, args...)
	}
}
