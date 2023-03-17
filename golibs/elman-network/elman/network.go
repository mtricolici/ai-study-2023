package elman

import (
	"fmt"
)

// Elman recurrent neural network
type ElmanNetwork struct {
	layer1 *Layer
	// the output of the layer1 in the previous time step
	context []float64
	layer2  *Layer

	LearningRate float64
}

func NewElmanNetwork(numInputs, numHiddenNeurons, numOutputs int) *ElmanNetwork {
	// layer2 takes as input: outputs from layer1 + context
	layer2_input_size := numHiddenNeurons + numHiddenNeurons

	return &ElmanNetwork{
		LearningRate: 0.03,
		layer1:       NewLayer(numInputs, numHiddenNeurons),
		layer2:       NewLayer(layer2_input_size, numOutputs),
		context:      make([]float64, numHiddenNeurons),
	}
}

func (en *ElmanNetwork) Predict(input []float64) []float64 {
	_, output2 := en.forward(input)
	return output2
}

func (en *ElmanNetwork) forward(input []float64) ([]float64, []float64) {
	if en.layer1.NumInputs != len(input) {
		msg := fmt.Sprintf(
			"ElmanNetwork:forward() BAD input. Expected: %d got %d",
			en.layer1.NumInputs, len(input))
		panic(msg)
	}

	output1 := en.layer1.Activate(input)

	layer2_Input := append(output1, en.context...)
	output2 := en.layer2.Activate(layer2_Input)

	// save context
	en.context = output1
	return output1, output2
}

func (en *ElmanNetwork) ResetContext() {
	en.context = make([]float64, en.layer1.NumNeurons)
}

func (en *ElmanNetwork) Train(inputs [][]float64, targets [][]float64, epochs int) {
	fmt.Printf("Elman BPTT starting. \nLearningRate %f\n", en.LearningRate)
	for epoch := 0; epoch < epochs; epoch++ {
		en.ResetContext()

		sum_errors := 0.0
		for i := range inputs {
			sum_errors += en.train_iteration(inputs[i], targets[i])
		}

		avgEror := sum_errors / float64(len(inputs))
		fmt.Printf("epoch %6d. AvgError: %f\n", epoch, avgEror)
	}
}

func (en *ElmanNetwork) train_iteration(input []float64, target []float64) float64 {
	// Forward pass
	output1, output2 := en.forward(input)

	// Backward pass
	// 1. Calculate error and gradient for layer2
	error2 := calculate_error(target, output2)
	gradient2 := calculate_gradient(output2, error2)

	// 2. Update weights for the output layer
	input2 := append(output1, en.context...)
	en.updateWeights(gradient2, input2, en.layer2)

	// 3. Calculate error and gradient for layer1
	error1 := calculate_hidden_error(en.layer1, en.layer2, error2)
	gradient1 := calculate_gradient(output1, error1)

	// 4. Update weights for layer1
	en.updateWeights(gradient1, input, en.layer1)

	// 5 return AVG output error for this input
	return calculate_avg_error(target, output2)
}

func (en *ElmanNetwork) updateWeights(gradient, inputs []float64, layer *Layer) {
	for i, neuron := range layer.Neurons {
		for w := range neuron.Weights {
			neuron.Weights[w] += en.LearningRate * gradient[i] * inputs[w]
		}
		neuron.Bias += en.LearningRate * gradient[i]
	}
}
