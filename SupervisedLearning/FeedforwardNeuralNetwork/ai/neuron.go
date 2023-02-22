package ai

import (
	"MyFeedforwardNeuralNetwork/utils"
	"math/rand"
	"time"
)

type Neuron struct {
	weights []float64 // the neuron's weight values
	bias    float64   // the neuron's bias value
}

func NewNeuron(numInputs int) *Neuron {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	weights := make([]float64, numInputs)
	for i := range weights {
		// Initialize each weight to a random value between -1 and 1
		weights[i] = r.Float64()*2 - 1
	}

	bias := r.Float64()*2 - 1 // Initialize bias to a random value between -1 and 1

	return &Neuron{weights, bias}
}

func (n *Neuron) Activate(inputs []float64) float64 {
	// Ensure the input length matches the number of weights
	if len(inputs) != len(n.weights) {
		panic("number of inputs does not match number of weights")
	}

	// Calculate the weighted sum of inputs and weights
	sum := n.bias
	for i := range inputs {
		sum += inputs[i] * n.weights[i]
	}

	return utils.Sigmoid(sum)
}
