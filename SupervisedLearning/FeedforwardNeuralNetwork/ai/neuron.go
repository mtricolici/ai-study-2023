package ai

import (
	"MyFeedforwardNeuralNetwork/utils"
	"math/rand"
	"time"
)

type Neuron struct {
	weights        []float64 // the neuron's weight values
	bias           float64   // the neuron's bias value
	activationType ActivationType
}

func NewNeuron(numInputs int, activationType ActivationType) *Neuron {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	weights := make([]float64, numInputs)
	for i := range weights {
		// Initialize each weight to a random value between -1 and 1
		weights[i] = r.Float64()*2 - 1
	}

	//bias := r.Float64()*2 - 1 // Initialize bias to a random value between -1 and 1
	bias := r.Float64()*0.01 - 0.005

	return &Neuron{weights, bias, activationType}
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

	return n.invokeActivationFunction(sum)
}

func (n *Neuron) invokeActivationFunction(x float64) float64 {

	switch n.activationType {
	case ActivationLinear:
		return x
	case ActivationSigmoid:
		return utils.Sigmoid(x)
	case ActivationRelu:
		return utils.Relu(x)
	case ActivationTanh:
		return utils.Tanh(x)
	case ActivationElu:
		return utils.Elu(x)
	}

	//TODO: figure out how to use SoftMax
	panic("Unsupported activation type found")
}
