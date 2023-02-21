package neuralnetwork

import (
	"math"
	"math/rand"
	"time"
)

type NeuralNet struct {
	numInputs             int
	numOutputs            int
	numHiddenLayers       int
	neuronsPerHiddenLayer int
	weights               [][][]float64
}

func NewNeuralNet(numInputs, numOutputs, numHiddenLayers, neuronsPerHiddenLayer int) *NeuralNet {
	nn := &NeuralNet{
		numInputs:             numInputs,
		numOutputs:            numOutputs,
		numHiddenLayers:       numHiddenLayers,
		neuronsPerHiddenLayer: neuronsPerHiddenLayer,
	}

	// initialize weights
	nn.weights = make([][][]float64, numHiddenLayers+1)
	// input layer to first hidden layer weights
	nn.weights[0] = make([][]float64, neuronsPerHiddenLayer)
	for i := range nn.weights[0] {
		nn.weights[0][i] = make([]float64, numInputs+1) // +1 for bias
	}
	// hidden layers to hidden layers or output layer weights
	for i := 1; i < numHiddenLayers; i++ {
		nn.weights[i] = make([][]float64, neuronsPerHiddenLayer)
		for j := range nn.weights[i] {
			nn.weights[i][j] = make([]float64, neuronsPerHiddenLayer+1) // +1 for bias
		}
	}
	// last hidden layer to output layer weights
	nn.weights[numHiddenLayers] = make([][]float64, numOutputs)
	for i := range nn.weights[numHiddenLayers] {
		nn.weights[numHiddenLayers][i] = make([]float64, neuronsPerHiddenLayer+1) // +1 for bias
	}

	return nn
}

func (nn *NeuralNet) RandomizeWeights() {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	for i := range nn.weights {
		for j := range nn.weights[i] {
			for k := range nn.weights[i][j] {
				nn.weights[i][j][k] = r.Float64()*2 - 1 // random weight between -1 and 1
			}
		}
	}
}

func (nn *NeuralNet) Compute(input []float64) []float64 {
	output, _ := nn.ComputeWithActivations(input)
	return output
}

func (nn *NeuralNet) ComputeWithActivations(input []float64) ([]float64, [][]float64) {
	if len(input) != nn.numInputs {
		panic("input size does not match network input size")
	}

	// add bias neuron to input
	input = append(input, 1.0)

	// compute activations for each layer
	activations := make([][]float64, nn.numHiddenLayers+2)
	activations[0] = input
	for i := 1; i < nn.numHiddenLayers+2; i++ {
		layerActivations := make([]float64, len(nn.weights[i-1]))
		for j, neuronWeights := range nn.weights[i-1] {
			sum := 0.0
			for k, weight := range neuronWeights {
				sum += weight * activations[i-1][k]
			}
			layerActivations[j] = sigmoid(sum)
		}
		// add bias neuron to all layers except output
		if i != nn.numHiddenLayers+1 {
			layerActivations = append(layerActivations, 1.0)
		}
		activations[i] = layerActivations
	}

	// output layer activations are the network output
	return activations[nn.numHiddenLayers+1], activations
}

func sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func (nn *NeuralNet) sigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

/*
Sample usage:
nn := NewNeuralNet(2, 1, 1, 3) // create a neural network with 2 inputs, 1 output, 1 hidden layer with 3 neurons
nn.RandomizeWeights() // randomize the weights

*/
