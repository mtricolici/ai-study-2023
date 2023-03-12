package neural_net

import (
	"math/rand"
	"time"
)

var (
	_rnd = rand.New(rand.NewSource(time.Now().UnixNano()))
)

type Neuron struct {
	Weights []float64
	Bias    float64
}

func NewNeuron(numInputs int, randomWeights bool) *Neuron {
	neuron := Neuron{
		Weights: make([]float64, numInputs),
		Bias:    0.0,
	}

	if randomWeights {
		neuron.RandomizeWeights()
	}

	return &neuron
}

func (n *Neuron) Clone() *Neuron {
	neuron := NewNeuron(len(n.Weights), false)
	copy(neuron.Weights, n.Weights)
	neuron.Bias = n.Bias
	return neuron
}

func (n *Neuron) RandomizeWeights() {
	for i := range n.Weights {
		n.Weights[i] = _rnd.Float64()*2 - 1
	}

	n.Bias = _rnd.Float64()*0.01 - 0.005
}

func (n *Neuron) Activate(inputs []float64) float64 {
	if len(inputs) != len(n.Weights) {
		panic("neuron.activate error: number of inputs does not match number of weights")
	}

	sum := n.Bias
	for i := range inputs {
		sum += inputs[i] * n.Weights[i]
	}

	return sigmoid(sum)
}
