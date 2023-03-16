package rnn

import (
	"github.com/mtricolici/ai-study-2023/golibs/recurrent-neural-network/nnmath"
	"github.com/mtricolici/ai-study-2023/golibs/recurrent-neural-network/random"
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

func (n *Neuron) RandomizeWeights() {
	for i := range n.Weights {
		n.Weights[i] = random.NormFloat64()
	}

	n.Bias = random.NormFloat64()
}

func (n *Neuron) Forward(inputs []float64) float64 {
	if len(inputs) != len(n.Weights) {
		panic("Neuron.Forward error: number of inputs does not match number of weights")
	}

	sum := n.Bias
	for i := range inputs {
		sum += inputs[i] * n.Weights[i]
	}

	return nnmath.Sigmoid(sum)
}
