package elman

import (
	"math/rand"

	"github.com/mtricolici/ai-study-2023/golibs/elman-network/emath"
)

type Neuron struct {
	Weights []float64
	Bias    float64
}

func NewNeuron(numInputs int) *Neuron {
	weights := make([]float64, numInputs)
	for i := range weights {
		weights[i] = rand.NormFloat64()
	}
	bias := rand.NormFloat64()
	return &Neuron{weights, bias}
}

func (n *Neuron) Activate(inputs []float64) float64 {
	sum := n.Bias
	for i, input := range inputs {
		sum += input * n.Weights[i]
	}
	return emath.Sigmoid(sum)
}
