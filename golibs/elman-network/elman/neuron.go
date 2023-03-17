package elman

import (
	"fmt"
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
	if len(n.Weights) != len(inputs) {
		msg := fmt.Sprintf(
			"Neuron:Activate() BAD input. Expected: %d got %d",
			len(n.Weights), len(inputs))
		panic(msg)
	}
	sum := n.Bias
	for i, input := range inputs {
		sum += input * n.Weights[i]
	}
	return emath.Sigmoid(sum)
}

func (n *Neuron) UpdateWeights(inputs []float64, gradient, learningRate float64) {
	if len(n.Weights) != len(inputs) {
		msg := fmt.Sprintf(
			"Neuron:UpdateWeights() BAD inputs. Expected: %d got %d",
			len(n.Weights), len(inputs))
		panic(msg)
	}

	for i := range n.Weights {
		n.Weights[i] += learningRate * gradient * inputs[i]
	}
	n.Bias += learningRate * gradient
}
