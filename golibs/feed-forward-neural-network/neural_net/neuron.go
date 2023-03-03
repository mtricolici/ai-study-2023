package neural_net

import (
	"math/rand"
	"time"
)

type Neuron struct {
	Weights []float64
	Bias    float64
}

func NewNeuron(numInputs int) *Neuron {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	weights := make([]float64, numInputs)
	for i := range weights {
		weights[i] = r.Float64()*2 - 1
	}

	bias := r.Float64()*0.01 - 0.005

	return &Neuron{weights, bias}
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
