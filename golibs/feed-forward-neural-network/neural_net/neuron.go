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
	weights := make([]float64, numInputs)
	bias := 0.0

	if randomWeights {
		for i := range weights {
			weights[i] = _rnd.Float64()*2 - 1
		}

		bias = _rnd.Float64()*0.01 - 0.005
	}

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
