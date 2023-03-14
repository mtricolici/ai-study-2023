package neural_net

import "github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/random"

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
		n.Weights[i] = random.NormFloat64()
	}

	n.Bias = random.NormFloat64()
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

func (n *Neuron) GetWeightsCount() int {
	return len(n.Weights) + 1 // 1 for bias
}

func (n *Neuron) SetWeights(weights []float64) {
	if len(weights) != n.GetWeightsCount() {
		panic("Neuron.SetWeights() FAILURE. bad number of weights")
	}

	n.Bias = weights[0]

	for i := range n.Weights {
		n.Weights[i] = weights[i+1]
	}
}

func (n *Neuron) GetWeights() []float64 {
	w := make([]float64, n.GetWeightsCount())
	w[0] = n.Bias
	for i := range n.Weights {
		w[i+1] = n.Weights[i]
	}
	return w
}
