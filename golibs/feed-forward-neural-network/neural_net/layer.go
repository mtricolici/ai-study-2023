package neural_net

import "fmt"

type Layer struct {
	Neurons   []*Neuron
	NumInputs int
}

func NewLayer(numNeurons, numInputs int, randomWeights bool) *Layer {
	layer := &Layer{
		NumInputs: numInputs,
		Neurons:   make([]*Neuron, numNeurons),
	}
	for i := range layer.Neurons {
		layer.Neurons[i] = NewNeuron(numInputs, randomWeights)
	}
	return layer
}

func (l *Layer) Clone() *Layer {
	layer := &Layer{
		NumInputs: l.NumInputs,
		Neurons:   make([]*Neuron, len(l.Neurons)),
	}

	for i, neuron := range l.Neurons {
		layer.Neurons[i] = neuron.Clone()
	}

	return layer
}

func (l *Layer) RandomizeWeights() {
	for _, neuron := range l.Neurons {
		neuron.RandomizeWeights()
	}
}

func (l *Layer) WeightsBiasesCount() (int, int) {
	wc := 0
	bc := 0
	for _, neuron := range l.Neurons {
		wc += len(neuron.Weights)
		bc += 1 // each neuron has 1 bias
	}
	return wc, bc
}

func (l *Layer) Activate(inputs []float64) []float64 {
	if l.NumInputs != len(inputs) {
		msg := fmt.Sprintf("layer.activate: bad number of inputs. Expected: %d. Got: %d", l.NumInputs, len(inputs))
		panic(msg)
	}

	outputs := make([]float64, len(l.Neurons))
	for i, neuron := range l.Neurons {
		outputs[i] = neuron.Activate(inputs)
	}
	return outputs
}
