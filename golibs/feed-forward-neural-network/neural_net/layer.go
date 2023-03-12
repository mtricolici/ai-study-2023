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
