package neural_net

import "fmt"

type Layer struct {
	Neurons   []*Neuron
	numInputs int
}

func NewLayer(numNeurons, numInputs int) *Layer {
	layer := &Layer{
		numInputs: numInputs,
		Neurons:   make([]*Neuron, numNeurons),
	}
	for i := range layer.Neurons {
		layer.Neurons[i] = NewNeuron(numInputs)
	}
	return layer
}

func (l *Layer) Activate(inputs []float64) []float64 {
	if l.numInputs != len(inputs) {
		msg := fmt.Sprintf("layer.activate: bad number of inputs. Expected: %d. Got: %d", l.numInputs, len(inputs))
		panic(msg)
	}

	outputs := make([]float64, len(l.Neurons))
	for i, neuron := range l.Neurons {
		outputs[i] = neuron.Activate(inputs)
	}
	return outputs
}
