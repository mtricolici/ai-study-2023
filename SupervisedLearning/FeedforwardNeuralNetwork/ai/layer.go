package ai

type Layer struct {
	neurons []*Neuron
}

func NewLayer(numNeurons, numInputs int, activationType ActivationType) *Layer {
	layer := &Layer{
		neurons: make([]*Neuron, numNeurons),
	}
	for i := range layer.neurons {
		layer.neurons[i] = NewNeuron(numInputs, activationType)
	}
	return layer
}

func (l *Layer) Activate(inputs []float64) []float64 {
	outputs := make([]float64, len(l.neurons))
	for i, neuron := range l.neurons {
		outputs[i] = neuron.Activate(inputs)
	}
	return outputs
}
