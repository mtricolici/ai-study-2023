package elman

type Layer struct {
	Neurons    []*Neuron
	NumInputs  int
	NumNeurons int
}

func NewLayer(numInputs, numNeurons int) *Layer {
	layer := Layer{
		Neurons:    make([]*Neuron, numNeurons),
		NumInputs:  numInputs,
		NumNeurons: numNeurons,
	}

	for i := 0; i < numNeurons; i++ {
		layer.Neurons[i] = NewNeuron(numInputs)
	}

	return &layer
}

func (l *Layer) Activate(inputs []float64) []float64 {
	outputs := make([]float64, len(l.Neurons))
	for i, neuron := range l.Neurons {
		outputs[i] = neuron.Activate(inputs)
	}
	return outputs
}
