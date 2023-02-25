package ai

type Layer struct {
	neurons []*Neuron
}

func NewLayer(numNeurons, numInputs int) *Layer {
	layer := &Layer{
		neurons: make([]*Neuron, numNeurons),
	}
	for i := range layer.neurons {
		layer.neurons[i] = NewNeuron(numInputs)
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

func (l *Layer) CalculateErrors(errors *[]float64, delta []float64) {
	for i, neuron := range l.neurons {
		neuron.CalculateError(errors, delta[i])
	}
}

func (l *Layer) UpdateWeights(inputs, delta []float64, learningRate float64) {
	for i, neuron := range l.neurons {
		neuron.UpdateWeights(inputs, delta[i], learningRate)
	}
}
