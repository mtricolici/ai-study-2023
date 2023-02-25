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

func (l *Layer) CalculateErrors(prevLayerOutpus, delta []float64) []float64 {

	errors := make([]float64, len(prevLayerOutpus))
	for i := range prevLayerOutpus {
		errorSum := 0.0
		for k, delta := range delta {
			errorSum += delta * l.neurons[k].weights[i]
		}
		errors[i] = errorSum
	}

	return errors
}

func (l *Layer) UpdateWeights(inputs, delta []float64, learningRate float64) {
	for i, neuron := range l.neurons {
		for k := range neuron.weights {
			neuron.weights[k] += learningRate * delta[i] * inputs[k]
		}
		neuron.bias += learningRate * delta[i]
	}
}
