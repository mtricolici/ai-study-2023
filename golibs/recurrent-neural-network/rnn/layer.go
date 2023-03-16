package rnn

import "fmt"

// "vanilla" RNN or Elman network Layer
type Layer struct {
	Neurons    []*Neuron
	Memory     []float64
	NumInputs  int
	NumNeurons int
}

func NewLayer(numNeurons, numInputs int, randomWeights bool) *Layer {
	neurons := make([]*Neuron, numNeurons)
	memory := make([]float64, numNeurons)

	for i := range neurons {
		neurons[i] = NewNeuron(numInputs+numNeurons, randomWeights)
	}

	return &Layer{
		Neurons:    neurons,
		Memory:     memory,
		NumInputs:  numInputs,
		NumNeurons: numNeurons,
	}
}

func (l *Layer) Activate(inputs []float64) []float64 {
	if l.NumInputs != len(inputs) {
		msg := fmt.Sprintf("layer.activate: bad number of inputs. Expected: %d. Got: %d", l.NumInputs, len(inputs))
		panic(msg)
	}

	outputs := make([]float64, l.NumNeurons)

	combinedInput := append(inputs, l.Memory...)

	for i, neuron := range l.Neurons {
		outputs[i] = neuron.Activate(combinedInput)
	}

	// remember outputs
	copy(l.Memory, outputs)
	return outputs
}
