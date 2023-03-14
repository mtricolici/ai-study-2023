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

func (l *Layer) GetWeightsCount() int {
	cnt := 0
	for _, neuron := range l.Neurons {
		cnt += neuron.GetWeightsCount()
	}
	return cnt
}

func (l *Layer) SetWeights(weights []float64) {
	layer_weights_count := l.GetWeightsCount()
	if len(weights) != layer_weights_count {
		panic("Layer.SetWeights() FAILURE. bad number of weights")
	}

	// all neurons have the same weights size!
	neurons_count := len(l.Neurons)
	neuron_weights_count := layer_weights_count / neurons_count

	for i, neuron := range l.Neurons {
		idx_start := i * neuron_weights_count
		idx_end := idx_start + neuron_weights_count
		neuron_weights := weights[idx_start:idx_end]
		neuron.SetWeights(neuron_weights)
	}

}
