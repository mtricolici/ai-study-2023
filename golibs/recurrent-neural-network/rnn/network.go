package rnn

type VanillaRecurrentNetwork struct {
	Layers   []*Layer
	Topology []int
}

func NewVanillaRecurrentNetwork(neurons []int, randomWeights bool) *VanillaRecurrentNetwork {
	if len(neurons) < 2 {
		panic("VanillaRNN: at least 1 layer is required (2 arguments - number of inputs and number of outputs)")
	}

	numLayers := len(neurons) - 1

	layers := make([]*Layer, len(neurons)-1)

	for i := 0; i < numLayers; i++ {
		layerNrOfInputs := neurons[i]
		layerNrOfNeurons := neurons[i+1]
		layers[i] = NewLayer(layerNrOfNeurons, layerNrOfInputs, randomWeights)
	}

	return &VanillaRecurrentNetwork{
		Layers:   layers,
		Topology: neurons,
	}
}

func (net *VanillaRecurrentNetwork) Forward(input []float64) []float64 {
	x := input
	for _, layer := range net.Layers {
		x = layer.Forward(x)
	}
	return x
}
