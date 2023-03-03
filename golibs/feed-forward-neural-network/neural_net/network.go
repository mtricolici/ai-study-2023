package neural_net

type FeedForwardNeuralNetwork struct {
	Layers []Layer
}

func NewFeedForwardNeuralNetwork(neurons []int) *FeedForwardNeuralNetwork {
	if len(neurons) < 2 {
		panic("NewFeedForwardNeuralNetwork: at least 1 layer is required (2 arguments - number of inputs and number of outputs)")
	}

	layers := make([]Layer, len(neurons)-1)

	for i := 0; i < len(neurons)-1; i++ {
		layers[i] = *NewLayer(neurons[i+1], neurons[i])
	}

	return &FeedForwardNeuralNetwork{
		Layers: layers,
	}
}

func (n *FeedForwardNeuralNetwork) Predict(inputs []float64) []float64 {
	inp := inputs
	var out []float64

	for _, layer := range n.Layers {
		out = layer.Activate(inp)
		inp = out // this 'out' is input for next layer
	}

	return out
}
