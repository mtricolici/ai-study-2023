package neural_net

type FeedForwardNeuralNetwork struct {
	Layers   []*Layer
	Topology []int
}

func NewFeedForwardNeuralNetwork(neurons []int, randomWeights bool) *FeedForwardNeuralNetwork {
	if len(neurons) < 2 {
		panic("NewFeedForwardNeuralNetwork: at least 1 layer is required (2 arguments - number of inputs and number of outputs)")
	}

	layers := make([]*Layer, len(neurons)-1)

	for i := 0; i < len(neurons)-1; i++ {
		layers[i] = NewLayer(neurons[i+1], neurons[i], randomWeights)
	}

	return &FeedForwardNeuralNetwork{
		Layers:   layers,
		Topology: neurons,
	}
}

func (n *FeedForwardNeuralNetwork) Clone() *FeedForwardNeuralNetwork {
	network := &FeedForwardNeuralNetwork{
		Layers:   make([]*Layer, len(n.Layers)),
		Topology: n.Topology, // No need to copy this. This is suppose to be const
	}

	for i, layer := range n.Layers {
		network.Layers[i] = layer.Clone()
	}

	return network
}

func (n *FeedForwardNeuralNetwork) RandomizeWeights() {
	for _, layer := range n.Layers {
		layer.RandomizeWeights()
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

func (n *FeedForwardNeuralNetwork) GetWeightsCount() int {
	count := 0

	for _, layer := range n.Layers {
		count += layer.GetWeightsCount()
	}

	return count
}

func (n *FeedForwardNeuralNetwork) SetWeights(weights []float64) {
	if n.GetWeightsCount() != len(weights) {
		panic("Network.SetWeights() bad number of weights")
	}

	for i, layer := range n.Layers {
		count := layer.GetWeightsCount()
		start := i * count
		end := start + count
		layer.SetWeights(weights[start:end])
	}
}

func (n *FeedForwardNeuralNetwork) GetWeights() []float64 {
	result := make([]float64, 0)
	for _, layer := range n.Layers {
		result = append(result, layer.GetWeights()...)
	}
	return result
}
