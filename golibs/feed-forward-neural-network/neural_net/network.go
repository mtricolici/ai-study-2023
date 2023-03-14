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

func (n *FeedForwardNeuralNetwork) WeightsBiasesCount() (int, int) {
	wc := 0
	bc := 0
	for _, layer := range n.Layers {
		w, b := layer.WeightsBiasesCount()
		wc += w
		bc += b
	}

	return wc, bc
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

func (n *FeedForwardNeuralNetwork) SetWeights(weights []float64) {
	panic("Network.SetWeights() Not implemented")
}

func (n *FeedForwardNeuralNetwork) GetWeights() []float64 {
	panic("Network.GetWeights() Not implemented")
}
