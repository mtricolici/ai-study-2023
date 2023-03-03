package neural_net

type FeedForwardNeuralNetwork struct {
	//TODO: refactoring - allow multiple hidden layers
	Layer1 *Layer
	Layer2 *Layer
}

func NewFeedForwardNeuralNetwork(numInput, numHidden, numOutput int) *FeedForwardNeuralNetwork {
	layer1 := NewLayer(numHidden, numInput)
	layer2 := NewLayer(numOutput, numHidden)

	return &FeedForwardNeuralNetwork{
		Layer1: layer1,
		Layer2: layer2,
	}
}

func (n *FeedForwardNeuralNetwork) Predict(inputs []float64) []float64 {
	layer1_outputs := n.Layer1.Activate(inputs)
	layer2_outputs := n.Layer2.Activate(layer1_outputs)
	return layer2_outputs
}
