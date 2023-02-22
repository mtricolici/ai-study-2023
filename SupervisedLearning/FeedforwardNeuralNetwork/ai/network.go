package ai

type FeedForwardNeuralNetwork struct {
	inputLayer  *Layer
	hiddenLayer *Layer
	outputLayer *Layer
}

func NewFeedForwardNeuralNetwork(numInput, numHidden, numOutput int) *FeedForwardNeuralNetwork {
	inputLayer := NewLayer(numInput, 0)
	hiddenLayer := NewLayer(numHidden, numInput)
	outputLayer := NewLayer(numOutput, numHidden)

	return &FeedForwardNeuralNetwork{
		inputLayer:  inputLayer,
		hiddenLayer: hiddenLayer,
		outputLayer: outputLayer,
	}
}

func (n *FeedForwardNeuralNetwork) Predict(inputs []float64) []float64 {
	hiddenOutputs := n.hiddenLayer.Activate(inputs)
	outputOutputs := n.outputLayer.Activate(hiddenOutputs)
	return outputOutputs
}
