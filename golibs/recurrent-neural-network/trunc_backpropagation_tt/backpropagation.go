package trunc_backpropagation_tt

import (
	"github.com/mtricolici/ai-study-2023/golibs/recurrent-neural-network/nnmath"
	"github.com/mtricolici/ai-study-2023/golibs/recurrent-neural-network/rnn"
)

type TruncatedBackpropagationThroughTime struct {
	Network      *rnn.VanillaRecurrentNetwork
	LearningRate float64
	TimeSteps    int
}

func NewTruncBackpropagationTT(network *rnn.VanillaRecurrentNetwork) *TruncatedBackpropagationThroughTime {
	return &TruncatedBackpropagationThroughTime{
		Network:      network,
		LearningRate: 0.01,
		TimeSteps:    5, // More steps means - the network may become too complex and difficult to train
	}
}

func (tbptt *TruncatedBackpropagationThroughTime) Train(inputs [][]float64, targets [][]float64) {
	for t := 0; t < tbptt.TimeSteps; t++ {
		netOutputs := make([][]float64, len(inputs))
		for i, input := range inputs {
			netOutputs[i] = tbptt.Network.Forward(input)
		}

		tbptt.backward(netOutputs, targets)
		tbptt.updateWeights()
	}
}

func (tbptt *TruncatedBackpropagationThroughTime) backward(outputs [][]float64, targets [][]float64) {
	// Compute output layer deltas and update biases
	for t := range outputs {
		outputLayer := tbptt.Network.Layers[len(tbptt.Network.Layers)-1]
		for j, neuron := range outputLayer.Neurons {
			delta := outputs[t][j] - targets[t][j]
			neuron.Bias -= tbptt.LearningRate * delta
			outputLayer.Memory[j] = delta
		}
	}

	// Compute hidden layer deltas and update biases
	for l := len(tbptt.Network.Layers) - 2; l >= 0; l-- {
		hiddenLayer := tbptt.Network.Layers[l]
		nextLayer := tbptt.Network.Layers[l+1]
		for i, neuron := range hiddenLayer.Neurons {
			hiddenDelta := 0.0
			for j, nextNeuron := range nextLayer.Neurons {
				hiddenDelta += nextLayer.Memory[j] * nextNeuron.Weights[i]
			}
			hiddenDelta *= nnmath.SigmoidDerivative(hiddenLayer.Memory[i])
			neuron.Bias -= tbptt.LearningRate * hiddenDelta
			hiddenLayer.Memory[i] = hiddenDelta
		}
	}
}

func (tbptt *TruncatedBackpropagationThroughTime) updateWeights() {
	for _, layer := range tbptt.Network.Layers {
		for _, neuron := range layer.Neurons {
			for i := range neuron.Weights {
				gradient := layer.Memory[i] * nnmath.SigmoidDerivative(neuron.Weights[i])
				neuron.Weights[i] -= tbptt.LearningRate * gradient
			}
		}
	}
}
