package neuralnetwork

func (nn *NeuralNet) Train(inputs [][]float64, labels [][]float64, learningRate float64, numEpochs int) {
	for epoch := 0; epoch < numEpochs; epoch++ {
		for i, input := range inputs {
			output, activations := nn.ComputeWithActivations(input)
			label := labels[i]

			outputError := nn.calculateOutputError(output, label)

			hiddenErrors := nn.calculateHiddenErrors(outputError, activations)

			nn.updateWeights(outputError, hiddenErrors, activations, input, learningRate)
		}
	}
}

func (nn *NeuralNet) calculateOutputError(output, label []float64) []float64 {
	outputError := make([]float64, nn.numOutputs)
	for j := range output {
		outputError[j] = (label[j] - output[j]) * nn.sigmoidDerivative(output[j])
	}
	return outputError
}

func (nn *NeuralNet) calculateHiddenErrors(outputError []float64, activations [][]float64) [][]float64 {
	hiddenErrors := make([][]float64, nn.numHiddenLayers)
	for i := range hiddenErrors {
		hiddenErrors[i] = make([]float64, nn.neuronsPerHiddenLayer)
	}
	for i := nn.numHiddenLayers - 1; i >= 0; i-- {
		for j := range nn.weights[i] {
			errorSum := 0.0
			for k := range nn.weights[i][j] {
				if i == nn.numHiddenLayers-1 {
					errorSum += outputError[k] * nn.weights[i+1][k][j]
				} else {
					errorSum += hiddenErrors[i+1][k] * nn.weights[i+1][k][j]
				}
			}
			neuronOutput := nn.sigmoidDerivative(activations[i+1][j])
			hiddenErrors[i][j] = errorSum * neuronOutput
		}
	}
	return hiddenErrors
}

func (nn *NeuralNet) updateWeights(
	outputError []float64, hiddenErrors, activations [][]float64, input []float64, learningRate float64) {

	for i := range nn.weights {
		for j, neuronWeights := range nn.weights[i] {
			for k, weight := range neuronWeights {
				var prevOutput float64

				if i == 0 {
					prevOutput = input[k]
				} else {
					prevOutput = activations[i][k]
				}

				var error float64
				if i == nn.numHiddenLayers {
					error = outputError[j]
				} else {
					error = hiddenErrors[i][j]
				}
				nn.weights[i][j][k] = weight + learningRate*error*prevOutput
			}
		}
	}
}
