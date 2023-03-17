package elman

import (
	"fmt"
	"time"
)

// Elman recurrent neural network
type ElmanNetwork struct {
	layer1 *Layer
	// the output of the layer1 in the previous time step
	context []float64
	layer2  *Layer

	LearningRate float64

	TrainingReportSeconds int
}

func NewElmanNetwork(numInputs, numHiddenNeurons, numOutputs int) *ElmanNetwork {
	// layer2 takes as input: outputs from layer1 + context
	layer2_input_size := numHiddenNeurons + numHiddenNeurons

	return &ElmanNetwork{
		LearningRate:          0.03,
		TrainingReportSeconds: 3,
		layer1:                NewLayer(numInputs, numHiddenNeurons),
		layer2:                NewLayer(layer2_input_size, numOutputs),
		context:               make([]float64, numHiddenNeurons),
	}
}

func (en *ElmanNetwork) Predict(input []float64) []float64 {
	_, output2 := en.forward(input)
	return output2
}

func (en *ElmanNetwork) forward(input []float64) ([]float64, []float64) {
	if en.layer1.NumInputs != len(input) {
		msg := fmt.Sprintf(
			"ElmanNetwork:forward() BAD input. Expected: %d got %d",
			en.layer1.NumInputs, len(input))
		panic(msg)
	}

	output1 := en.layer1.Activate(input)

	layer2_Input := append(output1, en.context...)
	output2 := en.layer2.Activate(layer2_Input)

	// save context
	en.context = output1
	return output1, output2
}

func (en *ElmanNetwork) ResetContext() {
	en.context = make([]float64, en.layer1.NumNeurons)
}

func (en *ElmanNetwork) Train(inputs [][]float64, targets [][]float64, epochs int) {
	fmt.Println("Elman Network training paramters:")
	fmt.Printf("-->LearnRate: %.2f, Epochs: %d\n", en.LearningRate, epochs)
	fmt.Println("Backpropagation through time training in progress ...")

	lastPrint := time.Now()
	var avgError float64

	for epoch := 0; epoch < epochs; epoch++ {
		en.ResetContext()

		avgError = en.train_one_epoch(inputs, targets)

		if time.Since(lastPrint) > time.Duration(en.TrainingReportSeconds)*time.Second {
			progress := float64(epoch+1) / float64(epochs) * 100
			fmt.Printf("progress %3.0f%% ==> AvgError: %f\n", progress, avgError)
			lastPrint = time.Now()
		}
	}

	fmt.Printf("Training Finished. AvgError: %f\n", avgError)
}

func (en *ElmanNetwork) train_one_epoch(inputs, targets [][]float64) float64 {
	sum_errors := 0.0
	for i := range inputs {
		sum_errors += en.train_one_input(inputs[i], targets[i])
	}

	return sum_errors / float64(len(inputs))
}

func (en *ElmanNetwork) train_one_input(input []float64, target []float64) float64 {
	// Forward pass
	output1, output2 := en.forward(input)

	// Backward pass
	// 1. Calculate error and gradient for layer2
	error2 := calculate_error(target, output2)
	gradient2 := calculate_gradient(output2, error2)

	// 2. Update weights for the output layer
	input2 := append(output1, en.context...)
	en.layer2.UpdateWeights(gradient2, input2, en.LearningRate)

	// 3. Calculate error and gradient for layer1
	error1 := calculate_hidden_error(en.layer1, en.layer2, error2)
	gradient1 := calculate_gradient(output1, error1)

	// 4. Update weights for layer1
	en.layer1.UpdateWeights(gradient1, input, en.LearningRate)

	// 5 return AVG output error for this input
	return calculate_avg_error(target, output2)
}
