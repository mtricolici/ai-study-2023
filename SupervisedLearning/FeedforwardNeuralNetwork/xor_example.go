package main

import (
	"MyFeedforwardNeuralNetwork/ai"
	"fmt"
	"log"
)

const (
	inputNeurons         = 2
	hiddenLayerNeurons   = 5
	outputNeuronsCount   = 1
	trainingIterations   = 50000
	trainingStopMaxError = 0.01 // If error is less than this the training stops.
	learningRate         = 0.07
)

func main() {

	nn := ai.NewFeedForwardNeuralNetwork(inputNeurons, hiddenLayerNeurons, outputNeuronsCount)
	log.Println("Training started ...")

	xorSamples := [][]float64{
		{0.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.1, 1.1},
	}

	xorLabels := [][]float64{
		{0.0},
		{1.0},
		{1.0},
		{0.0},
	}

	log.Println("Training started ...")
	nn.Train(xorSamples, xorLabels, trainingIterations, learningRate, trainingStopMaxError)

	log.Println("Training complete!")

	log.Println("Testing network")
	for i, sample := range xorSamples {
		result := nn.Predict(sample)

		fmt.Printf("%f xor %f expected %f. Actual: %f\n",
			sample[0], sample[1], xorLabels[i][0], result[0])
	}
}
