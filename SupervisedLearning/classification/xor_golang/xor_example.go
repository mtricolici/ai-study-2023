package main

import (
	"fmt"
	"log"

	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/neural_net"
)

const (
	inputNeurons         = 2
	hiddenLayerNeurons   = 5
	outputNeuronsCount   = 1
	trainingIterations   = 5000
	trainingStopMaxError = 0.01 // If error is less than this the training stops.
	learningRate         = 0.07
)

func main() {

	nn := neural_net.NewFeedForwardNeuralNetwork(inputNeurons, hiddenLayerNeurons, outputNeuronsCount)
	log.Println("Training started ...")

	xorSamples := [][]float64{
		{0.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0, 1.0},
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

		fmt.Printf("%.0f xor %.0f expected %.0f. Actual: %.2f\n",
			sample[0], sample[1], xorLabels[i][0], result[0])
	}
}
