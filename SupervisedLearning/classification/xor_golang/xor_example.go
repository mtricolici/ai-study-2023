package main

import (
	"fmt"
	"log"

	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/backpropagation"
	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/neural_net"
)

var (
	xorSamples = [][]float64{
		{0.0, 0.0},
		{0.0, 1.0},
		{1.0, 0.0},
		{1.0, 1.0},
	}

	xorLabels = [][]float64{
		{0.0},
		{1.0},
		{1.0},
		{0.0},
	}
)

func main() {

	neuralNetwork := neural_net.NewFeedForwardNeuralNetwork([]int{2, 5, 1}, true)

	training := backpropagation.NewBackpropagationTraining(neuralNetwork)
	training.LearningRate = 0.07
	training.StopTrainingMaxAvgError = 0.01

	log.Println("Training started ...")
	training.Train(xorSamples, xorLabels, 8000)

	log.Println("Training complete!")

	log.Println("Testing network")
	for i, sample := range xorSamples {
		result := neuralNetwork.Predict(sample)

		fmt.Printf("%.0f xor %.0f expected %.0f. Actual: %.2f\n",
			sample[0], sample[1], xorLabels[i][0], result[0])
	}
}
