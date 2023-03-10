package main

import (
	"fmt"
	"math"

	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/genetic"
	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/neural_net"
)

var (
	populationSize = 300
	maxGenerations = 500_000
	topology       = []int{2, 6, 1}

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

func xorFitnessFunction(network *neural_net.FeedForwardNeuralNetwork) float64 {
	fitness := 0.0

	for i, sample := range xorSamples {
		expectedValue := xorLabels[i][0]
		value := network.Predict(sample)[0]

		diff := math.Abs(expectedValue-value) * 100.0

		fitness += 100.0 - diff
	}

	return fitness // math.Pow(1.6, float64(fitness))
}

func main() {
	ga := genetic.NewGeneticAlgorithm(populationSize, topology, xorFitnessFunction)
	ga.MutationRate = 0.01
	ga.CrossoverRate = 0.92
	ga.NumberOfParents = 6

	best := ga.Run(maxGenerations).Network
	fmt.Println("\nTraining complete! Let's test the network")

	for i, sample := range xorSamples {
		result := best.Predict(sample)

		fmt.Printf("%.0f xor %.0f expected %.0f. Actual: %.2f\n",
			sample[0], sample[1], xorLabels[i][0], result[0])
	}
}
