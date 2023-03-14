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
	topology       = []int{2, 5, 1}

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

func calculateWeightsCountForTopology() int {
	network := neural_net.NewFeedForwardNeuralNetwork(topology, false)
	return len(network.GetWeights())
}

func xorFitnessFunction(weights []float64) float64 {
	fitness := 0.0

	network := neural_net.NewFeedForwardNeuralNetwork(topology, false)
	network.SetWeights(weights)

	for i, sample := range xorSamples {
		expectedValue := xorLabels[i][0]
		value := network.Predict(sample)[0]

		diff := math.Abs(expectedValue-value) * 100.0

		fitness += 100.0 - diff
	}

	return fitness // math.Pow(1.6, float64(fitness))
}

func main() {
	genesCount := calculateWeightsCountForTopology()
	ga := genetic.NewGeneticAlgorithm(populationSize, genesCount)
	ga.Elitism = 5
	ga.TournamentSize = 5
	ga.CrossoverRate = 0.8
	ga.MutationRate = 0.01
	ga.MutateGaussianDistribution = true
	ga.FitnessThreshold = 9_999_999 //TODO: define this
	ga.FitnessFunc = xorFitnessFunction

	best := ga.Run(maxGenerations)
	fmt.Println("\nTraining complete! Let's test the network")

	bestNetwork := neural_net.NewFeedForwardNeuralNetwork(topology, false)
	bestNetwork.SetWeights(best.GetGenes())

	for i, sample := range xorSamples {
		result := bestNetwork.Predict(sample)

		fmt.Printf("%.0f xor %.0f expected %.0f. Actual: %.2f\n",
			sample[0], sample[1], xorLabels[i][0], result[0])
	}
}
