package main

import (
	"fmt"
	"gapoc2/genetic"
)

var (
	populationSize = 100
	target         = []float64{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}
)

func calculateFitness(genes []float64) float64 {
	if len(genes) != len(target) {
		panic("genes and target must have the same length")
	}

	fitness := float64(len(target)) * 2.0

	for i, g := range genes {
		diff := g - target[i]
		fitness -= diff * diff
	}

	return fitness
}

func main() {
	ga := genetic.NewGeneticAlgorithm(populationSize, len(target))
	ga.Elitism = 5
	ga.TournamentSize = 5
	ga.CrossoverRate = 0.8
	ga.MutationRate = 0.01
	ga.MutateGaussianDistribution = true
	ga.FitnessThreshold = calculateFitness(target) - 0.001
	ga.FitnessFunc = calculateFitness

	best := ga.Run(5000)

	fmt.Printf("target fitness=%f\n", calculateFitness(target))
	fmt.Printf("best fitness=%f\n", best.GetFitness())

	bg := best.GetGenes()

	for i := range target {
		fmt.Printf("--> %d target=%.4f best=%.4f\n", i, target[i], bg[i])
	}
}
