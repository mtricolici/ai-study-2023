package genetic

import "fmt"

type GA struct {
	PopulationSize int
	MutationRate   float64
	CrossoverRate  float64
	Target         string
	Population     *Population
}

func NewGeneticAlgorithm(target string, populationSize int) *GA {
	return &GA{
		PopulationSize: populationSize,
		MutationRate:   0.01,
		CrossoverRate:  0.8,
		Target:         target,
		Population:     NewPopulation(populationSize, len(target)),
	}
}

func (ga *GA) Run(maxGenerations int) {

	// Evaluate the initial population
	ga.Population.EvaluatePopulation(ga.Target)

	// Track the best individual across all generations
	bestIndividual := ga.Population.Individuals[0]

	// Run the genetic algorithm for the specified number of generations or until convergence
	for generation := 1; generation <= maxGenerations; generation++ {
		// Select parents for mating
		parent1, parent2 := ga.Population.SelectParents()

		// Perform crossover to produce two offspring
		child1, child2 := ga.Population.Crossover(parent1, parent2, ga.CrossoverRate)

		child1.Mutate(ga.MutationRate)
		child2.Mutate(ga.MutationRate)

		// Evaluate the fitness of the offspring
		child1.CalculateFitness(ga.Target)
		child2.CalculateFitness(ga.Target)

		// Replace the two worst individuals in the population with the two offspring
		worstIndex1, worstIndex2 := ga.Population.FindWorstIndividuals()
		ga.Population.ReplaceIndividual(worstIndex1, child1)
		ga.Population.ReplaceIndividual(worstIndex2, child2)

		// Evaluate the fitness of the updated population
		ga.Population.EvaluatePopulation(ga.Target)

		// Find the best individual in the current population
		currentBestIndividual := ga.Population.FindBestIndividual()

		// Update the best individual across all generations
		if currentBestIndividual.Fitness > bestIndividual.Fitness {
			bestIndividual = currentBestIndividual
		}

		fmt.Printf("generation %d. Best output: '%s' fitness: %f \n", generation, bestIndividual.Genes, bestIndividual.Fitness)

		// Check for convergence
		if bestIndividual.Genes == ga.Target {
			fmt.Println("SOLUTION FOUND.")
			break
		}
	}

	fmt.Printf("Gentic DONE!. Best individual output: '%s'\n", bestIndividual.Genes)
}
