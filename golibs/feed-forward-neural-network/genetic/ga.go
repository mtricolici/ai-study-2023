package genetic

type GeneticAlgorithm struct {
	MutationRate    float64
	CrossoverRate   float64
	FitnessFunction GeneticFitnessFunction
	Population      *Population
}

func NewGeneticAlgorithm(populationSize int, topology []int, fitnessFunction GeneticFitnessFunction) *GeneticAlgorithm {
	return &GeneticAlgorithm{
		MutationRate:    0.01,
		CrossoverRate:   0.8,
		FitnessFunction: fitnessFunction,
		Population:      NewPopulation(populationSize, topology),
	}
}

func (ga *GeneticAlgorithm) Run(maxGenerations int) *Individual {
	report := NewProgressReport(maxGenerations, ga)

	ga.Population.Evaluate(ga.FitnessFunction)

	var bestIndividual *Individual

	report.PrintHeader()

	for generation := 1; generation <= maxGenerations; generation++ {
		bestIndividual = ga.run_generation()

		report.CollectAndPrint(generation, bestIndividual)
	}

	return bestIndividual
}

// Runs a generation and return the best individual in this generation
func (ga *GeneticAlgorithm) run_generation() *Individual {
	// Select parents for mating
	parent1, parent2 := ga.Population.SelectParents()

	// Perform crossover to produce two offspring
	child1, child2 := ga.Population.Crossover(parent1, parent2, ga.CrossoverRate)

	child1.Mutate(ga.MutationRate)
	child2.Mutate(ga.MutationRate)

	// Evaluate the fitness of the offspring
	child1.CalculateFitness(ga.FitnessFunction)
	child2.CalculateFitness(ga.FitnessFunction)

	// Replace the two worst individuals in the population with the two offspring
	worstIndex1, worstIndex2 := ga.Population.FindWorstIndividuals()
	ga.Population.ReplaceIndividual(worstIndex1, child1)
	ga.Population.ReplaceIndividual(worstIndex2, child2)

	// Evaluate the fitness of the updated population
	//ga.Population.EvaluatePopulation(ga.FitnessFunction)

	// Find the best individual in the current population
	return ga.Population.FindBestIndividual()
}
