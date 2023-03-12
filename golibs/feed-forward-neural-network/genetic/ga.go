package genetic

type GeneticAlgorithm struct {
	MutationRate  float64
	CrossoverRate float64
	// Number of best individuals to select for breed for next generation
	NumberOfParents int
	FitnessFunction GeneticFitnessFunction
	Population      *Population
}

func NewGeneticAlgorithm(populationSize int, topology []int, fitnessFunction GeneticFitnessFunction) *GeneticAlgorithm {
	return &GeneticAlgorithm{
		MutationRate:    0.01,
		CrossoverRate:   0.8,
		NumberOfParents: 10,
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

	parents := ga.Population.SelectBestIndividuals(ga.NumberOfParents)
	children := ga.Population.Breed(parents, ga.CrossoverRate, ga.MutationRate, ga.FitnessFunction)

	ga.Population.CreateNewPopulation(parents, children)

	// Find the best individual in the current population
	return ga.Population.FindBestIndividual()
}
