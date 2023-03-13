package genetic

type GeneticAlgorithm struct {
	MutationRate   float64
	CrossoverRate  float64
	RandomInitRate float64
	// Number of best individuals to select for breed for next generation
	NumberOfParents int
	FitnessFunction GeneticFitnessFunction
	Population      *Population
}

func NewGeneticAlgorithm(populationSize int, topology []int, fitnessFunction GeneticFitnessFunction) *GeneticAlgorithm {
	return &GeneticAlgorithm{
		MutationRate:    0.01,
		CrossoverRate:   0.8,
		RandomInitRate:  0.1,
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
		besti, mutations_count, aliens_count := ga.run_generation()
		bestIndividual = besti

		report.CollectAndPrint(generation, bestIndividual, mutations_count, aliens_count)
	}

	return bestIndividual
}

// Runs a generation and return the best individual in this generation
func (ga *GeneticAlgorithm) run_generation() (*Individual, int, int) {

	parents := ga.Population.SelectBestIndividuals(ga.NumberOfParents)

	children, mutations_count, aliens_count := ga.Population.Breed(
		parents,
		ga.CrossoverRate,
		ga.MutationRate,
		ga.RandomInitRate,
		ga.FitnessFunction)

	ga.Population.CreateNewPopulation(parents, children)

	// Find the best individual in the current population
	return ga.Population.FindBestIndividual(), mutations_count, aliens_count
}
