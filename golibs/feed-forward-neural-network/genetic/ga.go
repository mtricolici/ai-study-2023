package genetic

import (
	"fmt"

	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/random"
)

type GeneticAlgorithm struct {
	// How many of best individuals move to next generation
	Elitism        int
	TournamentSize int
	MutationRate   float64
	CrossoverRate  float64
	// New Random Individuals added during GA without crossover or mutation. Totally new individuals
	RandomSeedRate             float64
	FitnessThreshold           float64
	FitnessFunc                GeneticFitnessFunc
	MutateGaussianDistribution bool
	Population                 Population

	Generation     int
	MaxGenerations int

	Report *ProgressReport
}

func NewGeneticAlgorithm(populationSize, geneLength int) *GeneticAlgorithm {
	ga := GeneticAlgorithm{
		Population:                 NewPopulation(populationSize, geneLength),
		MaxGenerations:             100,
		Elitism:                    populationSize / 10,
		TournamentSize:             populationSize / 10,
		MutationRate:               0.01,
		CrossoverRate:              0.81,
		RandomSeedRate:             0.3,
		MutateGaussianDistribution: false,
		FitnessFunc:                nil,
	}

	ga.Report = NewReport(&ga)
	return &ga
}

func (ga *GeneticAlgorithm) Run() Individual {
	ga.verifyInputParameters()

	ga.Generation = 1

	ga.Report.PrintHeader()

	for ga.Generation < ga.MaxGenerations {

		ga.runOneGeneration()

		bestIndividual := ga.Population.Individuals[0]
		if bestIndividual.GetFitness() >= ga.FitnessThreshold {
			fmt.Println("best.fitness >= threshold !!! stop genetic evolution ;)")
			return bestIndividual
		}

		ga.Report.CollectAndPrint()
		// fmt.Printf("Generation %d best fitness=%f\n", ga.Generation, bestIndividual.GetFitness())

		ga.Generation++
	}

	ga.Population.Sort()
	ga.Report.Print()

	return ga.Population.Individuals[0]
}

func (ga *GeneticAlgorithm) runOneGeneration() {

	ga.Population.CalculateFitness(ga.FitnessFunc)
	ga.Population.Sort() // sort by fitness descendently

	// for ex ga.Elitism=3
	// find 3 best individuals and add them to next generation
	newIndividuals := make([]Individual, ga.Elitism)
	for i := 0; i < ga.Elitism; i++ {
		newIndividuals[i] = ga.Population.Individuals[i].Clone()
	}

	for len(newIndividuals) < ga.Population.GetSize() {
		parent1 := ga.Population.SelectTournament(ga.TournamentSize)
		parent2 := ga.Population.SelectTournament(ga.TournamentSize)
		if random.Float64() < ga.CrossoverRate {
			child := parent1.Crossover(parent2)
			child.Mutate(ga.MutationRate, ga.MutateGaussianDistribution)
			child.CalculateFitness(ga.FitnessFunc)
			newIndividuals = append(newIndividuals, child)
		} else {
			if random.Float64() < ga.RandomSeedRate {
				child := NewIndividual(parent1.GetGenesCount())
				child.CalculateFitness(ga.FitnessFunc)
				newIndividuals = append(newIndividuals, child)
			} else {
				newIndividuals = append(newIndividuals, parent1.Clone(), parent2.Clone())
			}
		}
	}

	ga.Population.Replace(newIndividuals)
	ga.Population.Sort()
}

func (ga *GeneticAlgorithm) verifyInputParameters() {
	if ga.FitnessThreshold == 0.0 {
		panic("GA.FitnessThreshold not defined!")
	}

	if ga.FitnessFunc == nil {
		panic("GA.FitnessFunc not defined!")
	}

	if ga.Elitism < 2 || ga.Elitism >= ga.Population.GetSize() {
		panic("GA.Elitism bad value!!")
	}

	if ga.TournamentSize < 2 || ga.TournamentSize >= ga.Population.GetSize() {
		panic("GA.TournamentSize bad value!!")
	}
}
