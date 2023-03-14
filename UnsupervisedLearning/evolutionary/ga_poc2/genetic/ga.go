package genetic

import (
	"fmt"
	"math/rand"
)

type GeneticAlgorithm struct {
	// How many of best individuals move to next generation
	Elitism                    int
	TournamentSize             int
	MutationRate               float64
	CrossoverRate              float64
	FitnessThreshold           float64
	FitnessFunc                GeneticFitnessFunc
	MutateGaussianDistribution bool
	population                 Population
	generation                 int
}

func NewGeneticAlgorithm(populationSize, geneLength int) GeneticAlgorithm {
	return GeneticAlgorithm{
		population:                 NewPopulation(populationSize, geneLength),
		generation:                 0,
		Elitism:                    populationSize / 10,
		TournamentSize:             populationSize / 10,
		MutationRate:               0.01,
		CrossoverRate:              0.81,
		MutateGaussianDistribution: false,
		FitnessFunc:                nil,
	}
}

func (ga *GeneticAlgorithm) Run(maxGenerations int) Individual {
	ga.verifyInputParameters()

	ga.generation = 1

	for ga.generation < maxGenerations {

		ga.runOneGeneration()

		bestIndividual := ga.population.individuals[0]
		if bestIndividual.GetFitness() >= ga.FitnessThreshold {
			fmt.Println("best.fitness >= threshold !!! stop genetic evolution ;)")
			return bestIndividual
		}

		fmt.Printf("Generation %d best fitness=%f\n", ga.generation, bestIndividual.GetFitness())

		ga.generation++
	}

	ga.population.Sort()
	return ga.population.individuals[0]
}

func (ga *GeneticAlgorithm) runOneGeneration() {

	ga.population.CalculateFitness(ga.FitnessFunc)
	ga.population.Sort() // sort by fitness descendently

	// for ex ga.Elitism=3
	// find 3 best individuals and add them to next generation
	newIndividuals := make([]Individual, ga.Elitism)
	for i := 0; i < ga.Elitism; i++ {
		newIndividuals[i] = ga.population.individuals[i].Clone()
	}

	for len(newIndividuals) < ga.population.GetSize() {
		parent1 := ga.population.SelectTournament(ga.TournamentSize)
		parent2 := ga.population.SelectTournament(ga.TournamentSize)
		if rand.Float64() < ga.CrossoverRate {
			child := parent1.Crossover(parent2)
			child.Mutate(ga.MutationRate, ga.MutateGaussianDistribution)
			child.CalculateFitness(ga.FitnessFunc)
			newIndividuals = append(newIndividuals, child)
		} else {
			newIndividuals = append(newIndividuals, parent1.Clone(), parent2.Clone())
		}
	}

	ga.population.Replace(newIndividuals)
}

func (ga *GeneticAlgorithm) verifyInputParameters() {
	if ga.FitnessThreshold == 0.0 {
		panic("GA.FitnessThreshold not defined!")
	}

	if ga.FitnessFunc == nil {
		panic("GA.FitnessFunc not defined!")
	}

	if ga.Elitism < 2 || ga.Elitism >= ga.population.GetSize() {
		panic("GA.Elitism bad value!!")
	}

	if ga.TournamentSize < 2 || ga.TournamentSize >= ga.population.GetSize() {
		panic("GA.TournamentSize bad value!!")
	}
}
