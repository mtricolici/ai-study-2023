package genetic

import (
	"math/rand"
	"sort"
)

type Population struct {
	individuals []Individual
}

func NewPopulation(populationSize, geneLength int) Population {
	individuals := make([]Individual, populationSize)
	for i := range individuals {
		individuals[i] = NewIndividual(geneLength)
	}
	return Population{individuals: individuals}
}

func (pop *Population) GetSize() int {
	return len(pop.individuals)
}

func (pop *Population) CalculateFitness(f GeneticFitnessFunc) {
	for i := range pop.individuals {
		pop.individuals[i].CalculateFitness(f)
	}
}

func (pop *Population) Sort() {
	sort.Slice(pop.individuals, func(i, j int) bool {
		return pop.individuals[i].GetFitness() > pop.individuals[j].GetFitness()
	})
}

func (pop *Population) SelectTournament(tournamentSize int) Individual {
	selected := make([]Individual, tournamentSize)
	for i := 0; i < tournamentSize; i++ {
		randIndex := rand.Intn(len(pop.individuals))
		selected[i] = pop.individuals[randIndex]
	}

	sort.Slice(selected, func(i, j int) bool {
		return selected[i].GetFitness() > selected[j].GetFitness()
	})

	return selected[0]
}

func (pop *Population) Replace(newIndividuals []Individual) {

	populationSize := pop.GetSize()

	// Size of newIndividuals must be >= pop.GetSize()
	if len(newIndividuals) < populationSize {
		panic("Population.Replace() FAILURE. not enough new individuals")
	}

	// sort new individuals by fitness. best fist, worst last
	sort.Slice(newIndividuals, func(i, j int) bool {
		return newIndividuals[i].GetFitness() > newIndividuals[j].GetFitness()
	})

	// replace population with new individuals (may ignore last elements if size is bigger)
	// Not a problem to ignore last - they're worst
	for i := 0; i < populationSize; i++ {
		pop.individuals[i] = newIndividuals[i]
	}
}
