package genetic

import "sort"

type Population struct {
	Individuals []*Individual
	Size        int
}

func NewPopulation(populationSize int, neurons []int, randomWeights bool) *Population {
	individuals := make([]*Individual, populationSize)
	for i := 0; i < populationSize; i++ {
		individuals[i] = NewIndividual(neurons, randomWeights)
	}

	return &Population{Individuals: individuals, Size: populationSize}
}

func (p *Population) EvaluatePopulation(fitnessFunction GeneticFitnessFunction) {
	for i := range p.Individuals {
		p.Individuals[i].CalculateFitness(fitnessFunction)
	}
}

func (p *Population) SelectParents() (*Individual, *Individual) {
	inds := make([]*Individual, len(p.Individuals))
	copy(inds, p.Individuals)

	sort.Slice(inds, func(i, j int) bool {
		return inds[i].Fitness > inds[j].Fitness
	})

	return inds[0], inds[1]
}

func (p *Population) FindBestIndividual() *Individual {
	bestIndividual := p.Individuals[0]

	for i := 1; i < len(p.Individuals); i++ {
		if p.Individuals[i].Fitness > bestIndividual.Fitness {
			bestIndividual = p.Individuals[i]
		}
	}

	return bestIndividual
}

func (p *Population) ReplaceIndividual(index int, newIndividual *Individual) {
	// Replace the individual at the specified index with the new individual
	p.Individuals[index] = newIndividual
}

func (p *Population) FindWorstIndividuals() (int, int) {
	// Initialize the indices of the two worst individuals to the first two individuals in the population
	worstIndex1 := 0
	worstIndex2 := 1

	// Iterate over the remaining individuals in the population
	for i := 2; i < len(p.Individuals); i++ {
		// If the current individual has a lower fitness than both of the worst individuals, update the worst indices
		if p.Individuals[i].Fitness < p.Individuals[worstIndex1].Fitness {
			worstIndex2 = worstIndex1
			worstIndex1 = i
		} else if p.Individuals[i].Fitness < p.Individuals[worstIndex2].Fitness {
			worstIndex2 = i
		}
	}

	// Return the indices of the two worst individuals
	return worstIndex1, worstIndex2
}

func (p *Population) Crossover(parent1 *Individual, parent2 *Individual) (*Individual, *Individual) {
	panic("not implemented")
}
