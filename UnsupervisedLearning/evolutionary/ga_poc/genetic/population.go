package genetic

import (
	"math/rand"
	"sort"
)

type Population struct {
	Individuals []Individual
}

func NewPopulation(size int, genesCount int) *Population {
	individuals := make([]Individual, size)
	for i := 0; i < size; i++ {
		individuals[i] = NewIndividual(genesCount)
	}

	return &Population{Individuals: individuals}
}

func (p *Population) EvaluatePopulation(target string) {
	for i := range p.Individuals {
		p.Individuals[i].CalculateFitness(target)
	}
}

func (p *Population) SelectParents() (Individual, Individual) {
	individuals_copy := make([]Individual, len(p.Individuals))
	copy(individuals_copy, p.Individuals)

	sort.Slice(individuals_copy, func(i, j int) bool {
		return individuals_copy[i].Fitness > individuals_copy[j].Fitness
	})

	return individuals_copy[0], individuals_copy[1]
}

// Crossover performs a crossover operation between two parents to produce two offspring.
func (p *Population) Crossover(parent1 Individual, parent2 Individual, crossoverRate float64) (Individual, Individual) {

	if rand.Float64() > crossoverRate {
		return parent1, parent2
	}

	// Randomly choose a crossover point
	crossoverPoint := rand.Intn(len(parent1.Genes))

	// Create two new offspring
	child1 := Individual{Genes: "", Fitness: 0.0}
	child2 := Individual{Genes: "", Fitness: 0.0}

	// Perform crossover by swapping the genes before and after the crossover point
	child1.Genes = parent1.Genes[:crossoverPoint] + parent2.Genes[crossoverPoint:]
	child2.Genes = parent2.Genes[:crossoverPoint] + parent1.Genes[crossoverPoint:]

	return child1, child2
}

func (p *Population) FindBestIndividual() Individual {
	bestIndividual := p.Individuals[0]

	for i := 1; i < len(p.Individuals); i++ {
		if p.Individuals[i].Fitness > bestIndividual.Fitness {
			bestIndividual = p.Individuals[i]
		}
	}

	return bestIndividual
}

func (p *Population) ReplaceIndividual(index int, newIndividual Individual) {
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
