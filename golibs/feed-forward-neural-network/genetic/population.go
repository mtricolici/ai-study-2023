package genetic

import "sort"

type Population struct {
	Individuals []*Individual
	Size        int
}

func NewPopulation(populationSize int, neurons []int) *Population {
	individuals := make([]*Individual, populationSize)
	for i := 0; i < populationSize; i++ {
		individuals[i] = NewIndividual(neurons, true)
	}

	return &Population{Individuals: individuals, Size: populationSize}
}

func (p *Population) Evaluate(fitnessFunction GeneticFitnessFunction) {
	for i := range p.Individuals {
		p.Individuals[i].CalculateFitness(fitnessFunction)
	}
}

func (p *Population) SelectParents() (*Individual, *Individual) {
	indexes := make([]int, len(p.Individuals))
	for i := range indexes {
		indexes[i] = i
	}

	sort.Slice(indexes, func(i, j int) bool {
		return p.Individuals[indexes[i]].Fitness > p.Individuals[indexes[j]].Fitness
	})

	return p.Individuals[indexes[0]], p.Individuals[indexes[1]]
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
	indexes := make([]int, len(p.Individuals))
	for i := range indexes {
		indexes[i] = i
	}

	sort.Slice(indexes, func(i, j int) bool {
		return p.Individuals[indexes[i]].Fitness < p.Individuals[indexes[j]].Fitness
	})

	return indexes[0], indexes[1]
}

func (p *Population) Crossover(parent1 *Individual, parent2 *Individual, crossoverRate float64) (*Individual, *Individual) {
	if _rnd.Float64() >= crossoverRate {
		// Example: if crossOverRate is 0.8 (i.e. 80% chance)
		// then 20% of cases do not invoke crossover, just return parents as they are
		return parent1.Clone(), parent2.Clone()
	}

	child1 := p.breed(parent1, parent2)
	child2 := p.breed(parent1, parent2)

	return child1, child2
}

func (p *Population) breed(parent1 *Individual, parent2 *Individual) *Individual {

	if !arraysEqual(parent1.Network.Topology, parent2.Network.Topology) {
		panic("Population:breed - ERROR: parents have different topology!")
	}

	ind := NewIndividual(parent1.Network.Topology, false)

	for li, p1_layer := range parent1.Network.Layers {
		p2_layer := parent2.Network.Layers[li]

		for ni, p1_neuron := range p1_layer.Neurons {
			p2_neuron := p2_layer.Neurons[ni]

			// Chose neuron bias from either parent 1 or 2
			if _rnd.Intn(2) == 1 {
				ind.Network.Layers[li].Neurons[ni].Bias = p1_neuron.Bias
			} else {
				ind.Network.Layers[li].Neurons[ni].Bias = p2_neuron.Bias
			}

			// Choose random weight from parent 1 or 2
			for wi, p1_weight := range p1_neuron.Weights {
				p2_weight := p2_neuron.Weights[wi]

				if _rnd.Intn(2) == 1 {
					ind.Network.Layers[li].Neurons[ni].Weights[wi] = p1_weight
				} else {
					ind.Network.Layers[li].Neurons[ni].Weights[wi] = p2_weight
				}
			}
		}

	}

	return ind
}

func arraysEqual(arr1 []int, arr2 []int) bool {
	if len(arr1) != len(arr2) {
		return false
	}

	for i := 0; i < len(arr1); i++ {
		if arr1[i] != arr2[i] {
			return false
		}
	}

	return true
}
