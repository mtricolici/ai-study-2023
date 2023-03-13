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

func (p *Population) SelectBestIndividuals(numberOfIndividuals int) []*Individual {
	indexes := make([]int, len(p.Individuals))
	for i := range indexes {
		indexes[i] = i
	}

	sort.Slice(indexes, func(i, j int) bool {
		return p.Individuals[indexes[i]].Fitness > p.Individuals[indexes[j]].Fitness
	})

	parents := make([]*Individual, numberOfIndividuals)
	for i := 0; i < numberOfIndividuals; i++ {
		parents[i] = p.Individuals[indexes[i]]
	}

	return parents
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

func (p *Population) Breed(
	parents []*Individual,
	crossoverRate float64,
	mutationRate float64,
	randomInitRate float64,
	fitnessFunction GeneticFitnessFunction) ([]*Individual, int, int) {

	mutations := 0
	aliens := 0

	children := make([]*Individual, 0)
	for {
		parent1, parent2 := p.selectRandomParents(parents)

		if _rnd.Float64() >= crossoverRate {
			parent1 = parent1.Clone()
			parent2 = parent2.Clone()
			mutations += parent1.Mutate(mutationRate)
			mutations += parent2.Mutate(mutationRate)
			parent1.CalculateFitness(fitnessFunction)
			parent2.CalculateFitness(fitnessFunction)

			children = append(children, parent1, parent2)
		} else {
			child := p.crossover(parent1, parent2)
			mutations += child.Mutate(mutationRate)
			child.CalculateFitness(fitnessFunction)
			children = append(children, child)
		}

		if _rnd.Float64() < randomInitRate {
			// Generate a totally NEW random individual for next generation
			alien := parent1.Clone()
			alien.Network.RandomizeWeights()
			alien.CalculateFitness(fitnessFunction)

			children = append(children, alien)
			aliens += 1
		}

		if len(children) >= p.Size {
			break
		}
	}

	return children,
		mutations, // how many individuals with mutation
		aliens // how many totally new individuals
}

func (p *Population) selectRandomParents(parents []*Individual) (*Individual, *Individual) {
	parent1 := parents[_rnd.Intn(len(parents))]
	parent2 := parents[_rnd.Intn(len(parents))]

	// make sure parents are different
	for parent2 == parent1 {
		parent2 = parents[_rnd.Intn(len(parents))]
	}
	return parent1, parent2
}

func (p *Population) CreateNewPopulation(parents []*Individual, children []*Individual) {
	p.Individuals = make([]*Individual, p.Size)

	// copy best parents to new population
	for i := 0; i < len(parents); i++ {
		p.Individuals[i] = parents[i].Clone()
	}

	// copy children to new population
	ind_index := len(parents)
	child_index := 0

	for {
		if ind_index >= p.Size {
			break
		}

		p.Individuals[ind_index] = children[child_index] // No need to clone new objects
		ind_index += 1
		child_index += 1
	}
}

func (p *Population) crossover(parent1 *Individual, parent2 *Individual) *Individual {

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
