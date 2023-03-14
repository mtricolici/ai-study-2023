package genetic

import "math/rand"

type GeneticFitnessFunc func([]float64) float64

type Individual struct {
	genes               []float64
	__fitness           float64
	__fitnessCalculated bool
}

func NewIndividual(length int) Individual {
	genes := make([]float64, length)
	for i := 0; i < length; i++ {
		genes[i] = rand.Float64()
	}

	return Individual{genes: genes, __fitnessCalculated: false}
}

func (ind *Individual) GetGenes() []float64 {
	return ind.genes
}

func (ind *Individual) Clone() Individual {
	newi := Individual{
		genes:               make([]float64, len(ind.genes)),
		__fitness:           ind.__fitness,
		__fitnessCalculated: ind.__fitnessCalculated,
	}
	copy(newi.genes, ind.genes)
	return newi
}

func (ind *Individual) CalculateFitness(f GeneticFitnessFunc) {
	// Make sure to not invoke calculation multiple times (performance issues)
	if !ind.__fitnessCalculated {
		ind.__fitness = f(ind.genes)
		ind.__fitnessCalculated = true
	}
}

func (ind *Individual) GetFitness() float64 {
	if !ind.__fitnessCalculated {
		panic("ind.GetFitness() FAILURE: CalculateFitness() was not called yet!")
	}
	return ind.__fitness
}

func (ind *Individual) Mutate(rate float64) {
	for i := range ind.genes {
		if rand.Float64() < rate {
			ind.genes[i] = rand.Float64()
			ind.__fitnessCalculated = false
		}
	}
}

// Crossover performs crossover with another individual and returns a new individual
func (ind *Individual) Crossover(other Individual) Individual {
	childGenes := make([]float64, len(ind.genes))
	for i := range ind.genes {
		if rand.Float64() < 0.5 {
			childGenes[i] = ind.genes[i]
		} else {
			childGenes[i] = other.genes[i]
		}
	}
	return Individual{
		genes:               childGenes,
		__fitness:           0.0,
		__fitnessCalculated: false,
	}
}
