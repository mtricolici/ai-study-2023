package genetic

import "github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/random"

type GeneticFitnessFunc func([]float64) float64

type Individual struct {
	genes               []float64
	__fitness           float64
	__fitnessCalculated bool
}

func NewIndividual(length int) Individual {
	genes := make([]float64, length)
	for i := 0; i < length; i++ {
		genes[i] = generateRandomGene()
	}

	return Individual{genes: genes, __fitnessCalculated: false}
}

func NewIndividualFromWeights(weights []float64) Individual {
	genes := make([]float64, len(weights))
	copy(genes, weights)
	return Individual{genes: genes, __fitnessCalculated: false}
}

func (ind *Individual) GetGenesCount() int {
	return len(ind.genes)
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

func (ind *Individual) Mutate(rate float64, gaussianDistribution bool) {
	for i := range ind.genes {
		if random.Float64() < rate {

			if gaussianDistribution {
				ind.genes[i] = randGaussianDistribution(ind.genes[i])
			} else {
				ind.genes[i] = generateRandomGene()
			}

			ind.__fitnessCalculated = false
		}
	}
}

// Crossover performs crossover with another individual and returns a new individual
func (ind *Individual) Crossover(other Individual) Individual {
	childGenes := make([]float64, len(ind.genes))
	for i := range ind.genes {
		if random.Float64() < 0.5 {
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

func generateRandomGene() float64 {
	return random.NormFloat64()
}
