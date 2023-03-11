package genetic

import "math/rand"

type Individual struct {
	Genes   string
	Fitness float64
}

func NewIndividual(geneLength int) Individual {
	// Create a new individual with random genes of a given length
	genes := make([]byte, geneLength)
	for i := 0; i < geneLength; i++ {
		genes[i] = randomChar()
	}
	return Individual{Genes: string(genes)}
}

func (i *Individual) CalculateFitness(target string) {
	var matches int
	for j := 0; j < len(i.Genes); j++ {
		if i.Genes[j] == target[j] {
			matches++
		}
	}
	i.Fitness = float64(matches) // / float64(len(target))
}

// Mutate randomly changes some genes of the individual based on the mutation rate
func (i *Individual) Mutate(mutationRate float64) {
	genes := []byte(i.Genes)
	for j := 0; j < len(genes); j++ {
		if rand.Float64() < mutationRate {
			genes[j] = randomChar()
		}
	}
	i.Genes = string(genes)
}

func randomChar() byte {
	// 32-127 are printable ASCI
	return byte(rand.Intn(127-32) + 32)
}
