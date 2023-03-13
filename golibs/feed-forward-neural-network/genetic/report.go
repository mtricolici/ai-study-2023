package genetic

import (
	"log"
	"math"
	"os"
	"time"
)

type ProgressReport struct {
	maxGenerations    int
	PrintPercent      float64
	total_generations int
	total_mutations   int
	total_aliens      int
	ga                *GeneticAlgorithm
	logger            *log.Logger
}

func NewProgressReport(maxGenerations int, ga *GeneticAlgorithm) *ProgressReport {
	l := log.New(os.Stdout, "", log.LstdFlags)
	l.SetFlags(0)

	return &ProgressReport{
		maxGenerations: maxGenerations,
		PrintPercent:   5.0, // Print progress every 5% of generations
		ga:             ga,
		logger:         l,
	}
}

func (rp *ProgressReport) PrintHeader() {
	rp.loggerSetPrefix()
	rp.logger.Printf("Start Genetic Algorithm training. Generations: %d\n", rp.maxGenerations)
	rp.logger.Printf("--> Population: %d, Mutation: %.2f%%, Crossover: %.2f%%",
		rp.ga.Population.Size,
		rp.ga.MutationRate*100.0,
		rp.ga.CrossoverRate*100.0)

	net := rp.ga.Population.Individuals[0].Network
	wc, bc := net.WeightsBiasesCount()

	rp.logger.Printf("--> Network Topology: %v. Weights: %d, Biases: %d", net.Topology, wc, bc)
	rp.logger.Printf("--> total weights + biases in population: %d", (wc+bc)*rp.ga.Population.Size)
}

func (rp *ProgressReport) CollectAndPrint(generation int, bestIndividual *Individual, mutations_count int, aliens_count int) {
	rp.total_generations += 1
	rp.total_mutations += mutations_count
	rp.total_aliens += aliens_count

	if rp.shouldPrintProgress(generation) {
		rp.print(generation, bestIndividual)
		rp.total_generations = 0
		rp.total_mutations = 0
		rp.total_aliens = 0
	}
}

func (rp *ProgressReport) shouldPrintProgress(generation int) bool {
	if generation == rp.maxGenerations || generation == 1 {
		return true
	}

	currentPercent := float64(generation) / float64(rp.maxGenerations) * 100.0

	return math.Mod(currentPercent, rp.PrintPercent) == 0
}

func (rp *ProgressReport) print(generation int, bestIndividual *Individual) {
	currentPercent := float64(generation) / float64(rp.maxGenerations) * 100.0

	fitnessSum := 0.0
	fitnessMin := 10_000.0

	for _, individ := range rp.ga.Population.Individuals {
		fitnessSum += individ.Fitness
		if individ.Fitness < fitnessMin {
			fitnessMin = individ.Fitness
		}
	}

	fitnessAvg := fitnessSum / float64(len(rp.ga.Population.Individuals))

	total_individuals := rp.total_generations * rp.ga.Population.Size
	mutationRate := float64(rp.total_mutations) / float64(total_individuals)
	aliensRate := float64(rp.total_aliens) / float64(total_individuals)

	rp.loggerSetPrefix()

	rp.logger.Printf("%3.0f%% - generation %7d, Fitness=> best: %f, avg:%.5f, min: %.5f. Mutation: %.2f%%, RandomNew: %.2f%%",
		currentPercent,
		generation,
		bestIndividual.Fitness,
		fitnessAvg,
		fitnessMin,
		mutationRate*100.0,
		aliensRate*100.0)
}

func (rp *ProgressReport) loggerSetPrefix() {
	rp.logger.SetPrefix(time.Now().Format("15:04:05 "))
}
