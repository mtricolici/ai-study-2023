package genetic

import (
	"log"
	"math"
	"os"
	"time"
)

type ProgressReport struct {
	maxGenerations int
	PrintPercent   float64
	ga             *GeneticAlgorithm
	logger         *log.Logger
}

func NewProgressReport(maxGenerations int, ga *GeneticAlgorithm) *ProgressReport {
	l := log.New(os.Stdout, "", log.LstdFlags)
	l.SetFlags(0)

	return &ProgressReport{
		maxGenerations: maxGenerations,
		PrintPercent:   10.0, // Print progress every 10% of generations
		ga:             ga,
		logger:         l,
	}
}

func (rp *ProgressReport) PrintHeader() {
	rp.loggerSetPrefix()
	rp.logger.Println("Start Genetic Algorithm training")
	rp.logger.Printf("--> Population: %d, Mutation: %.0f%%, Crossover: %.0f%%",
		rp.ga.Population.Size,
		rp.ga.MutationRate*100.0,
		rp.ga.CrossoverRate*100.0)
}

func (rp *ProgressReport) CollectAndPrint(generation int, bestIndividual *Individual) {
	if rp.shouldPrintProgress(generation) {
		rp.print(generation, bestIndividual)
	}
}

func (rp *ProgressReport) shouldPrintProgress(generation int) bool {
	if generation == rp.maxGenerations {
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

	rp.loggerSetPrefix()

	rp.logger.Printf("%3.0f%% - generation %7d, Fitness=> best: %.8f, avg:%.5f, min: %.5f",
		currentPercent,
		generation,
		bestIndividual.Fitness,
		fitnessAvg,
		fitnessMin)
}

func (rp *ProgressReport) loggerSetPrefix() {
	rp.logger.SetPrefix(time.Now().Format("15:04:05 "))
}
