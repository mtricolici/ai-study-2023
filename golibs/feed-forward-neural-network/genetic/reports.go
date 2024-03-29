package genetic

import (
	"log"
	"os"
	"time"
)

type ProgressReport struct {
	SecondsToReport int
	lastPrintTime   time.Time
	ga              *GeneticAlgorithm
	logger          *log.Logger
}

func NewReport(ga *GeneticAlgorithm) *ProgressReport {
	l := log.New(os.Stdout, "", log.LstdFlags)
	l.SetFlags(0)

	return &ProgressReport{
		SecondsToReport: 10,
		ga:              ga,
		logger:          l,
	}
}

func (rp *ProgressReport) PrintHeader() {
	rp.logger.SetPrefix(time.Now().Format("15:04:05 "))
	rp.logger.Printf("Starting GA with max-generations: %s\n", groupDigits(rp.ga.MaxGenerations))
	rp.logger.Printf("Population: %d, Mutation: %.2f%%, CrossOver: %.2f%%, RandomSeedRate: %.2f%%\n",
		rp.ga.Population.GetSize(),
		rp.ga.MutationRate*100.0,
		rp.ga.CrossoverRate*100.0,
		rp.ga.RandomSeedRate*100.0)
	rp.logger.Printf("Individ genes: %d\n", rp.ga.Population.Individuals[0].GetGenesCount())
	rp.lastPrintTime = time.Now()
}

func (rp *ProgressReport) CollectAndPrint() {
	if rp.shouldPrintProgress() {
		rp.Print()
		rp.lastPrintTime = time.Now()
	}
}

func (rp *ProgressReport) Print() {
	progress := float64(rp.ga.Generation) / float64(rp.ga.MaxGenerations) * 100.0

	bestFitness := rp.ga.Population.Individuals[0].GetFitness()

	rp.logger.SetPrefix(time.Now().Format("15:04:05 "))
	rp.logger.Printf("%3.0f%% Generation %8d - Fitness (best: %.9f)",
		progress,
		rp.ga.Generation,
		bestFitness)
}

func (rp *ProgressReport) shouldPrintProgress() bool {
	if rp.ga.Generation == 1 || rp.ga.Generation == rp.ga.MaxGenerations {
		return true
	}

	timeDiff := time.Since(rp.lastPrintTime)

	return timeDiff >= time.Duration(rp.SecondsToReport)*time.Second
}
