package genetic

import (
	"log"
	"math"
	"os"
	"time"
)

type ProgressReport struct {
	Percent        float64
	MaxGenerations int

	ga     *GeneticAlgorithm
	logger *log.Logger
}

func NewReport(ga *GeneticAlgorithm) *ProgressReport {
	l := log.New(os.Stdout, "", log.LstdFlags)
	l.SetFlags(0)

	return &ProgressReport{
		Percent: 10.0,
		ga:      ga,
		logger:  l,
	}
}

func (rp *ProgressReport) PrintHeader() {
}

func (rp *ProgressReport) CollectAndPrint() {
	if rp.shouldPrintProgress() {
		rp.Print()
	}
}

func (rp *ProgressReport) Print() {
	progress := float64(rp.ga.Generation) / float64(rp.ga.MaxGenerations) * 100.0

	bestFitness := rp.ga.Population.Individuals[0].GetFitness()

	rp.logger.SetPrefix(time.Now().Format("15:04:05 "))
	rp.logger.Printf("%3.0f%% Generation %8d - Fitness (best: %f)",
		progress,
		rp.ga.Generation,
		bestFitness)
}

func (rp *ProgressReport) shouldPrintProgress() bool {
	if rp.ga.Generation == 1 || rp.ga.Generation == rp.ga.MaxGenerations {
		return true
	}

	currentPercent := float64(rp.ga.Generation) / float64(rp.ga.MaxGenerations) * 100.0

	return math.Mod(currentPercent, rp.Percent) == 0
}
