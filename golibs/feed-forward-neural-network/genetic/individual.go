package genetic

import (
	"math/rand"
	"time"

	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/neural_net"
)

type GeneticFitnessFunction func(*neural_net.FeedForwardNeuralNetwork) float64

type Individual struct {
	Network *neural_net.FeedForwardNeuralNetwork
	Fitness float64
}

var (
	_rnd = rand.New(rand.NewSource(time.Now().UnixNano()))
)

func NewIndividual(neurons []int, randomWeights bool) *Individual {
	return &Individual{
		Network: neural_net.NewFeedForwardNeuralNetwork(neurons, randomWeights),
		Fitness: 0.0,
	}
}

func (ind *Individual) Clone() *Individual {
	return &Individual{
		Network: ind.Network.Clone(),
		Fitness: ind.Fitness,
	}
}

func (ind *Individual) CalculateFitness(fitnessFunction GeneticFitnessFunction) {
	ind.Fitness = fitnessFunction(ind.Network)
}

func (ind *Individual) Mutate(mutationRate float64) {
	for _, layer := range ind.Network.Layers {
		for _, neuron := range layer.Neurons {
			if _rnd.Float64() < mutationRate {
				// we need to invoke mutation to either one of weights or bias!

				if _rnd.Intn(2) == 1 {
					neuron.Bias = ind.mutateFloat(neuron.Bias)
				} else {
					ri := _rnd.Intn(len(neuron.Weights))
					neuron.Weights[ri] = ind.mutateFloat(neuron.Weights[ri])
				}
			}
		}
	}
}

func (ind *Individual) mutateFloat(originalValue float64) float64 {
	// Generate a random value between -1.0 and 1.0.
	mutation := (_rnd.Float64() * 2.0) - 1.0

	// Add a small amount of noise
	newValue := originalValue + mutation*0.1

	// Check value to be in range -1 .. 1
	if newValue > 1.0 {
		return 1.0
	}

	if newValue < -1.0 {
		return -1.0
	}

	return newValue
}
