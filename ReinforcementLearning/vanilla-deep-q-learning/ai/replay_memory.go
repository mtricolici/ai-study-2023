package ai

import "math/rand"

type Experience struct {
	state    []float64
	action   Action
	reward   float64
	newState []float64
	done     bool
}

type ReplayMemory struct {
	buffer []Experience
}

func NewReplayMemory() *ReplayMemory {
	return &ReplayMemory{
		buffer: make([]Experience, 0),
	}
}

func (rm *ReplayMemory) Add(state []float64, action Action, reward float64, newState []float64, done bool) {
	rm.buffer = append(rm.buffer, Experience{
		state:    state,
		action:   action,
		reward:   reward,
		newState: newState,
		done:     done,
	})
}

func (rm *ReplayMemory) Sample(batchSize int) []Experience {
	if batchSize > len(rm.buffer) {
		return nil
	}

	indices := rand.Perm(len(rm.buffer))[:batchSize]
	samples := make([]Experience, batchSize)
	for i, idx := range indices {
		samples[i] = rm.buffer[idx]
	}

	return samples
}
