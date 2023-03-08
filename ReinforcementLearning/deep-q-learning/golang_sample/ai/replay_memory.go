package ai

type Experience struct {
	state    []float64
	action   Action
	reward   float64
	newState []float64
	done     bool
}

type ReplayMemory struct {
	capacity int
	buffer   []Experience
}

func NewReplayMemory(capacity int) *ReplayMemory {
	return &ReplayMemory{
		capacity: capacity,
		buffer:   make([]Experience, 0, capacity),
	}
}

func (rm *ReplayMemory) Add(state []float64, action Action, reward float64, newState []float64, done bool) {
	if len(rm.buffer) >= rm.capacity {
		// buffer is full. Remove oldest element from it!
		rm.buffer = rm.buffer[1:]
	}

	rm.buffer = append(rm.buffer, Experience{
		state:    state,
		action:   action,
		reward:   reward,
		newState: newState,
		done:     done,
	})
}

func (rm *ReplayMemory) GetRandomSamples(batchSize int) []Experience {
	totalNumberOfElements := len(rm.buffer)
	if batchSize > totalNumberOfElements {
		return nil
	}

	samples := make([]Experience, 0, batchSize)

	selected := make(map[int]bool, totalNumberOfElements)

	for len(samples) < batchSize {
		// Choose a random index
		idx := rnd.Intn(len(rm.buffer))

		if !selected[idx] {
			selected[idx] = true
			samples = append(samples, rm.buffer[idx])
		}
	}

	return samples
}
