package ai

import (
	"math/rand"
	"time"
)

var (
	rnd = rand.New(rand.NewSource(time.Now().UnixNano()))
)

// TODO: use this later
func Allocate_nested_hash_if_needed(qt QTable, state string) {
	if _, ok := qt[state]; !ok {
		qt[state] = make(map[Action]float64)
	}
}
