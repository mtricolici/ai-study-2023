package ai

import (
	"fmt"

	"github.com/mtricolici/ai-study-2023/golibs/snake"
)

type ProgressReport struct {
	// Variables per training
	sum_scores float64
	sum_apples int
	sum_moves  int

	// Variables per report
	max_apples   int
	max_moves    int
	max_score    float64
	games_played int
}

func NewProgressReport() *ProgressReport {
	return &ProgressReport{}
}

func (rp *ProgressReport) CollectStatistics(game *snake.SnakeGame, game_score float64, apples int) {

	rp.sum_moves += game.Moves_made
	rp.sum_scores += game_score
	rp.sum_apples += apples
	rp.games_played += 1

	if rp.max_score < game_score {
		rp.max_score = game_score
	}

	if rp.max_apples < apples {
		rp.max_apples = apples
	}

	if rp.max_moves < game.Moves_made {
		rp.max_moves = game.Moves_made
	}
}

func (rp *ProgressReport) PrintProgress(i, numEpisodes int, percent, epsilon float64) {
	if rp.shouldPrintProgress(i, numEpisodes, percent) {

		avgMoves := float64(rp.sum_moves) / float64(i)
		avgScore := rp.sum_scores / float64(i)
		avgApples := float64(rp.sum_apples) / float64(i)

		fmt.Printf("%.2f%% - games: %8d. Apples{avg: %f, max: %d}. Moves{avg: %f, max: %d} Score{avg: %f, max: %f}. Randomness: %.4f\n",
			percent, rp.games_played,
			avgApples, rp.max_apples,
			avgMoves, rp.max_moves,
			avgScore, rp.max_score,
			epsilon)

		// reset max values for next reporting max calculation
		rp.max_apples = 0
		rp.max_moves = 0
		rp.max_score = 0
	}
}

func (rp *ProgressReport) shouldPrintProgress(i, numEpisodes int, percent float64) bool {
	if i == 0 {
		return false
	}

	if i == numEpisodes-1 {
		return true
	}

	threshold := percent / 100.0
	prevProp := float64(i-1) / float64(numEpisodes)
	currentProp := float64(i) / float64(numEpisodes)

	return currentProp >= threshold && prevProp < threshold
}
