package ai

import (
	"fmt"
	"log"

	"github.com/mtricolici/ai-study-2023/golibs/snake"
)

type ProgressReport struct {
	sum_scores       float64
	sum_apples       int
	sum_moves        int
	max_apples       int
	max_moves        int
	max_score        float64
	games_played     int
	games_per_report int
}

func NewProgressReport() *ProgressReport {
	return &ProgressReport{}
}

func (rp *ProgressReport) PrintHeader(game *snake.SnakeGame, learning *VanillaDeepQLearning, numberOfGames int) {
	fmt.Printf("== Game\n--> Size: %dx%d. Random: %v\n", game.Size, game.Size, game.Random_initial_position)
	fmt.Printf("== Deep Q-Learning\n--> LrnRate: %f, Discount: %f, BatchSize: %d, BackpropagationIterations: %d\n",
		learning.LearningRate, learning.DiscountFactor, learning.TrainBatchSize, learning.BackpropagationIterations)
	fmt.Printf("--> NeuralNetwork Neurons per layer: %v\n", learning.network.Topology)
	fmt.Printf("Total games to train: %d\n", numberOfGames)
	log.Println("Deep Q-Learning starting ...")
}

func (rp *ProgressReport) CollectStatistics(game *snake.SnakeGame, game_score float64, apples int) {

	rp.sum_moves += game.Moves_made
	rp.sum_scores += game_score
	rp.sum_apples += apples
	rp.games_played += 1
	rp.games_per_report += 1

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

		avgMoves := float64(rp.sum_moves) / float64(rp.games_per_report)
		avgScore := rp.sum_scores / float64(rp.games_per_report)
		avgApples := float64(rp.sum_apples) / float64(rp.games_per_report)

		log.Printf("%.2f%% - games: %8d. Apples{avg: %f, max: %d}. Moves{avg: %f, max: %d} Score{avg: %f, max: %f}. Randomness: %.4f\n",
			percent, rp.games_played,
			avgApples, rp.max_apples,
			avgMoves, rp.max_moves,
			avgScore, rp.max_score,
			epsilon)

		// reset values for next reporting calculation
		rp.max_apples = 0
		rp.max_moves = 0
		rp.max_score = 0
		rp.sum_apples = 0
		rp.sum_moves = 0
		rp.sum_scores = 0
		rp.games_per_report = 0
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
