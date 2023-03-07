package ai

import (
	"fmt"
	"log"
	"math"
	"os"
	"time"

	"github.com/mtricolici/ai-study-2023/golibs/snake"
	"golang.org/x/text/language"
	"golang.org/x/text/message"
)

type ProgressReport struct {
	sum_scores       float64
	sum_apples       int
	sum_moves        int
	max_apples       int
	max_moves        int
	max_score        float64
	games_per_report int

	logger *log.Logger
}

func NewProgressReport() *ProgressReport {
	l := log.New(os.Stdout, "", log.LstdFlags)
	l.SetFlags(0)

	return &ProgressReport{
		logger: l}
}

func (rp *ProgressReport) PrintHeader(game *snake.SnakeGame, learning *VanillaDeepQLearning, numberOfGames int) {
	fmt.Printf("== Game\n--> Size: %dx%d. Random: %v\n", game.Size, game.Size, game.Random_initial_position)
	fmt.Printf("== Deep Q-Learning\n--> LrnRate: %f, Discount: %f, BatchSize: %d, ReplayCapacity:%d, BackpropagationIterations: %d\n",
		learning.LearningRate,
		learning.DiscountFactor,
		learning.TrainBatchSize,
		learning.ReplayCapacity,
		learning.BackpropagationIterations)
	fmt.Printf("--> NeuralNetwork Neurons per layer: %v\n", learning.network.Topology)

	p := message.NewPrinter(language.English)
	strNumberOfGames := p.Sprintf("%d", numberOfGames)

	fmt.Printf("Total games to train: %s\n", strNumberOfGames)
	rp.logger.Println("Deep Q-Learning starting ...")
}

func (rp *ProgressReport) CollectStatistics(game *snake.SnakeGame, game_score float64, apples int) {

	rp.sum_moves += game.Moves_made
	rp.sum_scores += game_score
	rp.sum_apples += apples
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

		currentPercent := float64(i) / float64(numEpisodes) * 100.0

		avgMoves := float64(rp.sum_moves) / float64(rp.games_per_report)
		avgScore := rp.sum_scores / float64(rp.games_per_report)
		avgApples := float64(rp.sum_apples) / float64(rp.games_per_report)

		randomness := epsilon * 100
		if epsilon < 0 {
			randomness = 0.0
		}

		rp.logger.SetPrefix(time.Now().Format("15:04:05 "))

		rp.logger.Printf("%2.0f%% Apples{avg:%7.3f, max:%3d, sum:%4d}. Moves{avg:%7.3f, max:%3d, sum:%5d} Score{avg:%7.3f, max:%7.3f, sum:%9.3f}. Randomness:%6.2f%%\n",
			currentPercent,
			avgApples, rp.max_apples, rp.sum_apples,
			avgMoves, rp.max_moves, rp.sum_moves,
			avgScore, rp.max_score, rp.sum_scores,
			randomness)

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

	currentPercent := float64(i) / float64(numEpisodes) * 100.0

	return math.Mod(currentPercent, percent) == 0
}
