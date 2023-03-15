package main

import (
	"fmt"
	"os"
	"snakega/cimport"
	"time"

	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/genetic"
	"github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network/neural_net"
	"github.com/mtricolici/ai-study-2023/golibs/snake"
)

var (
	// Genetic Algorihm parameters
	populationSize     = 300
	maxGenerations     = 10_000
	numberOfGames2Play = 10
	ga_elitism         = 10 // How many best individuals move to next generation
	ga_tournamentSize  = 10 // Nr of best individs to select for breeding
	ga_crossoverRate   = 0.8
	ga_mutationRate    = 0.02
	ga_randomSeedRate  = 0.3
	ga_mutateGaussian  = true
	ga_report_seconds  = 25 // Print progress every 25 seconds

	// Neural Network parameters
	hiddenLayerNeurons  = 20
	outputNeurons       = 3 // Turn Left, Keep Forward, Turn Right
	topology            []int
	networkWeightsCount int
	snakeLimitedView    bool = true // snake does not view entire board
	saveNetworkLocation      = "/home/boris/temp/genetic.brain.zzz"

	// Snake Game Parameters and variables
	game                         *snake.SnakeGame
	game_size                    = 8
	game_random_position         = true
	game_max_moves_without_apple = game_size * 3

	score_apple           = 10.0  // reward for eating an apple
	score_move_from_apple = -0.2  // reward move from apple
	score_move_to_apple   = 0.1   // reward move to apple
	score_die             = -30.0 // score for game end
	score_move_in_circles = -10.0 // score if moves in circles without eating
)

func initializeGame() {
	game = snake.NewSnakeGame(game_size, game_random_position)
	game.Small_State_for_Neural = snakeLimitedView
	game.Reward_apple = score_apple
	game.Reward_move_from_apple = score_move_from_apple
	game.Reward_move_to_apple = score_move_to_apple
	game.Reward_die = score_die

	state := game.GetStateForNeuralNetwork()

	topology = make([]int, 3) // just 1 hidden layer I hope this is enough
	topology[0] = len(state)  // << number of inputs
	topology[1] = hiddenLayerNeurons
	topology[2] = outputNeurons // << number of outputs

	sampleNetwork := neural_net.NewFeedForwardNeuralNetwork(topology, false)
	networkWeightsCount = sampleNetwork.GetWeightsCount()
}

func snakeFitnessFunction(weights []float64) float64 {
	network := neural_net.NewFeedForwardNeuralNetwork(topology, false)
	network.SetWeights(weights)

	score := 0.0
	for i := 0; i < numberOfGames2Play; i += 1 {
		score += playSnakeGame(network)
	}
	return score
}

func invokeGeneticTraining() {
	ga := genetic.NewGeneticAlgorithm(populationSize, networkWeightsCount)
	ga.Elitism = ga_elitism
	ga.TournamentSize = ga_tournamentSize
	ga.CrossoverRate = ga_crossoverRate
	ga.MutationRate = ga_mutationRate
	ga.RandomSeedRate = ga_randomSeedRate
	ga.MutateGaussianDistribution = ga_mutateGaussian
	ga.FitnessThreshold = nil // no stop condition
	ga.FitnessFunc = snakeFitnessFunction
	ga.MaxGenerations = maxGenerations
	ga.Report.SecondsToReport = ga_report_seconds

	best := ga.Run()
	fmt.Println("\nTraining complete!")
	fmt.Printf("Best fitness: %f\n", best.GetFitness())
	network := neural_net.NewFeedForwardNeuralNetwork(topology, false)
	network.SetWeights(best.GetGenes())
	network.SaveToFile(saveNetworkLocation)
}

func continueGenericTraining() {
	network := neural_net.NewFeedForwardNeuralNetworkFromFile(saveNetworkLocation)
	ga := genetic.ContinueGenericAlgorithm(network.GetWeights(), populationSize)
	ga.Elitism = ga_elitism
	ga.TournamentSize = ga_tournamentSize
	ga.CrossoverRate = ga_crossoverRate
	ga.MutationRate = ga_mutationRate
	ga.RandomSeedRate = ga_randomSeedRate
	ga.MutateGaussianDistribution = ga_mutateGaussian
	ga.FitnessThreshold = nil // no stop condition
	ga.FitnessFunc = snakeFitnessFunction
	ga.MaxGenerations = maxGenerations
	ga.Report.SecondsToReport = ga_report_seconds

	best := ga.Run()
	fmt.Println("\nTraining complete!")
	fmt.Printf("Best fitness: %f\n", best.GetFitness())
	network = neural_net.NewFeedForwardNeuralNetwork(topology, false)
	network.SetWeights(best.GetGenes())
	network.SaveToFile(saveNetworkLocation)
}

func show_usage() {
	fmt.Println("Bad argument. Please use 'train', 'train-continue' or 'demo' argument")
}

func play_DemoGame() {
	network := neural_net.NewFeedForwardNeuralNetworkFromFile(saveNetworkLocation)

	cimport.Create_game(game.Size)
	cimport.X_create_window()

	cimport.UpdateGameData(game)
	time.Sleep(1 * time.Second)
	cimport.X_draw_objects()
	fmt.Println("Game starts in 10 seconds... prepare video recorder! ;)")
	// time.Sleep(10 * time.Second)
	fmt.Println("Game starts in 3 seconds... prepare video recorder! ;)")
	time.Sleep(0 * time.Second)

	for !game.GameOver {
		action := network.PredictMaxIndex(game.GetStateForNeuralNetwork())
		switch action {
		case 0:
			game.TurnLeft()
		case 1:
			game.TurnRight()
		case 2: // Continue in the same direction
		}
		game.NextTick() // Make the move!

		cimport.UpdateGameData(game)
		cimport.X_draw_objects()
		fmt.Printf("Direction: %s Reward: %f\n", game.GetDirectionAsString(), game.Reward)
		time.Sleep(100 * time.Millisecond)
	}

	fmt.Println("Game over")
	time.Sleep(1 * time.Second)
	cimport.Destroy_game()
}

func main() {
	initializeGame()

	args := os.Args[1:] // Skip the first argument

	if len(args) != 1 {
		show_usage()
		return
	}

	switch args[0] {
	case "train":
		invokeGeneticTraining()
	case "train-continue":
		continueGenericTraining()
	case "demo":
		play_DemoGame()
	default:
		show_usage()
	}
}

// Plays a snake game AI guided and return total score
func playSnakeGame(network *neural_net.FeedForwardNeuralNetwork) float64 {
	score := 0.0

	game.Reset() // << Start new game

	for !game.GameOver {
		state := game.GetStateForNeuralNetwork()
		action := network.PredictMaxIndex(state)

		switch action {
		case 0:
			game.TurnLeft()
		case 1:
			game.TurnRight()
		case 2: // Continue in the same direction
		}
		game.NextTick() // Make the move!

		if !game.GameOver && game.Moves_since_apple >= game_max_moves_without_apple {
			score += score_move_in_circles // Punish for moving in circes without eating an apple!
			game.GameOver = true           // End this pain
		} else {
			score += game.Reward
		}
	}

	return score
}
