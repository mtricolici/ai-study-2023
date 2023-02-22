package main

import (
	"MyFeedforwardNeuralNetwork/ai"
	"MyFeedforwardNeuralNetwork/utils"
	"log"
	"os"
)

const (
	inputNeurons       = utils.Size * utils.Size
	hiddenLayerNeurons = 300
	outputNeuronsCount = 2 // 1 for cat, 2nd for dog
	trainingIterations = 10
	learningRate       = 0.1
)

func main() {

	home, _ := os.UserHomeDir()
	trainImagesDirectory := home + "/ai-datasets/dogs-cats/train-normal"

	log.Println("Loading training data into memory ...")

	images, labels := utils.LoadImages(trainImagesDirectory)

	log.Println("Creating a Neural network")

	// Create a neural network
	nn := ai.NewFeedForwardNeuralNetwork(inputNeurons, hiddenLayerNeurons, outputNeuronsCount)

	log.Println("Training started ...")

	nn.Train(images, labels, trainingIterations, learningRate)

	log.Println("Training complete!")
	//TODO: load test images and test the NN with new examples
}
