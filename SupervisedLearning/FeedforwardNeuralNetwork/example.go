package main

import (
	"MyFeedforwardNeuralNetwork/ai"
	"MyFeedforwardNeuralNetwork/utils"
	"fmt"
	"log"
	"os"
	"path/filepath"
)

const (
	inputNeurons       = utils.Size * utils.Size
	hiddenLayerNeurons = 300
	outputNeuronsCount = 2 // 1 for cat, 2nd for dog
	trainingIterations = 10
	learningRate       = 0.1
	trainSamples       = 1000 // nr of dogs and cats for training
	testSamples        = 100
)

func main() {

	home, _ := os.UserHomeDir()
	trainImagesDirectory := home + "/ai-datasets/dogs-cats/train-normal"
	testImagesDirectory := home + "/ai-datasets/dogs-cats/test1-normal"
	outputDirectory := home + "/temp"

	log.Println("Loading training data into memory ...")

	// Load random X images of cats and X images of dogs
	catImages, catLabels := utils.LoadImagesData(trainImagesDirectory, "^cat.*\\.jp.?g$", trainSamples)
	dogImages, dogLabels := utils.LoadImagesData(trainImagesDirectory, "^dog.*\\.jp.?g$", trainSamples)

	trainingImages := append(catImages, dogImages...)
	trainingLabels := append(catLabels, dogLabels...)

	log.Println("Creating a Neural network")

	// Create a neural network
	nn := ai.NewFeedForwardNeuralNetwork(inputNeurons, hiddenLayerNeurons, outputNeuronsCount)

	log.Println("Training started ...")

	nn.Train(trainingImages, trainingLabels, trainingIterations, learningRate)

	log.Println("Training complete!")

	log.Println("Testing network")
	testNetwork(testImagesDirectory, outputDirectory, nn)

}

func testNetwork(testImagesDirectory, outputDirectory string, nn *ai.FeedForwardNeuralNetwork) {

	// Cleanup previous invokation output test result images
	utils.RemoveJpgFiles(outputDirectory)

	testImages := utils.LoadRandomImageFiles(testImagesDirectory, ".*\\.jp.?g$", testSamples)
	for idx, file := range testImages {
		image, err := utils.ReadJpegImage(file)
		if err != nil {
			panic("Could not read image")
		}

		res := nn.Predict(image)
		// 1st neuron in output - cat
		// 2nd neuron in output - dog
		petName := "cat"
		if res[1] > res[0] {
			petName = "dog"
		}

		outFileName := fmt.Sprintf("%s%d.jpg", petName, idx)
		outFilePath := filepath.Join(outputDirectory, outFileName)

		utils.CopyFile(file, outFilePath)
	}

	log.Printf("Tested images are saved in %s !", outputDirectory)
}
