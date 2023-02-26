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
	inputNeurons         = utils.Size * utils.Size
	hiddenLayerNeurons   = 50
	outputNeuronsCount   = 2 // 1 for cat, 2nd for dog
	trainingIterations   = 50000
	trainingStopMaxError = 0.01 // If error is less than this the training stops.
	learningRate         = 0.001
	trainSamples         = 2000 // nr of dogs and cats for training
	testSamples          = 100
)

func main() {

	home, _ := os.UserHomeDir()
	trainImagesDirectory := home + "/ai-datasets/dogs-cats/train-normal"
	testImagesDirectory := home + "/ai-datasets/dogs-cats/test1-normal"
	outputDirectory := home + "/temp"

	log.Println("Loading training data into memory ...")

	// Load random X images of cats and X images of dogs
	log.Printf("Loading %d images of cats", trainSamples)
	catImages, catLabels := utils.LoadImagesData(trainImagesDirectory, "^cat.*\\.jp.?g$", trainSamples)
	log.Printf("Loading %d images of dogs", trainSamples)
	dogImages, dogLabels := utils.LoadImagesData(trainImagesDirectory, "^dog.*\\.jp.?g$", trainSamples)

	trainingImages := append(catImages, dogImages...)
	trainingLabels := append(catLabels, dogLabels...)

	// Create a neural network
	nn := ai.NewFeedForwardNeuralNetwork(inputNeurons, hiddenLayerNeurons, outputNeuronsCount)

	log.Println("Training started ...")

	nn.Train(trainingImages, trainingLabels, trainingIterations, learningRate, trainingStopMaxError)

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

		log.Println(res, petName)

		outFileName := fmt.Sprintf("%s%d.jpg", petName, idx)
		outFilePath := filepath.Join(outputDirectory, outFileName)

		utils.CopyFile(file, outFilePath)
	}

	log.Printf("Tested images are saved in %s !", outputDirectory)
}
