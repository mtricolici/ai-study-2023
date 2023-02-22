package main

import (
	"MyFeedforwardNeuralNetwork/neuralnetwork"
	"MyFeedforwardNeuralNetwork/utils"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
)

const (
	hiddenLayersCount  = 1
	hiddenLayerNeurons = 200
	outputNeuronsCount = 2 // 1 for cat, 2nd for dog
	trainingIterations = 100
	learningRate       = 0.1
)

func main() {
	home, _ := os.UserHomeDir()
	trainImagesDirectory := home + "/ai-datasets/dogs-cats/train-normal"

	images, labels, err := LoadImages(trainImagesDirectory)

	if err != nil {
		log.Fatal(err)
	}

	// Create a neural network with 100 input neurons, 10 hidden neurons, and 4 output neurons
	nn := neuralnetwork.NewNeuralNet(utils.Size*utils.Size, outputNeuronsCount, hiddenLayersCount, hiddenLayerNeurons)
	nn.RandomizeWeights()

	// Train the neural network using the images and labels
	nn.Train(images, labels, learningRate, trainingIterations)
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Training complete!")
	//TODO: load test images and test the NN with new examples
}

// LoadImages reads JPEG images from the specified directory and returns a slice of
// image pixel arrays and a slice of corresponding labels.
func LoadImages(dir string) ([][]float64, [][]float64, error) {
	var images [][]float64
	var labels [][]float64

	// Walk through the files in the directory
	err := filepath.Walk(dir, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			// Load the image file
			image, err := utils.ReadJpegImage(path)
			if err != nil {
				return err
			}
			// Add the image and its label to the slices
			images = append(images, image)
			labels = append(labels, LabelFromFilename(info.Name()))
		}
		return nil
	})
	if err != nil {
		return nil, nil, err
	}

	return images, labels, nil
}

func LabelFromFilename(filename string) []float64 {
	switch {
	case strings.HasPrefix(filename, "cat"):
		return []float64{1, 0}
	case strings.HasPrefix(filename, "dog"):
		return []float64{0, 1}
	default:
		return []float64{0, 0}
	}
}
