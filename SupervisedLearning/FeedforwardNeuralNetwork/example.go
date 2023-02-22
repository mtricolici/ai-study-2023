package main

import (
	"MyFeedforwardNeuralNetwork/ai"
	"MyFeedforwardNeuralNetwork/utils"
	"fmt"
	"log"
	"os"
	"path/filepath"
	"strings"
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
	fmt.Println("Loading training data into memory ...")

	images, labels, err := LoadImages(trainImagesDirectory)

	if err != nil {
		log.Fatal(err)
	}

	fmt.Println("Creating a Neural network")

	// Create a neural network
	nn := ai.NewFeedForwardNeuralNetwork(inputNeurons, hiddenLayerNeurons, outputNeuronsCount)
	// result := nn.Predict(sample_imput)

	fmt.Println("Neural network training started ...")

	nn.Train(images, labels, trainingIterations, learningRate)

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
