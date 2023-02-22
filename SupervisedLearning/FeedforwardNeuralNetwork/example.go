package main

import (
	"MyFeedforwardNeuralNetwork/utils"
	"fmt"
	"os"
)

func main() {
	fmt.Println("Hi world")

	home, _ := os.UserHomeDir()
	testImage := home + "/ai-datasets/dogs-cats/train-normal/cat.0.jpg"

	image, err := utils.ReadJpegImage(testImage)
	if err != nil {
		panic(err)
	} else {
		fmt.Println(image)
	}
}
