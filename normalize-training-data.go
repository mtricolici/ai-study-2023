package main

import (
	"fmt"
	"image"
	"image/jpeg"
	"io/ioutil"
	"os"
	"path/filepath"

	"gocv.io/x/gocv"
)

const (
	inputDir  = "~/ai-datasets/dogs-cats/train/"        // Directory with original training images (different sizes)
	outputDir = "~/ai-datasets/dogs-cats/train-normal/" // Target dir to save normalized images
	size      = 100                                     // The size (in pixels) to which the images will be resized
)

func main() {
	// Create the output directory if it does not exist
	if _, err := os.Stat(outputDir); os.IsNotExist(err) {
		os.Mkdir(outputDir, 0755)
	}

	// Get the list of JPEG files in the input directory
	files, err := ioutil.ReadDir(inputDir)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Iterate over the JPEG files in the input directory
	for _, file := range files {
		if filepath.Ext(file.Name()) == ".jpg" {
			fmt.Printf("Processing file '%s'", file.Name())
			// Read the image from the file
			img := gocv.IMRead(filepath.Join(inputDir, file.Name()), gocv.IMReadColor)

			// Resize the image to the desired size
			resized := gocv.NewMat()
			gocv.Resize(img, &resized, image.Point{X: size, Y: size}, 0, 0, gocv.InterpolationDefault)
			resized_image, err := resized.ToImage()
			if err != nil {
				fmt.Printf("Erorr: %s", err)
				continue
			}

			// Save the resized image to the output directory
			outPath := filepath.Join(outputDir, file.Name())
			out, err := os.Create(outPath)
			if err != nil {
				fmt.Println(err)
				continue
			}
			defer out.Close()
			jpeg.Encode(out, resized_image, nil)
			break
		}
	}
}
