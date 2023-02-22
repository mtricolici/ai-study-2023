package utils

import (
	"os"
	"path/filepath"
	"strings"
)

const (
	maxNumberOfSampleForTraining = 10000
)

func labelFromFilename(filename string) []float64 {
	switch {
	case strings.HasPrefix(filename, "cat"):
		return []float64{1, 0}
	case strings.HasPrefix(filename, "dog"):
		return []float64{0, 1}
	default:
		return []float64{0, 0}
	}
}

func isJpegFile(name string) bool {
	return strings.HasSuffix(name, ".jpg") || strings.HasSuffix(name, ".jpeg")
}

// LoadImages reads JPEG images from the specified directory and returns a slice of
// image pixel arrays and a slice of corresponding labels.
func LoadImages(path string) ([][]float64, [][]float64) {

	// Open the directory
	dir, err := os.Open(path)
	if err != nil {
		panic(err)
	}
	defer dir.Close()

	// Get a list of all files in the directory
	files, err := dir.Readdir(-1)
	if err != nil {
		panic(err)
	}

	var images [][]float64
	var labels [][]float64

	count := 0

	for _, file := range files {
		if isJpegFile(strings.ToLower(file.Name())) {

			// Load the image file
			image, err := ReadJpegImage(filepath.Join(path, file.Name()))
			if err != nil {
				panic("Could not read image")
			}

			// Add the image and its label to the slices
			images = append(images, image)
			labels = append(labels, labelFromFilename(file.Name()))

			count += 1
			if count >= maxNumberOfSampleForTraining {
				break
			}
		}
	}

	return images, labels
}
