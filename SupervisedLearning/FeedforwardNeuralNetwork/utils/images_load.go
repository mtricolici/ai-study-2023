package utils

import (
	"math/rand"
	"os"
	"path/filepath"
	"regexp"
	"strings"
	"time"
)

func labelFromFilename(filename string) []float64 {
	switch {
	case strings.Contains(filename, "/cat"):
		return []float64{1, 0}
	case strings.Contains(filename, "/dog"):
		return []float64{0, 1}
	default:
		return []float64{0, 0}
	}
}

func LoadRandomImageFiles(path string, pattern string, max_items int) []string {
	fileRegex := regexp.MustCompile(pattern)

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

	// Randomly shuffle the array
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	r.Shuffle(len(files), func(i, j int) { files[i], files[j] = files[j], files[i] })

	var result []string
	count := 0

	for _, file := range files {
		if fileRegex.MatchString(file.Name()) {
			result = append(result, filepath.Join(path, file.Name()))

			count += 1
			if count >= max_items {
				break
			}
		}
	}

	return result
}

// LoadImages reads JPEG images from the specified directory and returns a slice of
// image pixel arrays and a slice of corresponding labels.
func LoadImagesData(path string, pattern string, max_items int) ([][]float64, [][]float64) {

	var images [][]float64
	var labels [][]float64

	files := LoadRandomImageFiles(path, pattern, max_items)

	for _, file := range files {
		// Load the image file
		image, err := ReadJpegImage(file)
		if err != nil {
			panic("Could not read image")
		}

		// Add the image and its label to the slices
		images = append(images, image)
		labels = append(labels, labelFromFilename(file))
	}

	return images, labels
}
