package utils

import (
	"fmt"
	"image/color"
	"image/jpeg"
	"math"
	"os"
	"reflect"
)

const (
	size = 100
)

func ReadJpegImage(filename string) ([]float64, error) {
	// Open the file
	file, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// Decode the JPEG image
	img, err := jpeg.Decode(file)
	if err != nil {
		return nil, err
	}

	// Check that the image is 100x100
	bounds := img.Bounds()
	if bounds.Dx() != size || bounds.Dy() != size {
		return nil, fmt.Errorf("image size must be %dx%d", size, size)
	}

	// Create an array to hold the pixel values
	pixels := make([]float64, size*size)

	// Convert the image to grayscale and normalize the pixel values
	for y := 0; y < bounds.Dy(); y++ {
		for x := 0; x < bounds.Dx(); x++ {
			r, _, _, _ := img.At(x, y).RGBA()
			gray := float64(r >> 8)
			maxVal := math.Pow(2, float64(bitsPerChannel(img.ColorModel()))) - 1
			pixels[y*bounds.Dx()+x] = 2*(gray/maxVal) - 1
		}
	}

	return pixels, nil
}

// bitsPerChannel returns the number of bits per channel for the given color model
func bitsPerChannel(cm color.Model) uint {
	switch v := reflect.ValueOf(cm); v.Type() {
	case reflect.TypeOf((*color.RGBA)(nil)).Elem():
		return 8
	case reflect.TypeOf((*color.RGBA64)(nil)).Elem():
		return 16
	case reflect.TypeOf((*color.Gray)(nil)).Elem():
		return 8
	case reflect.TypeOf((*color.Gray16)(nil)).Elem():
		return 16
	default:
		return 8
	}
}
