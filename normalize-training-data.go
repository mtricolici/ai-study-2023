package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

const (
	inputDir  = "~/ai-datasets/dogs-cats/train/"        // Directory with original training images (different sizes)
	outputDir = "~/ai-datasets/dogs-cats/train-normal/" // Target dir to save normalized images
	size      = 100                                     // The size (in pixels) to which the images will be resized
)

func normalizePath(path string) string {
	home, _ := os.UserHomeDir()
	absolutePath, _ := filepath.Abs(strings.Replace(path, "~", home, -1))
	return absolutePath
}

func resizeImage(source string, target string) {
	fmt.Printf("Resizing '%s' to '%s'\n", source, target)
	size := fmt.Sprintf("%dx%d", size, size)

	args := []string{source, "-gravity", "center", "-background", "red", "-resize", size, "-extent", size, target}
	cmd := exec.Command("convert", args[:]...)
	_, err := cmd.Output()
	if err != nil {
		fmt.Println("Error:", err)
	}
}

func main() {
	srcDir := normalizePath(inputDir)
	dstDir := normalizePath(outputDir)

	// Create the output directory if it does not exist
	if _, err := os.Stat(dstDir); os.IsNotExist(err) {
		os.Mkdir(dstDir, 0755)
	}

	// Get the list of JPEG files in the input directory
	files, err := os.ReadDir(srcDir)
	if err != nil {
		fmt.Println(err)
		return
	}

	// Iterate over the JPEG files in the input directory
	for _, file := range files {
		if filepath.Ext(file.Name()) == ".jpg" {

			srcFile := filepath.Join(srcDir, file.Name())
			dstFile := filepath.Join(dstDir, file.Name())
			resizeImage(srcFile, dstFile)
		}
	}
}
