package utils

import (
	"io"
	"os"
	"path/filepath"
	"strings"
)

func CopyFile(src, dst string) error {
	// Open the source file for reading
	srcFile, err := os.Open(src)
	if err != nil {
		return err
	}
	defer srcFile.Close()

	// Create the destination file for writing
	dstFile, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer dstFile.Close()

	// Copy the contents of the source file to the destination file
	_, err = io.Copy(dstFile, srcFile)
	if err != nil {
		return err
	}

	return nil
}

func RemoveJpgFiles(dirPath string) error {
	// Open the directory
	dir, err := os.Open(dirPath)
	if err != nil {
		return err
	}
	defer dir.Close()

	// Get a list of all files in the directory
	files, err := dir.Readdir(-1)
	if err != nil {
		return err
	}

	// Remove all JPG files in the directory
	for _, file := range files {
		if file.Mode().IsRegular() && strings.HasSuffix(strings.ToLower(file.Name()), ".jpg") {
			err = os.Remove(filepath.Join(dirPath, file.Name()))
			if err != nil {
				return err
			}
		}
	}

	return nil
}
