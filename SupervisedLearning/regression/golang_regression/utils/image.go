package utils

import (
	"fmt"
	"path/filepath"
)

func DrawRegressionToPngFile(inputs, predicted []float64, outputFileName string) {

	saveFileName := filepath.Join(Get_executable_folder(), outputFileName)
	fmt.Printf("Drawing results to '%s'\n", saveFileName)
	//TODO: implement drawing graph to a png file
}
