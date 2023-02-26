package utils

import (
	"encoding/csv"
	"os"
	"path/filepath"
	"runtime"
	"strconv"
	"time"
)

func Get_executable_folder() string {

	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		panic("No caller information")
	}

	utils_go_path := filepath.Dir(filename)

	return filepath.Join(utils_go_path, "..")
}

func Get_dataset_file_path() string {
	root := Get_executable_folder()

	return filepath.Join(root, "../dataset/us-eu.csv")
}

func Read_dataset() ([]float64, []float64) {
	var x []float64
	var y []float64

	rows := csv_dataset_read_all_lines()

	var previousY float64

	for i, row := range rows {
		if i == 0 {
			continue // skip header
		}

		dateStr := row[0]
		date, err := time.Parse("2006-01-02", dateStr) // Assuming date is in format "yyyy-mm-dd"
		if err != nil {
			panic(err)
		}
		xVal := float64(date.Unix())
		yVal, err := strconv.ParseFloat(row[1], 64)
		// Some dates in CVS have no value. using previous value.
		if err != nil {
			yVal = previousY
		} else {
			previousY = yVal
		}
		x = append(x, xVal)
		y = append(y, yVal)
	}

	return x, y
}

func csv_dataset_read_all_lines() [][]string {
	dataset_file := Get_dataset_file_path()

	// Open the CSV file
	f, err := os.Open(dataset_file)
	if err != nil {
		panic(err)
	}
	defer f.Close()

	// Parse the CSV file
	reader := csv.NewReader(f)
	rows, err := reader.ReadAll()
	if err != nil {
		panic(err)
	}

	return rows
}
