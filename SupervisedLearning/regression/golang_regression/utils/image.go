package utils

import (
	"fmt"
	"image/color"
	"path/filepath"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func DrawRegressionToPngFile(inputs, predicted []float64, title, outputFileName string) {
	p := plot.New()

	p.Title.Text = title
	p.X.Label.Text = "X-axis"
	p.Y.Label.Text = "Y-axis"

	inputLine := createPlotXYs(inputs, 0)
	predictedLine := createPlotXYs(predicted, len(inputs))

	// Draw input line with GREEN color
	inputLine.LineStyle.Width = vg.Points(1)
	inputLine.LineStyle.Color = color.RGBA{G: 255, A: 255}
	p.Add(inputLine)

	// Draw predicted line with RED color
	predictedLine.LineStyle.Width = vg.Points(1)
	predictedLine.LineStyle.Color = color.RGBA{R: 255, A: 255}
	p.Add(predictedLine)

	// Set the plot bounds
	p.X.Min = 0
	p.X.Max = float64(len(inputs) + len(predicted) + 2)
	p.Y.Min = 0
	p.Y.Max = calculatePlotMaxY(inputs, predicted)

	saveFileName := filepath.Join(Get_executable_folder(), outputFileName)
	p.Save(600, 400, saveFileName)
	fmt.Printf("%s - saved to %s\n", title, saveFileName)
}

func createPlotXYs(data []float64, deltaX int) *plotter.Line {
	xy := make(plotter.XYs, len(data))
	for i, y := range data {
		xy[i].X = float64(i) + float64(deltaX)
		xy[i].Y = y
	}

	line, err := plotter.NewLine(xy)
	if err != nil {
		panic(err)
	}

	return line
}

func calculatePlotMaxY(inputs, predicted []float64) float64 {
	max := 0.0
	for i := 0; i < len(inputs); i++ {
		if max < inputs[i] {
			max = inputs[i]
		}
	}

	for i := 0; i < len(predicted); i++ {
		if max < predicted[i] {
			max = predicted[i]
		}
	}

	return max + 2.0
}
