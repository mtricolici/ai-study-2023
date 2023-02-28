package polynomial

import (
	"errors"
	"fmt"
	"math"
	"regression_sample1/utils"
	"time"

	"gonum.org/v1/gonum/mat"
)

type PolynomialRegression struct {
	degree int       // degree of the polynomial
	coeffs []float64 // coefficients of the polynomial
}

func NewPolynomialRegression(degree int) *PolynomialRegression {
	return &PolynomialRegression{degree, make([]float64, degree+1)}
}

// Fit fits the polynomial regression model to the data
func (pr *PolynomialRegression) Fit(x, y []float64) error {
	n := len(x)
	if n != len(y) {
		return errors.New("x and y must have the same length")
	}

	// create the matrix of basis functions
	var basis [][]float64
	for i := 0; i < n; i++ {
		row := make([]float64, pr.degree+1)
		for j := 0; j <= pr.degree; j++ {
			row[j] = math.Pow(x[i], float64(j))
		}
		basis = append(basis, row)
	}

	// compute the coefficients of the polynomial
	A := mat.NewDense(n, pr.degree+1, nil)
	A.Apply(func(i, j int, v float64) float64 {
		return basis[i][j]
	}, A)
	yVec := mat.NewVecDense(n, y)
	var xVec mat.VecDense
	err := xVec.SolveVec(A, yVec)
	if err != nil {
		return err
	}

	pr.coeffs = xVec.RawVector().Data

	return nil
}

// Predict predicts the y value for a given x value
func (pr *PolynomialRegression) Predict(x float64) float64 {
	var y float64
	for i := 0; i <= pr.degree; i++ {
		y += pr.coeffs[i] * math.Pow(x, float64(i))
	}
	return y
}

func (p *PolynomialRegression) TrainAndTest(inputDates, inputValues []float64) {

	fmt.Println("Polynomial regression in progress ..")

	normal_x := normalizeDates(inputDates)
	normal_y := normalizeValues(inputValues)

	err := p.Fit(normal_x, normal_y)
	if err != nil {
		panic("polynomial regression Fit failure - " + err.Error())
	}

	predictedValuesCount := (len(inputValues) - 1) / 2

	lastDate := time.Unix(int64(inputDates[len(inputDates)-1]), 0)

	var predictedValues []float64

	for i := 0; i < predictedValuesCount; i++ {
		nextDate := lastDate.AddDate(0, 0, 1) // Add 1 day
		lastDate = nextDate                   //save for next iteration

		nextInput := float64(nextDate.Unix())
		nextValue := denormalizeValue(p.Predict(normalizeDate(nextInput)))

		//fmt.Printf("Predicted next value: %f\n", nextValue)

		predictedValues = append(predictedValues, nextValue)
	}

	// Save results to a PNG file for better visibility!
	utils.DrawRegressionToPngFile(
		inputValues, predictedValues, "Polynomial", "polynomial.png")
}
