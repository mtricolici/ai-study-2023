package main

import (
    "encoding/csv"
    "fmt"
    "os"
    "strconv"
    "time"
)

func main() {
    // Open the CSV file
    f, err := os.Open("us-eu.csv")
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

    // Convert the data into slices
    var x []float64
    var y []float64
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
        if err != nil {
            yVal = previousY
        } else {
            previousY = yVal
        }
        x = append(x, xVal)
        y = append(y, yVal)
    }

    // Train the linear regression model
    slope, intercept := trainLinearRegression(x, y)

    // Predict the next value
    lastDate := time.Unix(int64(x[len(x)-1]), 0)
    nextDate := lastDate.AddDate(0, 0, 1) // Assuming dates are sequential
    nextX := float64(nextDate.Unix())
    nextY := predictLinearRegression(nextX, slope, intercept)

    fmt.Printf("Predicted next value: %f\n", nextY)
}

func trainLinearRegression(x []float64, y []float64) (slope float64, intercept float64) {
    n := len(x)

    // Calculate the mean of x and y
    xMean := mean(x)
    yMean := mean(y)

    // Calculate the slope and intercept
    var numerator float64
    var denominator float64
    for i := 0; i < n; i++ {
        numerator += (x[i] - xMean) * (y[i] - yMean)
        denominator += (x[i] - xMean) * (x[i] - xMean)
    }
    slope = numerator / denominator
    intercept = yMean - slope*xMean

    return slope, intercept
}

func predictLinearRegression(x float64, slope float64, intercept float64) float64 {
    return slope*x + intercept
}

func mean(values []float64) float64 {
    sum := 0.0
    n := len(values)
    for i := 0; i < n; i++ {
        sum += values[i]
    }
    return sum / float64(n)
}

