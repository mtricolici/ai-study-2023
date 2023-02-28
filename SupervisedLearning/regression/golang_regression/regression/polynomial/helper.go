package polynomial

import "time"

var (
	maxTimestamp = float64(time.Now().Unix())
	maxValue     = 2.0
)

func normalizeDate(dt float64) float64 {
	return dt / maxTimestamp
}

func normalizeDates(arr []float64) []float64 {
	result := make([]float64, len(arr))
	for i, value := range arr {
		result[i] = normalizeDate(value)
	}
	return result
}

func normalizeValues(arr []float64) []float64 {
	result := make([]float64, len(arr))
	for i, value := range arr {
		result[i] = value / maxValue
	}
	return result
}

func denormalizeValue(val float64) float64 {
	return val * maxValue
}
