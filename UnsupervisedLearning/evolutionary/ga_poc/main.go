package main

import (
	"fmt"
	"ga_poc/genetic"
)

func main() {
	fmt.Println("Hello world!")

	ga := genetic.NewGeneticAlgorithm("hello world!", 100)
	ga.Run(20000)
}
