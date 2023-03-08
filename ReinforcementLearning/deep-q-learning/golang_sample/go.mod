module vanilla_deep_qlearn

go 1.20

replace github.com/mtricolici/ai-study-2023/golibs/snake => ../../../golibs/snake

replace github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network => ../../../golibs/feed-forward-neural-network

require (
	github.com/mtricolici/ai-study-2023/golibs/feed-forward-neural-network v0.0.0
	github.com/mtricolici/ai-study-2023/golibs/snake v0.0.0
	golang.org/x/text v0.8.0
)
