package ai

type ActivationType int

const (
	ActivationLinear = iota
	ActivationSigmoid
	ActivationRelu
	ActivationTanh
	ActivationElu
	ActivationSoftMax
)
