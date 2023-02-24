
def sigmoid(x)
    1.0 / (1.0 + Math.exp(-x))
end

def sigmoidDerivative(x)
    x * (1 - x)
end
