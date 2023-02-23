class Neuron
  attr_accessor :weights, :bias

  def initialize(num_inputs:)
    @weights = Array.new(num_inputs) { rand(-1.0..1.0) }
    @bias = rand(0.1..1.0) * 0.01
    @num_inputs = num_inputs
  end

  def activate(inputs)
    if @num_inputs != inputs.length then
      abort("Wrong number of inputs")
    end

    sum = @bias
    inputs.each_with_index do |input, i|
      sum += input * weights[i]
    end
    sigmoid(sum)
  end

  def sigmoid(x)
    1.0 / (1.0 + Math.exp(-x))
  end
end