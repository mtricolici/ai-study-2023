class Neuron
  attr_accessor :weights, :bias, :output

  def initialize(num_inputs)
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

    @output = sigmoid(sum)
    @output
  end

  # Update the weights of the neuron during backpropagation
  def update_weights(learning_rate, error)
    @weights = @weights.each_with_index.map do |weight, index|
      weight + learning_rate * error * @output
    end
    @bias = @bias + learning_rate * error
  end
end