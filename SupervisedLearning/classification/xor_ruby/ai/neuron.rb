class Neuron
  attr_accessor :weights

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

    sigmoid(sum)
  end

  def calculate_error(errors, delta)
    @weights.each_with_index do |w, i|
      errors[i] += delta * w
    end
  end

  # Update the weights of the neuron during backpropagation
  def update_weights(inputs, delta, learning_rate)
    @weights = @weights.each_with_index.map do |weight, i|
      weight + learning_rate * delta * inputs[i]
    end

    @bias = @bias + learning_rate * delta
  end

end