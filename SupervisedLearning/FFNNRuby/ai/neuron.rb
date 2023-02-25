class Neuron
  attr_accessor :weights, :output

  def initialize(num_inputs)
    @weights = Array.new(num_inputs) { rand(-1.0..1.0) }
    @num_inputs = num_inputs
  end

  def activate(inputs, bias)
    if @num_inputs != inputs.length then
      abort("Wrong number of inputs")
    end

    sum = bias
    inputs.each_with_index do |input, i|
      sum += input * weights[i]
    end

    @output = sigmoid(sum)
    @output
  end


  def calculate_error(deltas)
    sum = 0.0
    @weights.each do |weight|
      deltas.each do |delta|
        sum += weight * delta
      end
    end

    return sigmoidDerivative(@output) * sum
  end

  # Update the weights of the neuron during backpropagation
  def update_weights(learning_rate, delta)
    @weights = @weights.each_with_index.map do |weight, index|
      weight + learning_rate * delta * @output
    end
  end

end