#!/usr/bin/env ruby


# Define neural network architecture
input_layer_size = 2
hidden_layer_size = 5
output_layer_size = 1

# Define activation functions
def sigmoid(x)
  1 / (1 + Math.exp(-x))
end

def sigmoid_derivative(x)
  sigmoid(x) * (1 - sigmoid(x))
end

# Initialize weights and biases
weights_ih = Array.new(input_layer_size) { Array.new(hidden_layer_size) { rand } }
biases_h = Array.new(hidden_layer_size) { rand }
weights_ho = Array.new(hidden_layer_size) { Array.new(output_layer_size) { rand } }
bias_o = rand

# Define learning rate and number of epochs
learning_rate = 0.07
epochs = 50000

# XOR Problem Input and Output data
inputs = [[0,0], [0,1], [1,0], [1,1]]
labels = [0, 1, 1, 0]

# Train the neural network
epochs.times do |epoch|
  error = 0
  inputs.each_with_index do |input, i|
    # Forward pass
    hidden_layer = []
    hidden_layer_size.times do |j|
      net = biases_h[j]
      input_layer_size.times do |k|
        net += input[k] * weights_ih[k][j]
      end
      hidden_layer << sigmoid(net)
    end
    output = 0
    hidden_layer_size.times do |j|
      net = bias_o
      net += hidden_layer[j] * weights_ho[j][0]
      output += sigmoid(net)
    end

    # Backward pass
    delta_o = (labels[i] - output) * sigmoid_derivative(output)
    delta_h = []
    hidden_layer_size.times do |j|
      delta_h << delta_o * weights_ho[j][0] * sigmoid_derivative(hidden_layer[j])
    end

    # Update weights and biases
    hidden_layer_size.times do |j|
      weights_ho[j][0] += learning_rate * delta_o * hidden_layer[j]
    end
    bias_o += learning_rate * delta_o
    input_layer_size.times do |k|
      hidden_layer_size.times do |j|
        weights_ih[k][j] += learning_rate * delta_h[j] * input[k]
      end
    end
    hidden_layer_size.times do |j|
      biases_h[j] += learning_rate * delta_h[j]
    end

    error += (labels[i] - output) ** 2
  end
  puts "Epoch: #{epoch+1}, Error: #{error / inputs.size}"
end

# Test the neural network
puts "Testing neural network..."
inputs.each_with_index do |input, i|
  hidden_layer = []
  hidden_layer_size.times do |j|
    net = biases_h[j]
    input_layer_size.times do |k|
      net += input[k] * weights_ih[k][j]
    end
    hidden_layer << sigmoid(net)
  end
  output = 0
  hidden_layer_size.times do |j|
    net = bias_o
    net += hidden_layer[j] * weights_ho[j][0]
    output += sigmoid(net)
  end
  puts "#{input[0]} XOR #{input[1]} = #{output.round}"
end

