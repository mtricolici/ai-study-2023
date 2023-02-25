#!/usr/bin/env ruby

require 'pp'

Dir[File.dirname(__FILE__) + '/ai/*.rb'].each {|file| require file }

# Define the XOR problem inputs and labels
inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
labels = [[0], [1], [1], [0]]

nn = Network.new(num_inputs:2, num_hidden:10, num_outputs:1)

nn.train(inputs: inputs, labels: labels, num_epochs: 30, learning_rate: 0.1)

puts "Let's test NN:"
inputs.each_with_index do |inp, i|
    res = nn.predict(inp).first # Prediction
    exp = labels[i] # Expected
    puts("predict(#{inp}) = #{res}. expected: #{exp}")
end
