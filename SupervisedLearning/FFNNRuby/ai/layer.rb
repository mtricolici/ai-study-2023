class Layer
    attr_accessor :neurons

    def initialize(num_neurons:, num_inputs:)
      @neurons = Array.new(num_neurons) {
        Neuron.new(num_inputs)
      }
    end
  
    def activate(inputs)
      @neurons.map { |neuron| neuron.activate(inputs) }
    end

    def update_weights(learning_rate, errors)
      # update weights and biases of each neuron in the layer
      @neurons.each_with_index do |neuron, i|
        neuron.update_weights(learning_rate, errors[i])
      end
    end
end