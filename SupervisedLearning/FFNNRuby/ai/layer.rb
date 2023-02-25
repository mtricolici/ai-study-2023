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

    def calculate_errors(deltas)
      @neurons.each_with_index.map do |neuron, i|
        neuron.calculate_error(deltas)
      end
    end

    def update_weights(learning_rate, deltas)
      # update weights and biases of each neuron in the layer
      @neurons.each_with_index do |neuron, i|
        neuron.update_weights(learning_rate, deltas[i])
      end
    end
end