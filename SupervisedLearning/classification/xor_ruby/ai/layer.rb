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

    def calculate_errors(errors, deltas)
      @neurons.each_with_index do |neuron, i|
        neuron.calculate_error(errors, deltas[i])
      end
    end

    def update_weights(inputs, deltas, learning_rate)
      @neurons.each_with_index do |neuron, i|
        neuron.update_weights(inputs, deltas[i], learning_rate)
      end

    end
end