class Layer
    attr_accessor :neurons, :bias

    def initialize(num_neurons:, num_inputs:)
      @bias = rand(-1.0..1.0) #rand(0.1..1.0) * 0.01
      @neurons = Array.new(num_neurons) {
        Neuron.new(num_inputs)
      }
    end
  
    def activate(inputs)
      @neurons.map { |neuron| neuron.activate(inputs, @bias) }
    end

    def calculate_errors(deltas)
      @neurons.each_with_index.map do |neuron, i|
        neuron.calculate_error(deltas)
      end
    end

    def update_weights(learning_rate, deltas)
      @neurons.each_with_index do |neuron, i|
        neuron.update_weights(learning_rate, deltas[i])
      end

      @bias += learning_rate * array_sum_elements(deltas)
    end
end