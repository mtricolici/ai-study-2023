class Network
    attr_accessor :layer1, :layer2

    def initialize(num_inputs:, num_hidden:, num_outputs:)
        @layer1 = Layer.new(num_inputs: num_inputs, num_neurons: num_hidden)
        @layer2 = Layer.new(num_inputs: num_hidden, num_neurons: num_outputs)
    end

    def predict(inputs)
        outputs = @layer1.activate(inputs)
        @layer2.activate(outputs)
    end
end