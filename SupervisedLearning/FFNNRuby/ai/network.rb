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

    def train(inputs:, labels:, num_epochs:, learning_rate:)
        puts "Backpropagation training started"

        num_epochs.times do |epoch|
            avgError = train_iteration(
                inputs: inputs, labels: labels, learning_rate: learning_rate)
            
            puts "Epoch #{epoch + 1}: Average error = #{avgError}"
        end
    end

    private

    def train_iteration(inputs:, labels:, learning_rate:)
        return 0.666
    end

end