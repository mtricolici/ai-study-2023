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

    # Backpropagation training
    def train(inputs:, labels:, num_epochs:, learning_rate:)
        puts "Backpropagation training started"

        num_epochs.times do |epoch|
            error = train_epoch(
                inputs: inputs, labels: labels, learning_rate: learning_rate)
            
            puts "Epoch #{epoch + 1}, Error: #{error}"
        end
    end

    private

    def train_epoch(inputs:, labels:, learning_rate:)
        sum_errors = 0.0

        inputs.each_with_index do |inp, i|
            sum_errors += train_one_input(
                input: inp, label: labels[i], learning_rate: learning_rate)
        end

        return sum_errors / inputs.length()
    end

    def train_one_input(input:, label:, learning_rate:)
        # compute specific input
        layer1_output = @layer1.activate(input)
        layer2_output = @layer2.activate(layer1_output)

        # Calculate error: difference between expected-output (i.e. label) and actual-output (i.e. layer2_output)
        networkErrors = array_minus_array(label, layer2_output)

        # Backward pass
        layer2_delta = calculate_delta(networkErrors, layer2_output) # size=1
        #layer1_errors = @layer1.calculate_errors(layer2_delta)

        #layer1_delta = calculate_delta(layer1_errors, layer1_output) # size=5
        layer1_delta = @layer1.calculate_errors(layer2_delta)

        # Update weights and biases
        @layer2.update_weights(learning_rate, layer2_delta)
        @layer1.update_weights(learning_rate, layer1_delta)

        return array_sum_elements(networkErrors)
    end
end