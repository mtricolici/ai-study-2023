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

        return sum_errors # / inputs.length()
    end

    def train_one_input(input:, label:, learning_rate:)
        # compute specific input
        layer1_output = @layer1.activate(input)
        layer2_output = @layer2.activate(layer1_output)

        # Calculate error: difference between expected-output (i.e. label) and actual-output (i.e. layer2_output)
        errors = array_minus_array(label, layer2_output)
        sumErrors = array_sum_elements(errors)

        # Backward pass


        return sumErrors
    end
end