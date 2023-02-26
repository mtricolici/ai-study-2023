
def sigmoid(x)
    1.0 / (1.0 + Math.exp(-x))
end

def sigmoidDerivative(x)
    x * (1.0 - x)
end

def calculate_error_sum(arr)
    sum = 0
    arr.each do |num|
      sum += num.abs
    end
    return sum
end

def array_minus_array(arr1, arr2)
    if arr1.size != arr2.size
        raise "array_minus_array: Arrays must be of the same size!"
    end

    result = []
    arr1.each_with_index do |val, i|
        result[i] = val - arr2[i]
    end

    return result
end

def calculate_delta(errorArr, outArr)
    if errorArr.length != outArr.length
        raise "calculate_delta: Arrays must be of the same size!"
    end

    result = Array.new(errorArr.length) { 0.0 }
    errorArr.each_with_index do |err, i|
        result[i] = err * sigmoidDerivative(outArr[i])
    end

    return result
end