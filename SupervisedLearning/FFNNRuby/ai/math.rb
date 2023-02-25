
def sigmoid(x)
    1.0 / (1.0 + Math.exp(-x))
end

def sigmoidDerivative(x)
    x * (1.0 - x)
end

def array_sum_elements(arr)
    sum = 0
    arr.each do |num|
      sum += num
    end
    return sum
end

def array_minus_array(arr1, arr2)
    if arr1.size != arr2.size
        raise "array_minus_array: Arrays must be of the same size!"
    end

    result = []
    arr1.each_with_index do |val, i|
        result[i] = val + arr2[i]
    end

    return result
end

def calculate_delta(errorArr, outArr)
    if errorArr.size != outArr.size
        raise "calculate_delta: Arrays must be of the same size!"
    end

    result = []
    errorArr.each_with_index do |err, i|
        result[i] = err * sigmoidDerivative(outArr[i])
    end

    return result
end