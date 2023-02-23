#!/usr/bin/env ruby

require 'pp'

Dir[File.dirname(__FILE__) + '/ai/*.rb'].each {|file| require file }


n = Neuron.new(num_inputs: 3)
o = n.activate([0.1, 0.2, 0.3])
pp(o)

l = Layer.new(num_neurons: 3, num_inputs: 5)
o = l.activate([1,2,3,4,5])
pp(o)


puts "end"