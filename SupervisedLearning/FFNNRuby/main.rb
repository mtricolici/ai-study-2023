#!/usr/bin/env ruby

require 'pp'

Dir[File.dirname(__FILE__) + '/ai/*.rb'].each {|file| require file }


n = Neuron.new(3)
pp(n.activate([0.1, 0.2, 0.3]))


puts "end"