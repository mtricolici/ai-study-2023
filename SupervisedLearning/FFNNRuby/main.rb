#!/usr/bin/env ruby

require 'pp'

Dir[File.dirname(__FILE__) + '/ai/*.rb'].each {|file| require file }

nn = Network.new(num_inputs:2, num_hidden:5, num_outputs:1)
o = nn.predict([0.1, 0.2])
pp(o)
