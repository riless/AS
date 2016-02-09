require 'torch'
local CharLMMinibatchLoader=require 'CharLMMinibatchLoader'

local v=CharLMMinibatchLoader.create("data.t7","vocab.t7",1,50)
print(v)
print(v.x_batches[1])
print(v.y_batches[1])
