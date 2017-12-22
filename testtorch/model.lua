require 'torch'   -- torch
require 'image'   -- for image transforms
require 'nn'      -- provides all sorts of trainable modules/layers

model={}
function model.constructModel()
   print '==> construct model'

   print '==> define parameters'

   -- 10-class problem
   noutputs = 10

   -- input dimensions
   nfeats = 3
   width = 32
   height = 32
   ninputs = nfeats*width*height

   -- number of hidden units (for MLP only):
   nhiddens = ninputs / 2

   -- hidden units, filter sizes (for ConvNet only):
   nstates = {64,64,128}
   filtsize = 5
   poolsize = 2
--   normkernel = image.gaussian1D(7)

   -- a typical modern convolution network (conv+relu+pool)
   net = nn.Sequential()

   -- stage 1 : filter bank -> squashing -> L2 pooling -> normalization
   net:add(nn.SpatialConvolutionMM(nfeats, nstates[1], filtsize, filtsize))
   net:add(nn.ReLU())
   net:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

   -- stage 2 : filter bank -> squashing -> L2 pooling -> normalization
   net:add(nn.SpatialConvolutionMM(nstates[1], nstates[2], filtsize, filtsize))
   net:add(nn.ReLU())
   net:add(nn.SpatialMaxPooling(poolsize,poolsize,poolsize,poolsize))

   -- stage 3 : standard 2-layer neural network
   net:add(nn.View(nstates[2]*filtsize*filtsize))
   net:add(nn.Dropout(0.5))
   net:add(nn.Linear(nstates[2]*filtsize*filtsize, nstates[3]))
   net:add(nn.ReLU())
   net:add(nn.Linear(nstates[3], noutputs))
  
   return net
end
