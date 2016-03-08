require 'nn'

function createModel(nGPU)

   -- from https://code.google.com/p/cuda-convnet2/source/browse/layers/layers-imagenet-1gpu.cfg
   -- this is AlexNet that was presented in the One Weird Trick paper. http://arxiv.org/abs/1404.5997
   local model = nn.Sequential()
   model:add(nn.SpatialConvolution(3,64,11,11,4,4,2,2))       -- 224 -> 55
   model:add(nn.ReLU(true))
   model:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 55 ->  27
   model:add(nn.SpatialConvolution(64,192,5,5,1,1,2,2))       --  27 -> 27
   model:add(nn.ReLU(true))
   model:add(nn.SpatialMaxPooling(3,3,2,2))                   --  27 ->  13
   model:add(nn.SpatialConvolution(192,384,3,3,1,1,1,1))      --  13 ->  13
   model:add(nn.ReLU(true))
   model:add(nn.SpatialConvolution(384,256,3,3,1,1,1,1))      --  13 ->  13
   model:add(nn.ReLU(true))
   model:add(nn.SpatialConvolution(256,256,3,3,1,1,1,1))      --  13 ->  13
   model:add(nn.ReLU(true))
   model:add(nn.SpatialMaxPooling(3,3,2,2))                   -- 13 -> 6
   model:add(nn.View(256*6*6))

   model:add(nn.Dropout(0.5))
   model:add(nn.Linear(256*6*6, 4096))
   model:add(nn.ReLU())

   --model:add(nn.Dropout(0.5))
   model:add(nn.Linear(4096, 4096))
   model:add(nn.ReLU())

   model:add(nn.Linear(4096, 1000))
   --model:add(nn.LogSoftMax())

   return model
end

net = createModel()
torch.save('model.net', net)
print(net)
