--[[ 
   We download CIFAR-10 dataset from http://www.cs.toronto.edu/~kriz/cifar.html and 
   converts it to Torch tables.
   
   It will create two files: cifar10-train.t7, cifar10-test.t7 Each of them is a table of the form:

      th> c10 = torch.load('cifar10-train.t7')
      th> print(c10)
      {
              data : ByteTensor - size: 50000x3x32x32
              label : ByteTensor - size: 50000
      }

   Refer to https://github.com/soumith/cifar.torch for conversion process.
   
   ----------------------------------
   Also, even not in use here, there is a publicly availabele CIFAR-10 in Tortch 7 format:
   http://torch7.s3-website-us-east-1.amazonaws.com/data/cifar10.t7.tgz
   
--]]

require 'torch'
require 'paths'

cifar10={}

cifar10.path_remote_file = 'http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'    --some 163M
cifar10.path_project_root_dir = '.'
cifar10.path_dataset_dir = paths.concat(paths.dirname(cifar10.path_project_root_dir), 'datasets/cifar10_32x32_t7') --local files
cifar10.path_trainset_file = paths.concat(cifar10.path_dataset_dir, 'cifar10-train.t7')
cifar10.path_testset_file = paths.concat(cifar10.path_dataset_dir, 'cifar10-test.t7')
cifar10.path_tar_file = paths.concat(paths.dirname(cifar10.path_dataset_dir), paths.basename(cifar10.path_remote_file))
cifar10.keep_tar_file_flag = true

--  adapted from https://github.com/soumith/cifar.torch
local function convertCifar10BinToTorchTensor(inputFnames, outputFname)
   local nSamples = 0
   for i=1,#inputFnames do
      local inputFname = inputFnames[i]
      local m=torch.DiskFile(inputFname, 'r'):binary()
      m:seekEnd()
      local length = m:position() - 1
      local nSamplesF = length / 3073 -- 1 label byte, 3072 pixel bytes
      assert(nSamplesF == math.floor(nSamplesF), 'expecting numSamples to be an exact integer')
      nSamples = nSamples + nSamplesF
      m:close()
   end

   local label = torch.ByteTensor(nSamples)
   local data = torch.ByteTensor(nSamples, 3, 32, 32)

   local index = 1
   for i=1,#inputFnames do
      local inputFname = inputFnames[i]
      local m=torch.DiskFile(inputFname, 'r'):binary()
      m:seekEnd()
      local length = m:position() - 1
      local nSamplesF = length / 3073 -- 1 label byte, 3072 pixel bytes
      m:seek(1)
      for j=1,nSamplesF do
         label[index] = m:readByte()
         local store = m:readByte(3072)
         data[index]:copy(torch.ByteTensor(store))
         index = index + 1
      end
      m:close()
   end

   local out = {}
   out.data = data
   out.label = label
   print(out)
   torch.save(outputFname, out)
end


-- Ensure CIFAR-10 dataset in Torch7 binary format exists on local disk
-- If required, the dataset will be downloaded
function cifar10.download()
   local remote_file = cifar10.path_remote_file
   local tar_file = cifar10.path_tar_file

   if not paths.filep(cifar10.path_trainset_file) or not paths.filep(cifar10.path_testset_file) then
      print('No local dataset in Torch7 format.')
      
      -- get original dataset to the local disk if missing   
      if not paths.filep(tar_file) then
         print('can\'t find local ' .. tar_file .. '. Going to download original ' .. remote_file)  
         os.execute('wget ' .. remote_file .. ' -P ' .. paths.dirname(tar_file)) -- download to the given directory
      else
         print('local CIFAR-10 tar found: ' .. tar_file)   
      end
      
      print('untar the dataset')
      os.execute('tar xvf ' .. tar_file .. ' -C ' .. paths.dirname(tar_file))  -- untar the dataset
      
      if not cifar10.keep_tar_file_flag then
         print('deleting ' .. tar_file)
         os.execute('rm -f ' .. tar_file)
      end
      
      -- CIFAR-10 bin files are ready to be converted to Torch tensors
      data_dir = paths.dirname(cifar10.path_dataset_dir)
      os.execute('mkdir -p ' .. cifar10.path_dataset_dir)
      convertCifar10BinToTorchTensor({ paths.concat(data_dir, 'cifar-10-batches-bin/data_batch_1.bin'),
                                       paths.concat(data_dir, 'cifar-10-batches-bin/data_batch_2.bin'),
                                       paths.concat(data_dir, 'cifar-10-batches-bin/data_batch_3.bin'),
                                       paths.concat(data_dir, 'cifar-10-batches-bin/data_batch_4.bin'),
                                       paths.concat(data_dir, 'cifar-10-batches-bin/data_batch_5.bin')},
                                       cifar10.path_trainset_file)

      convertCifar10BinToTorchTensor({paths.concat(data_dir, 'cifar-10-batches-bin/test_batch.bin')}, cifar10.path_testset_file)
   
      
   else
      print('CIFAR-10 dataset in Torch7 binary format exists on the local drive')     
   end
   
   
end

