Overview
--------
This code uses matconvnet for training. This code works with
matconvnet-1.0-beta20 release and cuda 7.5. I have not tested it with later
releases of matconvnet.


MatConvNet Compilation
----------------------
To compile MatConvNet make sure that the path to the cudatool kit is correct.
By default the compilation will use matlabs version which may be lagging behind.
To fix this explicitly specify the path to the toolkit during compilation

 vl_compilenn('enableGpu', true, 'cudaMethod', 'nvcc', ...
  'cudaRoot', '/usr/local/cuda-7.5', ...
  'enableCudnn', true, 'cudnnRoot', 'local/') ;

Note that these may need to be adapted for the more recent release of
matconvnet.

Starting Matlab
---------------

Navigate to the root dir of the code and run the following

LD_LIBRARY_PATH=/usr/local/cuda-7.5/lib64:/home/saurabh/Work/cv-libs/matconvnet-1.0-beta20/local/cudnn-7-v4/lib64 matlab -nodesktop

This assumes that you have cuda-7.5 installed along with cudnn-7-v4. You can get
these from nvidia website.


Setting up the startup
----------------------

You may need to clean up the startup.m to remove addition of dependencies that
are not required. 

Setting up a new Dataset
------------------------

Refer to datasets/prepareCarDoorHandleDataset.m for an examples of getting the 
data ready.


Starting the training
---------------------

To start training a model run
centroid_chain/warpTrainCNNLSPCentroidChainGridPredSharedRevFastExp3.m

This is done by running at the matlab prompt
>> warpTrainCNNLSPCentroidChainGridPredSharedRevFastExp3(CONFIG)

Note that CONFIG global variable is created by startup.m and it contains various
configuration flags.


If everything is setup right then the training should start.

