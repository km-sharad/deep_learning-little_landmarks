[unused, hostname] = system('hostname');

dataset_root = [];
%if ~isempty(strfind(hostname, 'shunya'))
  fprintf('Local run\n');
  %outRoot = '/media/ssd2/outputs/';
  outRoot = ' /home/sharad/CS503-Thesis/Code/CDHD/data/output/';
  %procRoot = '/home/saurabh/Work/outputs/';
  procRoot = '/home/sharad/CS503-Thesis/Code/CDHD/data/procroot/';
  %dataset_root = '/media/ssd1/datasets/';
  dataset_root = '/home/sharad/CS503-Thesis/Code/CDHD/';
%end

if isempty(dataset_root)
  dataset_root = [procRoot '../datasets/'];
end

CONFIG.procRoot = procRoot;
CONFIG.procDir = [outRoot 'little_landmarks/'];
%CONFIG.libDir = [procRoot '../cv-libs/'];
CONFIG.libDir = ['/home/sharad/CS503-Thesis/Code/CDHD/cv-libs/'];
%CONFIG.flicDataDir = [procRoot '../datasets/FLIC/'];
%CONFIG.indoor67DataDir = [procRoot '../datasets/Indoor67/'];
%CONFIG.fashionPoseDataDir = [procRoot '../datasets/fashion_pose/'];
%CONFIG.lspDataDir = [dataset_root 'LSP/'];
%CONFIG.lspEvalRoot = [CONFIG.libDir 'eval_LSP_leonid/'];
%CONFIG.cub2002011DataDir = [dataset_root 'CUB_200_2011/'];
%CONFIG.stanfordFGCarDataDir = [procRoot '../datasets/StanfordFGCar/'];
CONFIG.stanfordFGCarDataDir = '/home/sharad/CS503-Thesis/car_dataset/';
%CONFIG.lightSwitchDataDir = [procRoot '../datasets/lightswitch/'];
%CONFIG.mylibsPath = '../../mylibs/';
%CONFIG.dataDir = '../data/';

% Add paths
%addpath(genpath([CONFIG.libDir 'descriptors/']));
%addpath(genpath([CONFIG.libDir 'eval_LSP_leonid/']));

% Add paths to piotr's toolbox, used for padding etc.
% Download from https://pdollar.github.io/toolbox/
addpath(genpath([CONFIG.libDir 'toolbox-master/']));

% Add matconvnet
addpath([CONFIG.libDir 'matconvnet-1.0-beta20/matlab']);
addpath([CONFIG.libDir 'matconvnet-1.0-beta20/utils']);
addpath(genpath([CONFIG.libDir 'matconvnet-1.0-beta20/examples']));
vl_setupnn;

% Add my libs and current code to path.
%addpath(genpath(CONFIG.mylibsPath));
addpath(genpath('.'));


% Add the path to the birds eval code
%addpath(genpath('../../cnn_birds/code/'));


