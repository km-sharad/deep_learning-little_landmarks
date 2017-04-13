function warpTrainCNNCDHDCentroidChainGridPredSharedRevFastExp3(CONFIG)
% This experiment trains a model that passes only one feature channel (the RBF
% at the offset) and is trained with Adam.
%
% Author: saurabh.me@gmail.com (Saurabh Singh).

rng('default');

PROC_DIR = [CONFIG.procDir 'LSP/cnn/'];
% PROC_DIR = [CONFIG.procRoot 'structPat_sgd/LSP/cnn/'];
if ~exist(PROC_DIR, 'dir')
  mkdir(PROC_DIR);
end

% Load the data
data_file = [CONFIG.stanfordFGCarDataDir 'cdhd_anno.mat'];
data = load(data_file, 'cdhd');
imdb = data.cdhd;

% Determine the training images.
is_visible = [imdb.is_visible];
is_train = [imdb.istrain];
is_train = find(is_train > 0 & is_visible > 0);
is_test = [imdb.istest];
is_test = find(is_test > 0 & is_visible > 0);

exp_id = 'e_cnn_centroid_chain_cdhd_dbg';

out_dir = [PROC_DIR exp_id '/'];
if ~exist(out_dir, 'dir')
  mkdir(out_dir);
end
model_dir = [out_dir 'models/'];
if ~exist(model_dir, 'dir')
  mkdir(model_dir);
end


target_inds = 1;

opts.target_joint_name = {'car_door_handle'};
opts.target_joints = target_inds;
opts.crop_target_joints = target_inds;
opts.do_augmentation = true;
opts.do_lr_flip = true;
opts.add_photometric_distortions = true;
opts.photometric_distortion_scale = 0.1;
opts.add_random_scale_jitter = true;
opts.scale_jitter_range = [6/10 10/8];
opts.add_random_crop_jitter = true;
opts.crop_jitter_fraction = 0.3;
opts.add_random_rotation_jitter = false;
opts.rotation_jitter_range = [-15 15]; % Angle in degrees
opts.add_dither = false;
opts.dither_factor = 25.5;
opts.adjust_for_scale = true;
opts.scaling_factor = 500/3;
opts.cudnnWorkspaceLimit = 1024*1024*1204 ; % 1GB

% Default scaling of the images.
opts.default_scaling = [];
opts.max_height = [];
opts.max_im_side = 500;
% Padding type: value, 'replicate', 'symmetric', 'circular'
opts.padding_type = 'replicate';
% Necessary to make regression learn anything.
opts.do_align_padding = false;
% Function used to flip the record.
opts.lr_flip_function = @lrFlipCDHDDataRecord;
opts.norm_height_function = @getCarHeight;
opts.image_aug_fn = @getAugmentedImageNew;

opts.start_offset = 16;
opts.output_stride = 16;

% opts.zero_margin_dist = 2 * opts.default_scaling;
% opts.one_margin_dist = 40 * opts.default_scaling;
% opts.margin_scale = 0.1;
opts.sigma = 15;

% opts.loc_pred_scale = 25 * opts.default_scaling;
% opts.l2_dist = 2 * opts.default_scaling / opts.loc_pred_scale;
opts.loc_pred_scale = 1;
opts.l2_dist = 1;

opts.img_root = CONFIG.stanfordFGCarDataDir;

opts.lite = false ;
opts.numFetchThreads = 0 ;
opts.train.batchSize = 10 ;
opts.train.super_batch = 1;
opts.train.numEpochs = 2600;
opts.train.continue = true ;
opts.train.useGpu = true ;
opts.train.prefetch = false ;
% Good performing is 0.0005
opts.train.learningRate = ...
  [0.01*ones(1, 2500) 0.001*ones(1, 50) 0.0001*ones(1,50)] * 0.005;
opts.train.weightDecay = 0.00001 ;
opts.train.expDir = out_dir;
% opts.train.errorType = 'location';
opts.train.errorType = 'loc_reg';
opts.train.display_errors = false;
opts.train.acceleration = 'adam';
% opts.train.acceleration = 'adadelta';
opts.train.adadelta_epsilon = 1e-8;
opts.train.adadelta_rho = 0.9;
opts.train.adam_epsilon = 1e-8;
opts.train.adam_beta1 = 0.9;
opts.train.adam_beta2 = 0.999;
opts.train.do_gradient_scaling = true;
opts.train.max_gradient_scale = 1000;
opts.train.do_gradient_clipping = false;
opts.train.grad_clip_thresh = 10;

% Set the image sets.
opts.train.train = is_train;
opts.train.val = is_test(round(linspace(1, length(is_test), 100)));

% [min_loc, max_loc, one_dist_loc] = determineLocFeatParams(imdb(is_train));
% opts.loc_params.min_loc = min_loc;
% opts.loc_params.max_loc = max_loc;
% opts.loc_params.one_dist_loc = one_dist_loc;
opts.num_landmark_feats = 400;
opts.useGpu = opts.train.useGpu;
opts.init_padding = 32;
% opts.max_dim = computeMaxInputDims(imdb(1:1000), opts);
opts.max_dim = [];
opts.do_random_padding = false;

net = initializeNetwork(opts) ;

net.comment = ['run on larger images (500 pix high).'];

% Compute the normalization parameters
norm_param_path = [opts.train.expDir 'norm_params.mat'];
if ~exist(norm_param_path, 'file')
  norm_params = computeNormalizationParameters(imdb, opts.img_root, 200, ...
    opts);
  save(norm_param_path, 'norm_params');
else
  load(norm_param_path, 'norm_params');
end
net.normalization.averagePixel = norm_params.mean_pixel;
% opts.mean_pixel = norm_params.mean_pixel;
% opts.std_pixel = norm_params.std_pixel;
opts.norm_params = norm_params;

% Save the options
opts_file = [opts.train.expDir 'options.mat'];
% if ~exist(opts_file, 'file')
save(opts_file, 'opts');
% end

rng('default');
batch_fn = @(imdb, batch, is_train) getBatchRevGrid(imdb, batch, ...
  is_train, opts);

opts.train.epochFunction = @epochNetTransformer;

[net, info] = pose_cnn_train(net, imdb, batch_fn, opts.train, ...
  'conserveMemory', true);
end

function max_dim = computeMaxInputDims(imdb, opts)
im_sizes = [imdb.imgdims];
max_dim = max(im_sizes(:));
max_dim = max_dim * opts.default_scaling;
if opts.add_random_scale_jitter
  max_dim = max_dim * opts.scale_jitter_range(end);
end
max_dim = ceil(max_dim);
max_dim = [max_dim max_dim];
end

function net = epochNetTransformer(epoch, net)
% if epoch  == 30
%   net.layers{end}.learningRate(5:6) = 1;
%   net.layers{end}.weights{5} = gpuArray(randn(1, 202, 'single')) ./ sqrt(202);
% end
end

function params = computeNormalizationParameters(imdb, img_root, num_imgs, opts)
% Compute the parameters to normalize the features. This includes the mean
% and the absolute max to limit the range of features to [-1 1]
is_train = [imdb.istrain];
is_train = find(is_train > 0);
num_imgs = min(num_imgs, length(is_train));
inx = randperm(length(is_train));
inx = inx(1:num_imgs);

pbar = createProgressBar();
mean_pixel = zeros(1, 3);
mean_pixel_sq = zeros(1, 3);
pixel_covar = zeros(3, 3);

num_pixel = 0;
for i = 1 : length(inx)
  pbar(i, length(inx));
  
  im_path = [img_root imdb(inx(i)).filepath];
  im = double(imread(im_path));
  
%   if opts.do_augmentation
%     [im, ~, ~] = getAugmentedImage(imdb(inx(i)), im, opts);
%   end
  
  scale = computeDefaultScaleForImage(im, opts);
  im = imresize(im, scale, 'bilinear');
  
  im_size = size(im);
  if length(im_size) < 3 ||  im_size(3) == 1
    im = repmat(im, [1 1 3]);
  end

  im = reshape(im, [], 3);
  
  npix = size(im, 1);
  
  mean_pixel = mean_pixel * (num_pixel / (num_pixel + npix)) ...
    + sum(im, 1) / (num_pixel + npix);
  mean_pixel_sq = mean_pixel_sq * (num_pixel / (num_pixel + npix)) ...
    + sum(im.^2, 1) / (num_pixel + npix);
  pixel_covar = pixel_covar * (num_pixel / (num_pixel + npix)) ...
    + (im' * im) / (num_pixel + npix);

  num_pixel = num_pixel + npix;
end

epsilon = 0.001;
params.mean_pixel = reshape(mean_pixel, 1, 1, []);
params.std_pixel = reshape(sqrt(mean_pixel_sq - mean_pixel.^2), 1, 1, []) ...
  + epsilon;
params.pixel_covar = pixel_covar;
[params.pixel_eigvec, params.pixel_eigval] = eigs( ...
  pixel_covar - mean_pixel'*mean_pixel);
params.pixel_eigval = sqrt(diag(params.pixel_eigval));

if opts.add_photometric_distortions
  params.std_pixel = params.std_pixel ...
    + reshape(opts.photometric_distortion_scale * sqrt(params.pixel_eigval), ...
    1, 1, 3);
end
end

function net = initializeNetwork(opts)

opts.weightDecay = 1;
opts.init_bias = 0.1;

grid_size = 50;
offset_grid = reshape(getOffsetGrid(grid_size, 5)', 1, 2, []);

net.layers = {} ;

net = addBlock(net, opts, 1, 3, 3, 3, 32, 2, 0) ;
net = addBlock(net, opts, 2, 3, 3, 32, 64, 2, 0) ;
                         
net = addBlock(net, opts, 3, 3, 3, 64, 64, 2, 0) ;
net = addBlock(net, opts, 4, 3, 3, 64, 64, 1, 1) ;

net = addBlock(net, opts, 5, 3, 3, 64, 128, 2, 0) ;
net = addBlock(net, opts, 6, 5, 5, 128, 128, 1, 2) ;

filter_width = 5;
padding = (filter_width - 1) / 2;
prev_pred_weight = 0.1;
% Weights for the individual offset predictions.
offset_pred_weight = 0.1;

num_steps = 3;
num_out_filters = size(offset_grid, 3) + 1;
nfc = 128;

net.layers{end+1} = struct( ...
  'type', 'custom', ...
  'name', 'loc_attn', ...
  'forward', @(layer, res_i, res_ip1) centroidChainGrid9LossLayer( ...
    layer, res_i, res_ip1, true), ...
  'backward', @(layer, res_i, res_ip1) centroidChainGrid9LossLayer(...
    layer, res_i, res_ip1, false), ...
  'gt_coords', [], ...
  'loc_pred_scale', opts.loc_pred_scale, ...
  'num_targets', length(opts.target_joints), ...
  'visualize', false, ...
  'pred', [], ...
  'pad', padding, ...
  'stride', 1, ...
  'wt_transfer', 'softmax', ...
  'use_gaussian_at_offset', true, ...
  'sigma', opts.sigma, ...
  'l2_dist', opts.l2_dist, ...
  'steps', num_steps, ...
  'offset_grid', offset_grid, ...
  'prev_pred_weight', prev_pred_weight, ...
  'offset_pred_weight', offset_pred_weight, ...
  'pred_factor', 50, ...
  'do_gradient_scaling', true, ...
  'max_gradient_scale', 1000, ...
  'shared_layers', [7 8 9 10], ...
  'weights', {{ ...
      randn(filter_width, filter_width, nfc+1, nfc, 'single'),  ...
      opts.init_bias * ones(1, nfc, 'single'), ...
      randn(filter_width, filter_width, nfc+1, nfc, 'single'),  ...
      opts.init_bias * ones(1, nfc, 'single'), ...
      randn(filter_width, filter_width, nfc+1, nfc, 'single'),  ...
      opts.init_bias * ones(1, nfc, 'single'), ...
      randn(filter_width, filter_width, nfc, nfc, 'single'),  ...
      opts.init_bias * ones(1, nfc, 'single'), ...
      randn(filter_width, filter_width, nfc, num_out_filters, 'single'),  ...
      zeros(1, num_out_filters, 'single'), ...
      }}, ...
  'learningRate', [1 1 1 1 1 1 1 1 1 1], ...
  'weightDecay', [1 0 1 0 1 0 1 0 1 0]);


% Scale the initializations. 0.01 seems better for momentum.
% for i = 1 : length(net.layers)-2
wt_scale = sqrt(2);
for i = 1 : length(net.layers)
  if strcmp(net.layers{i}.type, 'conv')
    s = size(net.layers{i}.weights{1});
    scal = wt_scale * sqrt(1 / prod(s(1:3)));
%     scal = 0.01;
    net.layers{i}.weights{1} = net.layers{i}.weights{1}.* scal;
  end
  if isfield(net.layers{i}, 'name') && strcmp(net.layers{i}.name, 'loc_attn')
    wts_to_normalize = [1 3 5 7 9];
    for j = wts_to_normalize
      s = size(net.layers{i}.weights{j});
      scal = wt_scale * sqrt(1 / prod(s(1:3)));
      net.layers{i}.weights{j} = net.layers{i}.weights{j}.* scal;
    end
  end
  if isfield(net.layers{i}, 'name') && strcmp(net.layers{i}.name, 'recurconv')
    s = size(net.layers{i}.weights{1});
    scal = wt_scale * sqrt(1 / prod(s(1:3)));
%     scal = 0.01;
    net.layers{i}.weights{1} = net.layers{i}.weights{1}.* scal;
  end
end

if opts.useGpu
  for l = 1 : length(net.layers)
    if isfield(net.layers{l}, 'weights')
      for i = 1 : length(net.layers{l}.weights)
        net.layers{l}.weights{i} = gpuArray(net.layers{l}.weights{i});
      end
    end
  end
end

% Fill in default values
net = vl_simplenn_tidy(net) ;

end

function net = addBlock(net, opts, id, h, w, in, out, stride, pad)
info = vl_simplenn_display(net) ;
fc = (h == info.dataSize(1,end) && w == info.dataSize(2,end)) ;
if fc
  name = 'fc' ;
else
  name = 'conv' ;
end
net.layers{end+1} = struct( ...
  'type', 'conv', ...
  'name', sprintf('%s%d', name, id), ...
  'weights', {{randn(h, w, in, out, 'single'), ...
      opts.init_bias * ones(1, out, 'single')}}, ...
  'stride', stride, ...
  'pad', pad, ...
  'learningRate', [1 2], ...
  'weightDecay', [opts.weightDecay 0]) ;
net.layers{end+1} = struct('type', 'relu', 'name', sprintf('relu%d',id)) ;

end

