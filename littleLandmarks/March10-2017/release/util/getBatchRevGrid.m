function [im, meta] = getBatchRevGrid(imdb, batch, is_train, opts)
% is_train: flag indicating if this is training or testing.
%
% im: rescaled im such that the longest size is 227
% gt_coords: coordinate to be considered gt in the output map.
% meta: metadata.
%
% Author: saurabh.me@gmail.com (Saurabh Singh).

opts = getParamsWithDefaults(opts, {'default_scaling', 1, 'init_padding', 0, ...
  'max_dim', [], 'do_random_padding', false, 'norm_height_function', ...
  @getTorsoHeight, 'padding_type', 'replicate', ...
  'image_aug_fn', @getAugmentedImage});

% Output locations are start_offset:output_stride:end
start_offset = opts.start_offset;
output_stride = opts.output_stride;

images = cell(size(batch));
infos = cell(size(batch));
target_locs = cell(size(batch));
for i = 1 : length(batch)
  [images{i}, target_locs{i}, infos{i}] = getImage(imdb, batch(i), opts, ...
    is_train, output_stride);
end

im_sizes = cellfun(@(x) x.im_size, infos, 'UniformOutput', false);
im_sizes = cell2mat(im_sizes');
padding = cellfun(@(x) x.padding, infos, 'UniformOutput', false);
padding = cat(4, padding{:});
aug_target_loc = cellfun(@(x) x.aug_target_loc, infos, 'UniformOutput', false);
aug_target_loc = cat(4, aug_target_loc{:});
final_scale = cellfun(@(x) x.final_scale, infos, 'UniformOutput', false);
final_scale = cat(4, final_scale{:});
torso_height = cellfun(@(x) x.torso_height, infos, 'UniformOutput', false);
torso_height = cat(4, torso_height{:});
im_org = cellfun(@(x) x.im_org, infos, 'UniformOutput', false);
im_org_scaled = cellfun(@(x) x.im_org_scaled, infos, 'UniformOutput', false);

[im, cat_padding] = concatWithPadding(images, im_sizes, opts.max_dim, ...
  opts.do_random_padding, opts.padding_type);

new_im_size = size(im);
yy = 1 : new_im_size(1);
xx = 1 : new_im_size(2);
[x, y] = meshgrid(xx, yy);

out_yi = start_offset : output_stride : new_im_size(1)-start_offset+1;
out_xi = start_offset : output_stride : new_im_size(2)-start_offset+1;
out_x = x(out_yi, out_xi);
out_y = y(out_yi, out_xi);
out_locs = [out_x(:) out_y(:)];


if opts.useGpu
  out_locs = gpuArray(out_locs);
end

target_loc = cat(4, target_locs{:});
target_loc = bsxfun(@plus, target_loc, ...
  permute(cat_padding(:, [3 1], 1, :), [2 1 3 4]));
gt_coords = (permute(target_loc, [3 1 2 4]) - start_offset) / output_stride + 1;
org_gt_coords = permute(target_loc, [3 1 2 4]);
padding = padding + cat_padding;

if opts.adjust_for_scale
  out_locs = out_locs ./ opts.scaling_factor;
  org_gt_coords = org_gt_coords ./ opts.scaling_factor;
  aug_target_loc = aug_target_loc ./ opts.scaling_factor;
end


meta.margins = [];
meta.out_locs = out_locs;
meta.scale = final_scale;
meta.padding = padding;
meta.gt_coords = gt_coords;
meta.org_gt_coords = org_gt_coords;
meta.aug_gt_coords = aug_target_loc;
meta.torso_height = torso_height;
meta.im_org = im_org;
meta.im_org_scaled = im_org_scaled;
end

function [batch_im, padding] = concatWithPadding(images, im_sizes, max_dim, ...
  do_random_padding, padding_type)
if isempty(max_dim)
  max_dim = max(im_sizes, [], 1);
end

if ~do_random_padding
  padding = zeros(length(images), 4);
  padding(:, [2 4]) = bsxfun(@minus, max_dim(1:2), im_sizes(:, 1:2));
else
  total_padding = bsxfun(@minus, max_dim(1:2), im_sizes(:, 1:2));
  prefix_padding = floor(rand(size(total_padding)) .* total_padding);
  
  padding = [prefix_padding(:, 1) total_padding(:, 1)-prefix_padding(:, 1) ...
    prefix_padding(:, 2) total_padding(:, 2)-prefix_padding(:, 2)];
end

for i = 1 : length(images)
  images{i} = imPad(images{i}, padding(i, :), padding_type);
end
batch_im = cat(4, images{:});
padding = reshape(padding', 1, 4, 1, []);
end

function [im, target_loc, info] = getImage(imdb, im_id, opts, is_train, ...
  output_stride)
im_record = imdb(im_id);
im_path = [opts.img_root im_record.filepath];
im = single(imread(im_path));
im_org = im;

% Compute the default scaling.
scale = computeDefaultScaleForImage(im, opts);

% Do data augmentation.
aug_scale = 1;
if is_train && opts.do_augmentation
  [im, target_loc, aug_scale] = opts.image_aug_fn(im_record, im, opts);
else
  target_loc = im_record.coords(:, opts.target_joints);
%   target_loc = target_loc + [30; -30];
end

aug_target_loc = target_loc;

try
im_org_scaled = imresize(im, scale, 'bilinear');
catch
  keyboard;
end
im = im_org_scaled;

im_size = size(im);
if length(im_size) < 3 || im_size(3) == 1
  im = repmat(im, [1 1 3]);
  im_size = size(im);
end

im = bsxfun(@minus, im, opts.norm_params.mean_pixel(1:3));
im = bsxfun(@rdivide, im, opts.norm_params.std_pixel(1:3));

% target_loc = round(target_loc .* scale);
target_loc = target_loc .* scale;
padding = zeros(1, 4); % T/B/L/R

padding = padding + opts.init_padding;
target_loc = target_loc + opts.init_padding;
im_size(1:2) = im_size(1:2) + 2 * opts.init_padding;

% Handle the case where the target is outside the boundary. In some datasets
% this may happen due to rounding e.g. LSP and in others the joints are
% annotated outside the boundary e.g. fashion pose.

min_target_loc = floor(min(target_loc, [], 2));
lt_pad = max(max([1 1] - min_target_loc', 0));
padding([1 3]) = padding([1 3]) + lt_pad;
im_size(1:2) = im_size(1:2) + lt_pad;
target_loc = target_loc + lt_pad;

max_target_loc = ceil(max(target_loc, [], 2));
br_pad = max(max(max_target_loc([2 1])' - im_size(1:2), 0));
padding([2 4]) = padding([2 4]) + br_pad;
im_size(1:2) = im_size(1:2) + br_pad;

% Pad if smaller than minimum size.
min_side = 64;
min_side_padding = max(min_side - im_size(1:2), 0);
padding([2 4]) = padding([2 4]) + min_side_padding;
im_size(1:2) = im_size(1:2) + min_side_padding;

% Pad the image if necessary to get the final size correct.
size_padding = computePaddingForImage(im_size, 0, output_stride);
padding([2 4]) = padding([2 4]) + size_padding;
im_size(1:2) = im_size(1:2) + size_padding;

% im = padarray(im, padding, 'replicate', 'post');
im = imPad(im, padding, opts.padding_type);

assert(all(im_size == size(im)));

final_scale = aug_scale * scale;

info.aug_scale = aug_scale;
info.aug_target_loc = aug_target_loc';
info.im_org = im_org;
info.im_size = im_size;
info.final_scale = final_scale;
info.torso_height = opts.norm_height_function(im_record, ...
  'diagonal') * final_scale;
info.im_org_scaled = imPad(im_org_scaled, padding, opts.padding_type);
info.padding = padding;
end

function padding = computePaddingForImage(im_size, filter_size, stride)
% The network eats a boundary of 3 pixels and has a stride of 4 pixels.
padding = rem(im_size(1:2) - filter_size, stride);
padding = rem(stride - padding, stride);
end

