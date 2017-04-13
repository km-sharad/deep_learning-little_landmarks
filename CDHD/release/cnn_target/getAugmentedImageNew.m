function [im, targets, scale] = getAugmentedImageNew(im_record, im, opts)
opts = getParamsWithDefaults(opts, {'lr_flip_function', @lrFlipDataRecord, ...
  'max_padding', 32, 'padding_type', 'replicate', ...
  'add_dither', false, 'dither_factor', 1});

% Randomly flip the image.
if opts.do_lr_flip
  lr_flip_function = opts.lr_flip_function;
  if rand(1) > 0.5
    [im, im_record] = lr_flip_function(im, im_record);
  end
end
targets = im_record.coords(:, opts.target_joints);
% targets = targets + [30; -30];

if opts.add_photometric_distortions
%   distortions = generateRandomDistortionsForImage(im);
%   im = distortImage(im, distortions);
  
  im = doEigenBasedDistortion(im, ...
    opts.norm_params.pixel_eigvec, ...
    opts.norm_params.pixel_eigval, ...
    opts.photometric_distortion_scale);
end

scale = 1;
if opts.add_random_scale_jitter
  scale_range = log(opts.scale_jitter_range);
  scale = exp(rand(1) ...
    * (scale_range(2) - scale_range(1)) + scale_range(1));
  im = imresize(im, scale, 'bilinear');
  targets = targets * scale;
end
  
if opts.add_random_crop_jitter
  crop_targets = im_record.coords(:, opts.crop_target_joints);
  crop_targets = crop_targets * scale;
  [im, cropped_box] = addRandomCropJitter(im, crop_targets, ...
    opts.crop_jitter_fraction, opts.max_padding, opts.padding_type);
  targets = bsxfun(@minus, targets, cropped_box(1:2)'-1);
else
  cropped_box = [];
end

if opts.add_random_rotation_jitter
  [im, targets, theta] = addRandomRotationJitter(opts.rotation_jitter_range, ...
    im, targets);
else
  theta = 0;
end

if opts.add_dither
  im = addDither(opts.dither_factor, im);
end
end

function im = addDither(dither_factor, im)
im  = im + randn(size(im), 'like', im) * dither_factor;
end

function [I, targets, theta] = addRandomRotationJitter(range, I, targets)
theta = rand(1) * (range(2) - range(1)) + range(1);
[I, targets] = rotateImageDataNew(I, targets, theta);
end

function [im, cropped_box] = addRandomCropJitter(im, targets, ...
  crop_jitter_fraction, max_padding, padding_type)
im_size = size(im);
im_size = im_size(1:2);
crops = getCroppingBounds(targets, im_size) .* crop_jitter_fraction;
crops = floor(crops);
% negative padding is cropping.
padding = round(rand(size(crops)) .* (max_padding + crops)) - crops;
% cropped_box = getRandomCrop(crops, im_size);
cropped_box = [1 1 im_size([2 1])] - [padding(1:2) -padding(3:4)] ;
% im = im(cropped_box(2):cropped_box(4), cropped_box(1):cropped_box(3), :);
im = imPad(im, padding([2 4 1 3]), padding_type);
end
