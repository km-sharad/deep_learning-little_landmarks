function scale = computeDefaultScaleForImage(im, opts)

if ~isempty(opts.default_scaling)
  scale = opts.default_scaling;
  return;
end

if ~isempty(opts.max_height)
  max_height = opts.max_height;
  scale = max_height / size(im, 1);
  return;
end

if ~isempty(opts.max_im_side)
  max_im_side = opts.max_im_side;
  im_size = size(im);
  scale = (max_im_side - 0.01) / max(im_size(1:2));
  return;
end

error('Couldnt compute the default scaling.');
end
