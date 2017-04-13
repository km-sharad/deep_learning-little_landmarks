function res = centroidChainGrid9LossLayer(layer, res_i, res_ip1, do_forward)
% This is almost the same as grid8 except that only one feature channel (the
% rbf at predicted offset) is passed ahead. Experimentally this works as well as
% grid8.
%
% Author: saurabh.me@gmail.com (Saurabh Singh).

layer = getParamsWithDefaults(layer, {'do_gradient_scaling', false, ...
  'max_gradient_scale', 100, 'do_gradient_block', false, ...
  'shared_prefix', false});

steps = layer.steps;
transition_dist = layer.l2_dist;
loc_pred_scale = layer.loc_pred_scale;
offset_pred_weight = layer.offset_pred_weight;
pred_factor = layer.pred_factor;
gt_loc = layer.org_gt_coords ./ loc_pred_scale;
shared_layers = layer.shared_layers;
shared_prefix = layer.shared_prefix;

x = res_i.x;

if do_forward
  % Append the features with extra channels
  n = size(x, 1) * size(x, 2);
  xa = ones(size(x, 1), size(x, 2), 1, size(x, 4), 'like', x) ./ n ...
    * pred_factor;
  x = {cat(3, xa, x), [], [], [], 0};
  
  % Do the forwards
  res_steps = cell(1, steps);
  all_preds = zeros(steps, 2, 1, size(x{1}, 4), 'like', x{1});
  all_cents = zeros(steps, 2, 1, size(x{1}, 4), 'like', x{1});
  for i = 1 : steps
    if shared_prefix
      step_wts = [shared_layers 2*i-1 2*i];
    else
      step_wts = [2*i-1 2*i shared_layers];
    end
    res_steps{i} = doForwardPass(layer, x, layer.weights(step_wts));
    x{1}(:, :, 1, :) = res_steps{i}.x{1} * pred_factor;
    x{2} = res_steps{i}.x{2};
    x{3} = res_steps{i}.x{3};
    x{4} = res_steps{i}.x{4};
    x{5} = res_steps{i}.x{5};
    all_preds(i, :, :, :) = x{2} * loc_pred_scale;
    all_cents(i, :, :, :) = res_steps{i}.pc * loc_pred_scale;
  end
  
  % Compute the loss.
  [target_loss, target_residue] = computePredictionLossSL1( ...
    res_steps{end}.x{2}, gt_loc, transition_dist);
  target_loss = sum(target_loss, 2);
  [offs_loss, offs_residue] = computePredictionLossSL1(res_steps{end}.x{3}, ...
    gt_loc, transition_dist);
  offs_loss = sum(bsxfun(@times, sum(offs_loss, 2), res_steps{end}.x{4}), 1);
  
  loss = sum(target_loss + res_steps{end}.x{end} ...
    + offs_loss * offset_pred_weight);
  
  pred = res_steps{end}.x{2} * loc_pred_scale;
  
  res = res_ip1;
  res.x = loss;
  res.aux.pred = pred;
  res.aux.all_preds = all_preds;
  res.aux.all_cents = all_cents;
  res.aux.res_steps = res_steps;
  res.aux.nzw_frac = 0;
  res.aux.target_residue = target_residue;
  res.aux.offs_residue = offs_residue;
  res.aux.offs_loss = offs_loss * offset_pred_weight;
else
  res = res_i;
  target_residue = res_ip1.aux.target_residue;
  offs_residue = res_ip1.aux.offs_residue;
  offs_loss = res_ip1.aux.offs_loss;
  res_steps = res_ip1.aux.res_steps;
%   res_x = res_ip1.x;
  
  res.dzdx = x .* 0;
  
  target_grad = residue2gradientSL1(target_residue, transition_dist);
  grad = cell(1, 3);
  grad{1} = zeros(size(res_steps{end}.x{1}), 'like', res_steps{end}.x{1});
  grad{2} = target_grad;
  
  offs_gradient = residue2gradientSL1(offs_residue, transition_dist);
  grad{3} = bsxfun(@times, offs_gradient, res_steps{end}.x{4}) ...
    * offset_pred_weight;
  grad{4} = getOffsetNWGradient(offs_residue, transition_dist) ...
    * offset_pred_weight;

  grad{5} = 0;
  back_res = cell(1, steps);
  res.dzdw = cell(size(layer.weights));
  
  % Append the features with extra channels
  n = size(x, 1) * size(x, 2);
  xa = ones(size(x, 1), size(x, 2), 1, size(x, 4), 'like', x) ./ n ...
    * pred_factor;
  x_in = {cat(3, xa, x), [], [], [], 0};
  next_offs_loss = offs_loss;
  
  for i = steps : -1 : 1
    if i == 1
      x_in{1}(:, :, 1, :) = xa;
      x_in{2} = [];
      x_in{3} = [];
      x_in{4} = 0;
      x_in{5} = 0;
    else
      x_in{1}(:, :, 1, :) = res_steps{i-1}.x{1} * pred_factor;
      x_in{2} = res_steps{i-1}.x{2};
      x_in{3} = res_steps{i-1}.x{3};
      x_in{4} = res_steps{i-1}.x{4};
      x_in{5} = res_steps{i-1}.x{5};
    end
    
    res_steps{i}.dzdx = grad;
    res_steps{i}.next_offs_loss = next_offs_loss;

    if shared_prefix
      step_wts = [shared_layers 2*i-1 2*i];
    else
      step_wts = [2*i-1 2*i shared_layers];
    end

    back_res{i} = doBackwardPass(layer, x_in, layer.weights(step_wts), ...
      res_steps{i});
    grad = back_res{i}.dzdx;
    grad{1} = grad{1}(:, :, 1, :) * pred_factor;
    next_offs_loss = res_steps{i}.offs_loss;
    
    for j = 1 : length(step_wts)
      if isempty(res.dzdw{step_wts(j)})
        res.dzdw{step_wts(j)} = back_res{i}.dzdw{j};
      else
        res.dzdw{step_wts(j)} = res.dzdw{step_wts(j)} + back_res{i}.dzdw{j};
      end
    end
    
    res.dzdx = res.dzdx + back_res{i}.dzdx{1}(:, :, 2:end, :);
    
%     res_steps{i} = [];
%     back_res{i} = [];
  end
end

% res = doPasses(layer, res_i, res_ip1, do_forward);
end

function res = doForwardPass(layer, x, weights)

if ~iscell(x)
  error('x should be a cell array of length 3');
end

prev_pred = x{2};
prev_loss = x{5};
prev_offsets = x{3};
prev_nw = x{4};
x = x{1};

chained = true;
if isempty(prev_pred)
  chained = false;
end

assert(size(x, 1) * size(x, 2) == size(layer.out_locs, 1));

loc_pred_scale = layer.loc_pred_scale;
wt_transfer = layer.wt_transfer;
use_gaussian_at_offset = layer.use_gaussian_at_offset;
prev_pred_weight = layer.prev_pred_weight;
offset_pred_weight = layer.offset_pred_weight;
transition_dist = layer.l2_dist;

out_locs_rs = layer.out_locs ./ loc_pred_scale;
sigma = layer.sigma / loc_pred_scale;
offset_grid = layer.offset_grid ./ loc_pred_scale;
num_offset_channels = size(offset_grid, 3);
offset_channels = 1 + (1 : num_offset_channels);
num_chans = length(offset_channels) + 1;

steps = length(weights) / 2;
interim = cell(1, steps);
for i = 1 : steps
  if i == 1
    x_in = x;
  else
    x_in = interim{i-1};
  end
  
  a = vl_nnconv(x_in, weights{2*i-1}, weights{2*i}, ...
    'pad', layer.pad, 'stride', layer.stride);
  if i == steps
    interim{i} = a;
  else
    interim{i} = a .* (a > 0);
  end
end

w = interim{end}(:, :, 1, :);
[nw, dwn] = getNormalizedLocationWeightsFast(wt_transfer, w);

% Predict the centroid.
pc = sum(bsxfun(@times, out_locs_rs, reshape(nw, [], 1, 1, size(nw, 4))), 1);

% Predict the offset.
% Use the offset grid to compute the offsetd.
offset_wts = interim{end}(:, :, offset_channels, :);
offset_max = max(offset_wts, [], 3);
offset_wts = bsxfun(@minus, offset_wts, offset_max);
offset_wts = exp(offset_wts);
sum_offset_wts = sum(offset_wts, 3);
offset_wts = bsxfun(@rdivide, offset_wts, sum_offset_wts);
of_x = bsxfun(@times, offset_grid(1, 1, :), offset_wts);
of_y = bsxfun(@times, offset_grid(1, 2, :), offset_wts);
of_x = sum(of_x, 3);
of_y = sum(of_y, 3);

po = cat(3, of_x, of_y);
%     po = a(:, :, 2:3);
poc = sum(sum(bsxfun(@times, po, nw), 1), 2);
poc = reshape(poc, 1, [], 1, size(poc, 4));

if use_gaussian_at_offset
  f = interim{end};
  feat_size = [size(f, 1) size(f, 2) 1 size(f, 4)];
  offset_gauss = doOffset2GaussianForward(pc + poc, out_locs_rs, sigma, ...
    feat_size);
end
  
if chained
  [cent_loss, cent_residue] = computePredictionLossSL1(prev_pred, pc, ...
    transition_dist);
  cent_loss = sum(cent_loss, 2);
  [offs_loss, offs_residue] = computePredictionLossSL1(prev_offsets, pc, ...
    transition_dist);
  offs_loss = sum(bsxfun(@times, sum(offs_loss, 2), prev_nw), 1);
else
  cent_residue = pc * 0;
  cent_loss = 0;
  offs_residue = 0;
  offs_loss = 0;
end

indiv_preds = bsxfun(@plus, out_locs_rs, reshape(po, [], 2, 1, size(x, 4)));
indiv_nw = reshape(nw, [], 1, 1, size(x, 4));

loss = prev_loss ...
  + (cent_loss + offs_loss * offset_pred_weight) * prev_pred_weight;

if use_gaussian_at_offset
  if rem(num_chans, 2) ~= 0
    error('Number of channels to fill should be even.');
  end
  xx = offset_gauss;
  out_x = {xx, pc + poc, indiv_preds, indiv_nw, loss};
else
  offset_feat = interim{end}(:, :, offset_channels, :);
  xx = offset_feat;
  out_x = {xx,  pc + poc, indiv_preds, indiv_nw, loss};
end

res.x = out_x;
res.pred = pc * loc_pred_scale;
res.w = w;
res.pc = pc;
res.po = po;
res.poc = poc;
res.nw = nw;
res.dwn = dwn;
res.offset_wts = offset_wts;
res.offs_loss = offs_loss * offset_pred_weight * prev_pred_weight;
res.cent_residue = cent_residue;
res.offs_residue = offs_residue;
res.nzw_frac = sum(nw(:) > 0) / numel(nw);
res.interim = interim;
end

function res = doBackwardPass(layer, x, weights, res_ip1)
if ~iscell(x)
  error('x should be a cell array of length 3');
end

prev_pred = x{2};
prev_loss = x{5};
prev_offsets = x{3};
prev_nw = x{4};
x = x{1};

chained = true;
if isempty(prev_pred)
  chained = false;
end

assert(size(x, 1) * size(x, 2) == size(layer.out_locs, 1));

loc_pred_scale = layer.loc_pred_scale;
use_gaussian_at_offset = layer.use_gaussian_at_offset;
prev_pred_weight = layer.prev_pred_weight;
offset_pred_weight = layer.offset_pred_weight;
transition_dist = layer.l2_dist;
do_gradient_block = layer.do_gradient_block;

out_locs_rs = layer.out_locs ./ loc_pred_scale;
sigma = layer.sigma / loc_pred_scale;
offset_grid = layer.offset_grid ./ loc_pred_scale;
num_offset_channels = size(offset_grid, 3);
offset_channels = 1 + (1 : num_offset_channels);

cent_residue = res_ip1.cent_residue;
offs_residue = res_ip1.offs_residue;
pc = res_ip1.pc;
po = res_ip1.po;
poc = res_ip1.poc;
dwn = res_ip1.dwn;
nw = res_ip1.nw;
offset_wts = res_ip1.offset_wts;
interim = res_ip1.interim;
res_x = res_ip1.x;

cent_residue = residue2gradientSL1(cent_residue, transition_dist);
cent_residue = cent_residue * prev_pred_weight;

offs_gradient = residue2gradientSL1(offs_residue, transition_dist);
offs_gradient = bsxfun(@times, offs_gradient, prev_nw) ...
  * offset_pred_weight * prev_pred_weight;

offs_nw_gradient = getOffsetNWGradient(offs_residue, transition_dist) ...
  * offset_pred_weight * prev_pred_weight;

del_o = reshape(res_ip1.dzdx{2}, [1 1 2 size(x, 4)]);

di = res_ip1.dzdx{1};

if use_gaussian_at_offset
  doff = doOffset2GaussianBackward(pc+poc, out_locs_rs, sigma, ...
    res_x{1}(:, :, 1, :), di(:, :, 1, :));
  del_o = del_o + doff;
end

dn_o = sum(bsxfun(@times, del_o, po), 3);

pc_residue = reshape(del_o, 1, 2, 1, size(x, 4));
if chained
  if ~do_gradient_block
    pc_residue = pc_residue - cent_residue - sum(offs_gradient, 1);
  end
end
dn_c = reshape(sum(bsxfun(@times, pc_residue, out_locs_rs), 2), size(nw));

% Gradient due to individual regression losses.
dn_oi = reshape(res_ip1.dzdx{4}, size(dn_o));

dn = dn_o + dn_c + dn_oi;

dw = dwn .* bsxfun(@minus, dn, sum(sum(nw .* dn, 1), 2));

dpo = bsxfun(@times, del_o, nw);
dpo = dpo + reshape(res_ip1.dzdx{3}, size(dpo));

% Gradients for the offset.
dpo_f = (bsxfun(@times, ...
  dpo(:, :, 1, :), ...
  bsxfun(@minus, offset_grid(1, 1, :, :), po(:, :, 1, :))) ...
  + bsxfun(@times, ...
  dpo(:, :, 2, :), ...
  bsxfun(@minus, offset_grid(1, 2, :, :), po(:, :, 2, :)))) .* offset_wts;
dpo = dpo_f;
if use_gaussian_at_offset
  di(:, :, 1, :) = dw;
  di(:, :, offset_channels, :) = dpo;
else
  error('Not supported');
end

res_dzdx = di;
steps = length(weights) / 2;
dzdw = cell(size(weights));
for i = steps : -1 : 1
  if i ~= steps
    res_dzdx = res_dzdx .* (interim{i} > 0);
  end
  
  if i == 1
    x_in = x;
  else
    x_in = interim{i-1};
  end
  
  [dzdx, dzdw{2*i-1}, dzdw{2*i}] = vl_nnconv(x_in, weights{2*i-1}, ...
    weights{2*i}, res_dzdx, 'pad', layer.pad, 'stride', layer.stride);
  res_dzdx = dzdx;
end


res.dzdx = {dzdx, cent_residue, offs_gradient, sum(offs_nw_gradient, 2), 0};
res.dzdw = dzdw;
end

function offs_nw_gradient = getOffsetNWGradient(offs_residue, transition_dist)
offs_nw_gradient = abs(offs_residue);
s = abs(offs_nw_gradient) >= transition_dist;
offs_nw_gradient(s) = offs_nw_gradient(s) - transition_dist / 2;
offs_nw_gradient(~s) = (offs_nw_gradient(~s).^2) / 2 / transition_dist;
offs_nw_gradient = sum(offs_nw_gradient, 2);
end

function feat = doOffset2GaussianForward(offset, locs, sigma, feat_size)
feat = sum(bsxfun(@minus, offset, locs).^2, 2) / 2 ./ sigma^2;
feat = exp(-reshape(feat, feat_size));
end

function dzdx = doOffset2GaussianBackward(offset, locs, sigma, feat, dzdy)
delta = feat .* dzdy;
residue = -bsxfun(@minus, offset, locs);
doff = sum( ...
  bsxfun(@times, residue, reshape(delta, [], 1, 1, size(feat, 4))), 1);
dzdx = reshape(doff, [1 1 2 size(feat, 4)])  ./ sigma^2;
end
