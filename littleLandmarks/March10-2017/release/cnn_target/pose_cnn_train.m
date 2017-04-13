function [net, info] = pose_cnn_train(net, imdb, getBatch, varargin)
% CNN_TRAIN   Demonstrates training a CNN
%    CNN_TRAIN() is an example learner implementing stochastic gradient
%    descent with momentum to train a CNN for image classification.
%    It can be used with different datasets by providing a suitable
%    getBatch function.

opts.train = [] ;
opts.val = [] ;
opts.numEpochs = 300 ;
opts.batchSize = 256 ;
opts.useGpu = false ;
opts.learningRate = 0.001 ;
opts.continue = false ;
opts.expDir = fullfile('data','exp') ;
opts.conserveMemory = false ;
opts.sync = true ;
opts.prefetch = false ;
opts.weightDecay = 0.0005 ;
opts.momentum = 0.9 ;
opts.errorType = 'multiclass' ;
opts.plotDiagnostics = false ;
opts.display_errors = true;
opts.acceleration = 'momentum';
opts.adadelta_epsilon = 1e-8;
opts.adadelta_rho = 0.95;
opts.rmsprop_epsilon = 1e-8;
opts.rmsprop_rho = 0.9;
opts.adam_epsilon = 1e-8;
opts.adam_beta1 = 0.9;
opts.adam_beta2 = 0.999;
opts.super_batch = 5;
opts.epochFunction = [];
opts.checkpoint_interval = 1;
opts.do_gradient_scaling = false;
opts.max_gradient_scale = 1000;
opts.do_gradient_clipping = false;
opts.grad_clip_thresh = 1;

opts = vl_argparse(opts, varargin) ;

if ~exist(opts.expDir, 'dir'), mkdir(opts.expDir) ; end
if isempty(opts.train), opts.train = find(imdb.images.set==1) ; end
if isempty(opts.val), opts.val = find(imdb.images.set==2) ; end
if isnan(opts.train), opts.train = [] ; end

% -------------------------------------------------------------------------
%                                                    Network initialization
% -------------------------------------------------------------------------

evaluateMode = isempty(opts.train) ;

if ~evaluateMode
  for i=1:numel(net.layers)
    if isfield(net.layers{i}, 'weights')
      J = numel(net.layers{i}.weights) ;
      for j=1:J
        net.layers{i}.momentum{j} = zeros(size(net.layers{i}.weights{j}), ...
          'like', net.layers{i}.weights{j}) ;
      end
      if ~isfield(net.layers{i}, 'learningRate')
        net.layers{i}.learningRate = ones(1, J, 'single') ;
      end
      if ~isfield(net.layers{i}, 'weightDecay')
        net.layers{i}.weightDecay = ones(1, J, 'single') ;
      end
    end
  end
end

if opts.useGpu
  net = vl_simplenn_move(net, 'gpu') ;
end

% -------------------------------------------------------------------------
%                                                        Train and validate
% -------------------------------------------------------------------------

rng(0) ;

if opts.useGpu
  one = gpuArray(single(1)) ;
else
  one = single(1) ;
end

info.train.objective = [] ;
info.train.error = [] ;
info.train.topFiveError = [] ;
info.train.speed = [] ;
info.val.objective = [] ;
info.val.error = [] ;
info.val.topFiveError = [] ;
info.val.speed = [] ;

modelPath = @(ep) fullfile(opts.expDir, 'models', ...
  sprintf('net-epoch-%d.mat', ep));
start = opts.continue * findLastCheckpoint([opts.expDir 'models/']) ;
if start >= 1
  fprintf('%s: resuming by loading epoch %d\n', mfilename, start) ;
  load(modelPath(start), 'net', 'info') ;
  net = vl_simplenn_tidy(net) ; % just in case MatConvNet was updated
end

res = [] ;

for epoch = start+1 : opts.numEpochs

  % Call the epoch function on the network
  if ~isempty(opts.epochFunction)
    net = opts.epochFunction(epoch, net);
  end
  
  train = opts.train(randperm(numel(opts.train))) ;

  info.train.objective(end+1) = 0 ;
  info.train.error(end+1) = 0 ;
  info.train.topFiveError(end+1) = 0 ;
  info.train.speed(end+1) = 0 ;
  info.val.objective(end+1) = 0 ;
  info.val.error(end+1) = 0 ;
  info.val.topFiveError(end+1) = 0 ;
  info.val.speed(end+1) = 0 ;

  future = [];
  
  for t=1:opts.batchSize:numel(train)
    % get next image batch and labels
    batch = train(t:min(t+opts.batchSize-1, numel(train))) ;
    batch_size = numel(batch);
    batch_time = tic ;
    fprintf('tr: ep %02d: bat %3d of %3d ...', epoch, ...
            fix(t/opts.batchSize)+1, ceil(numel(train)/opts.batchSize)) ;
    
    if opts.prefetch
      if isempty(future)
        [im, meta] = getBatch(imdb, batch, true);
      else
        [im, meta] = fetchOutputs(future);
      end
      nt = t + opts.batchSize;
      next_batch = train(nt:min(nt+opts.batchSize-1, numel(train)));
      if isempty(next_batch)
        future = [];
      else
        future = parfeval(gcp(), getBatch, 2, imdb, next_batch, true);
      end
    else
      [im, meta] = getBatch(imdb, batch, true);
    end
    
    if opts.useGpu
      im = gpuArray(im) ;
    end

    % backprop
    net.layers{end}.gt_coords = meta.gt_coords;
    net.layers{end}.org_gt_coords = meta.org_gt_coords;
    net.layers{end}.margins = meta.margins;
    net.layers{end}.out_locs = meta.out_locs;
    net.layers{end-1}.out_locs = meta.out_locs;
    net.layers{end-2}.out_locs = meta.out_locs;
    net.layers{end}.im = im;
    net.layers{end}.im_scaled = meta.im_org_scaled;
    
    res = vl_simplenn(net, im, one, res, ...
      'conserveMemory', opts.conserveMemory, ...
      'sync', opts.sync) ;
    
%     figure(101); clf;
%     imagesc(res(end).aux.res_steps{3}.nw(:, :, 1, 1)); axis image;
%     pause(0.5);
    
    % gradient step
    [net, res] = accumulateGradients(net, res, opts, batch_size, epoch, ...
      epoch * numel(train) + t);

    % print information
    batch_time = toc(batch_time) ;
    speed = numel(batch)/batch_time ;
    info.train = updateError(opts, info.train, net, res, batch_time, ...
      meta.scale) ;

    if isnan(info.train.objective(end)) || isinf(info.train.objective(end)) ...
        || any(isnan(net.layers{1}.weights{1}(:)))
      keyboard;
    end
    
    fprintf(' %.2f s (%.1f im/s)', batch_time, speed) ;
    n = t + numel(batch) - 1 ;
    fprintf(' log10(obj) %.3f ', log10(info.train.objective(end)/n));

%     fprintf(' Loss %.4f, im_size:[%d %d]', res(end).x, size(im, 1), ...
%       size(im, 2));
%     fprintf(' Min %.4f, Max: %.4f', min(res(end).aux.res_steps{1}.w(:)), ...
%       max(res(end).aux.res_steps{1}.w(:)));
%     fprintf(' Mig %.4f, Mag: %.4f', min(res(end-2).dzdw{1}(:)), ...
%       max(res(end-2).dzdw{1}(:)));
%     fprintf(' Mig %.4f, Mag: %.4f', min(res(end-1).dzdw{1}(:)), ...
%       max(res(end-1).dzdw{1}(:)));
%     fprintf(' Mig %.4f, Mag: %.4f', min(res(end-1).dzdw{2}(:)), ...
%       max(res(end-1).dzdw{2}(:)));
%     fprintf(' Mig %.4f, Mag: %.4f', min(res(end-1).dzdx(:)), ...
%       max(res(end-1).dzdx(:)));
    fprintf('\n') ;

    % debug info
    if opts.plotDiagnostics
      figure(2) ; vl_simplenn_diagnose(net,res) ; drawnow ;
      figure(3) ; my_simplenn_diagnose(net,res) ; drawnow ;
    end
  end % next batch

  % save
  info.train.objective(end) = info.train.objective(end) / numel(train) ;
  info.train.error(end) = info.train.error(end) / numel(train)  ;
  info.train.topFiveError(end) = info.train.topFiveError(end) / numel(train) ;
  info.train.speed(end) = numel(train) / info.train.speed(end) ;

  if rem(epoch, opts.checkpoint_interval) == 0
    save(modelPath(epoch), 'net', 'info') ;
  end
end

function [net, res] = accumulateGradients(net, res, opts, batch_size, epoch, ...
  update_count)
lr = opts.learningRate(min(epoch, numel(opts.learningRate)));

for l=1:numel(net.layers)
  for j=1:numel(res(l).dzdw)
    if j == 3 && strcmp(net.layers{l}.type, 'bnorm')
      % special case for learning bnorm moments
      thisLR = net.layers{l}.learningRate(j) ;
      net.layers{l}.weights{j} = ...
        (1-thisLR) * net.layers{l}.weights{j} + ...
        (thisLR/batch_size) * res(l).dzdw{j} ;
      continue;
    end
    
    if strcmpi(opts.acceleration, 'momentum')
      thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
      thisLR = lr * net.layers{l}.learningRate(j) ;

      grad = (1 / batch_size) * res(l).dzdw{j};
      if opts.do_gradient_scaling
        grad_scale = norm(grad(:));
        if grad_scale > opts.max_gradient_scale
          grad = grad * opts.max_gradient_scale / grad_scale;
        end
      elseif opts.do_gradient_clipping
        grad_clip_thresh = opts.grad_clip_thresh;
        grad(grad > grad_clip_thresh) = grad_clip_thresh;
        grad(grad < -grad_clip_thresh) = -grad_clip_thresh;
      end

      net.layers{l}.momentum{j} = ...
        opts.momentum * net.layers{l}.momentum{j} ...
        - thisDecay * net.layers{l}.weights{j} ...
        - grad ;
      net.layers{l}.weights{j} = net.layers{l}.weights{j} ...
        + thisLR * net.layers{l}.momentum{j} ;
    elseif strcmpi(opts.acceleration, 'nesterov')
      % Simplified Nesterov Momentum as in 'Advances in Optimizing Recurrent
      % Networks'.
      thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
      thisLR = lr * net.layers{l}.learningRate(j) ;
      momentum = opts.momentum;

      grad = (1 / batch_size) * res(l).dzdw{j} ...
        + thisDecay * net.layers{l}.weights{j};
      if opts.do_gradient_scaling
        grad_scale = norm(grad(:));
        if grad_scale > opts.max_gradient_scale
          grad = grad * opts.max_gradient_scale / grad_scale;
        end
      elseif opts.do_gradient_clipping
        grad_clip_thresh = opts.grad_clip_thresh;
        grad(grad > grad_clip_thresh) = grad_clip_thresh;
        grad(grad < -grad_clip_thresh) = -grad_clip_thresh;
      end

      grad = thisLR * grad;
      net.layers{l}.weights{j} = net.layers{l}.weights{j} ...
        + momentum^2 * net.layers{l}.momentum{j} ...
        - (1 + momentum) * grad;
      net.layers{l}.momentum{j} = momentum * net.layers{l}.momentum{j} ...
        - grad;
    elseif strcmpi(opts.acceleration, 'adam')
      if ~isfield(net.layers{l}, 'm_grad')
        net.layers{l}.m_grad = cell(size(net.layers{l}.weights));
        net.layers{l}.ms_grad = cell(size(net.layers{l}.weights));
      end

      epsilon = opts.adam_epsilon;
      beta_1 = opts.adam_beta1;
      beta_2 = opts.adam_beta2;

      % Initialize if not already
      if isempty(net.layers{l}.m_grad{j})
        net.layers{l}.m_grad{j} = 0 .* net.layers{l}.weights{j};
        net.layers{l}.ms_grad{j} = 0 .* net.layers{l}.weights{j};
      end

      % Update weights.
      thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j);
      thisLR = lr * net.layers{l}.learningRate(j);

      grad = (1 / batch_size) * res(l).dzdw{j} ...
        + thisDecay * net.layers{l}.weights{j};

      if opts.do_gradient_scaling
        grad_scale = norm(grad(:));
        if grad_scale > opts.max_gradient_scale
          grad = grad * opts.max_gradient_scale / grad_scale;
        end
      elseif opts.do_gradient_clipping
        grad_clip_thresh = opts.grad_clip_thresh;
        grad(grad > grad_clip_thresh) = grad_clip_thresh;
        grad(grad < -grad_clip_thresh) = -grad_clip_thresh;
      end

      m_grad = net.layers{l}.m_grad{j};
      m_grad = m_grad * beta_1 + (1 - beta_1) * grad;
      ms_grad = net.layers{l}.ms_grad{j};
      ms_grad = ms_grad * beta_2 + (1 - beta_2) * (grad .^2);

      ti = round(update_count / opts.batchSize);
      mh = m_grad ./ (1 - beta_1^ti);
      vh = ms_grad ./ (1 - beta_2^ti);
      net.layers{l}.weights{j} = net.layers{l}.weights{j} ...
        - thisLR * mh ./ (sqrt(vh) + epsilon);

      net.layers{l}.m_grad{j} = m_grad;
      net.layers{l}.ms_grad{j} = ms_grad;
    elseif strcmpi(opts.acceleration, 'rmspropmom')
      if ~isfield(net.layers{l}, 'ms_grad')
        net.layers{l}.ms_grad = cell(size(net.layers{l}.weights));
      end

      epsilon = opts.rmsprop_epsilon;
      rho = opts.rmsprop_rho;
      momentum = opts.momentum;

      % Initialize if not already
      if isempty(net.layers{l}.ms_grad{j})
        net.layers{l}.ms_grad{j} = double(0 .* net.layers{l}.weights{j});
      end

      thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
      thisLR = lr * net.layers{l}.learningRate(j) ;
      grad = double(res(l).dzdw{j}) ./ batch_size ...
        + thisDecay * double(net.layers{l}.weights{j});

      if opts.do_gradient_scaling
        grad_scale = norm(grad(:));
        if grad_scale > opts.max_gradient_scale
          grad = grad * opts.max_gradient_scale / grad_scale;
        end
      elseif opts.do_gradient_clipping
        grad_clip_thresh = opts.grad_clip_thresh;
        grad(grad > grad_clip_thresh) = grad_clip_thresh;
        grad(grad < -grad_clip_thresh) = -grad_clip_thresh;
      end

      ms_grad = net.layers{l}.ms_grad{j};
      ms_grad = ms_grad .* rho + (grad.^2) * (1 - rho);
      rms_grad = sqrt(ms_grad + epsilon);

      % This update is based on the 'Simplified Nesterov Momentum' suggested
      % in 'Advances in Optimizing Recurrent Networks'.
      grad = thisLR * grad ./ rms_grad;
      net.layers{l}.weights{j} = net.layers{l}.weights{j} ...
        + momentum^2 * net.layers{l}.momentum{j} ...
        - (1 + momentum) * grad;
      net.layers{l}.momentum{j} = momentum * net.layers{l}.momentum{j} ...
        - grad;
    elseif strcmpi(opts.acceleration, 'rmsprop')
      if ~isfield(net.layers{l}, 'ms_grad')
        net.layers{l}.ms_grad = cell(size(net.layers{l}.weights));
      end

      epsilon = opts.rmsprop_epsilon;
      rho = opts.rmsprop_rho;

      % Initialize if not already
      if isempty(net.layers{l}.ms_grad{j})
        net.layers{l}.ms_grad{j} = double(0 .* net.layers{l}.weights{j});
      end

      thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
      thisLR = lr * net.layers{l}.learningRate(j) ;
      grad = double(res(l).dzdw{j}) ./ batch_size ...
        + thisDecay * double(net.layers{l}.weights{j});

      if opts.do_gradient_scaling
        grad_scale = norm(grad(:));
        if grad_scale > opts.max_gradient_scale
          grad = grad * opts.max_gradient_scale / grad_scale;
        end
      elseif opts.do_gradient_clipping
        grad_clip_thresh = opts.grad_clip_thresh;
        grad(grad > grad_clip_thresh) = grad_clip_thresh;
        grad(grad < -grad_clip_thresh) = -grad_clip_thresh;
      end

      ms_grad = net.layers{l}.ms_grad{j};
      ms_grad = ms_grad .* rho + (grad.^2) * (1 - rho);
      rms_grad = sqrt(ms_grad + epsilon);

      rates = thisLR ./ rms_grad;
      net.layers{l}.weights{j} = net.layers{l}.weights{j} ...
        - rates .* grad;
    elseif strcmpi(opts.acceleration, 'adadelta')
      if ~isfield(net.layers{l}, 'ms_del')
        net.layers{l}.ms_del = cell(size(net.layers{l}.weights));
        net.layers{l}.ms_grad = cell(size(net.layers{l}.weights));
      end

      epsilon = opts.adadelta_epsilon;
      rho = opts.adadelta_rho;

      % Initialize if not already
      if isempty(net.layers{l}.ms_del{j})
        net.layers{l}.ms_del{j} = double(0 .* net.layers{l}.weights{j});
        net.layers{l}.ms_grad{j} = double(0 .* net.layers{l}.weights{j});
      end

      % Update weights.
      thisDecay = opts.weightDecay * net.layers{l}.weightDecay(j) ;
      grad = double(res(l).dzdw{j}) ./ batch_size ...
        + thisDecay * double(net.layers{l}.weights{j});

      if opts.do_gradient_scaling
        grad_scale = norm(grad(:));
        if grad_scale > opts.max_gradient_scale
          grad = grad * opts.max_gradient_scale / grad_scale;
        end
      elseif opts.do_gradient_clipping
        grad_clip_thresh = opts.grad_clip_thresh;
        grad(grad > grad_clip_thresh) = grad_clip_thresh;
        grad(grad < -grad_clip_thresh) = -grad_clip_thresh;
      end

      ms_grad = net.layers{l}.ms_grad{j};
      ms_grad = ms_grad .* rho + (grad.^2) * (1 - rho);
      rms_grad = sqrt(ms_grad + epsilon);
      m_grad = net.layers{l}.ms_del{j};
      rms_del = sqrt(m_grad + epsilon);

      rates = rms_del ./ rms_grad;
      pdel = rates .* grad;
      net.layers{l}.weights{j} = net.layers{l}.weights{j} - pdel;
      m_grad = m_grad .* rho + (pdel.^2) * (1 - rho);

      net.layers{l}.ms_del{j} = m_grad;
      net.layers{l}.ms_grad{j} = ms_grad;
    else
      error('Unsupported acceleration');
    end
  end
end


% -------------------------------------------------------------------------
function info = updateError(opts, info, net, res, speed, scale)
% -------------------------------------------------------------------------
predictions = gather(res(end-1).x) ;
sz = size(predictions) ;
n = prod(sz(1:2)) ;

info.objective(end) = info.objective(end) + sum(double(gather(res(end).x))) ;
info.speed(end) = info.speed(end) + speed ;
switch opts.errorType
  case 'multiclass'
    labels = net.layers{end}.class ;
    [~,predictions] = sort(predictions, 3, 'descend') ;
    error = ~bsxfun(@eq, predictions, reshape(labels, 1, 1, 1, [])) ;
    info.error(end) = info.error(end) +....
      sum(sum(sum(error(:,:,1,:))))/n ;
    info.topFiveError(end) = info.topFiveError(end) + ...
      sum(sum(sum(min(error(:,:,1:5,:),[],3))))/n ;
  case 'binary'
    labels = net.layers{end}.class ;
    error = bsxfun(@times, predictions, labels) < 0 ;
    info.error(end) = info.error(end) + sum(error(:))/n ;
  case 'location'
    gt_loc = net.layers{end}.org_gt_coords;
    out_locs = net.layers{end}.out_locs;
    
    [score, pred_loc] = max(predictions(:));
    
%     gt_ind = sub2ind(sz(1:2), gt_coords(2), gt_coords(1));
%     gt_loc = out_locs(gt_ind, :);
    pd_loc = out_locs(pred_loc, :);
    info.error(end) = info.error(end) ...
      + sum(pdist2(gt_loc, pd_loc, 'euclidean')) / scale;
end

% -------------------------------------------------------------------------
function epoch = findLastCheckpoint(modelDir)
% -------------------------------------------------------------------------
list = dir(fullfile(modelDir, 'net-epoch-*.mat')) ;
tokens = regexp({list.name}, 'net-epoch-([\d]+).mat', 'tokens') ;
epoch = cellfun(@(x) sscanf(x{1}{1}, '%d'), tokens) ;
epoch = max([epoch 0]) ;


