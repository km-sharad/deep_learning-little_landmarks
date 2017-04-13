function [nw, dwn, maxi, sew] = getNormalizedLocationWeightsFast(wt_transfer, w)
% Author: saurabh.me@gmail.com (Saurabh Singh).

if ~isstruct(wt_transfer)
  transfer.type = wt_transfer;
  transfer.margin = 10;
else
  transfer = wt_transfer;
end

maxi = 0;
sew = 0;
switch(transfer.type)
  case 'hinge'
    sum_w = sum(reshape(w, [], 1)) + 0.00001;
    if sum_w < eps
      fprintf('All weights are zero\n');
    end
    nw = w ./ sum_w;
    dwn = (w > 0) ./ sum_w;
  case 'softmax'
    a = max(max(w, [], 1), [], 2);
    ew = bsxfun(@(x, y) exp(x - y), w, a);
    sew = sum(sum(ew, 1), 2);
%     nw = exp(w - a - log(sew));
    nw = bsxfun(@rdivide, ew, sew);
    dwn = nw;
    sew = sew .* exp(a);
  case 'hingemax'
    [mw, maxi] = max(reshape(w, [], size(w, 4)), [], 1);
    slice_size = size(w, 1) * size(w, 2);
    maxi = reshape(((1 : size(w, 4)) - 1) * slice_size + maxi, 1, 1, 1, []);
    tw = max(0, transfer.margin + bsxfun(@minus, w, reshape(mw, 1, 1, 1, [])));
    sum_w = sum(sum(tw, 1), 2);
    nw = bsxfun(@rdivide, tw, sum_w);
    dwn = bsxfun(@rdivide, tw > 0, sum_w);
  otherwise
    error('Unsupported weight transfer function');
end
end
