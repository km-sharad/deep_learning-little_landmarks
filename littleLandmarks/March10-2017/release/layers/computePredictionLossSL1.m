function [loss, residue] = computePredictionLossSL1(pred, target, ...
  transition_dist)
% This function computes the soft L1 loss between a prediction and ground truth.
%
% pred: prediction
% target: ground truth
% transition_distance: L1 distance below which the loss is L2.
%
% Author: saurabh.me@gmail.com (Saurabh Singh).

residue = bsxfun(@minus, pred, target);
dim_losses = abs(residue);
% In below the multiplication by transition_dist ensures that the loss is smooth
% at the transition i.e. the gradients match to the first order.
dim_losses(dim_losses < transition_dist) = ...
  (dim_losses(dim_losses < transition_dist).^2) / 2 / transition_dist;
dim_losses(dim_losses >= transition_dist) = ...
  dim_losses(dim_losses >= transition_dist) - transition_dist / 2;
loss = sum(sum(dim_losses, 2), 3);
end

