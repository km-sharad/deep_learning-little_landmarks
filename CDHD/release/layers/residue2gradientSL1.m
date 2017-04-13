function grad = residue2gradientSL1(residue, transition_dist)
% This function computes the gradient for the smooth L1 loss.
%
% residue: y - y^
% transition_distance: L1 distance below which the loss is L2.
%
% Author: saurabh.me@gmail.com (Saurabh Singh).

grad = residue;
s = abs(residue) >= transition_dist;
grad(s) = 1 * (residue(s) > 0) - 1 * (residue(s) < 0);
grad(~s) = grad(~s) ./ transition_dist;
end
