function [I, targets, T] = rotateImageDataNew(I, targets, theta, do_fill)
if ~exist('do_fill', 'var')
  do_fill = true;
end

im_size = size(I);

T = getAffineTransform(theta);
tform = affine2d(T);

if ~do_fill
  I = imwarp(I, tform);
  [targets, T] = computeTransformCoordinates(targets, theta, im_size, size(I));
else
  if theta > 0
    theta_alt = theta;
  else
    theta_alt = -theta;
  end
  pad = size(I) * sind(theta_alt);
  I2 = imPad(I, pad([2 2 1 1]), 'replicate');
  I3 = imwarp(I2, tform);
  corners = [1 1; im_size(2) 1; 1 im_size(1); im_size(2) im_size(1)]';
  new_corners = bsxfun(@plus, corners, [pad(1); pad(2)]);
  new_targets = bsxfun(@plus, targets, [pad(1); pad(2)]);
  t_corners = computeTransformCoordinates(new_corners, theta, size(I2), size(I3));
  [t_targets, T] = computeTransformCoordinates(new_targets, theta, size(I2), ...
    size(I3));
  ii = round(min(t_corners, [], 2));
  ff = round(max(t_corners, [], 2));
  bbox = [ii' ff'];
  bbox = clipPatchBoxToBoundary(bbox, size(I3));
  I4 = I3(bbox(2):bbox(4), bbox(1):bbox(3), :);

  t_targets = bsxfun(@minus, t_targets, ii);
%   T(1, 3) = T(1, 3) - ii(1);
%   T(2, 3) = T(2, 3) - ii(2);
  T = [1 0 -ii(1); 0 1 -ii(2); 0 0 1] * T * [1 0 pad(1); 0 1 pad(2); 0 0 1];
  I = I4;
  targets = t_targets;
end
end

function [targets, T] = computeTransformCoordinates(targets, theta, ...
  init_im_size, final_im_size)

a = targets;
a(3, :) = 1;
ti = init_im_size([2 1])/2+0.5;
tf = final_im_size([2 1])/2+0.5;
T = getAffineTransformFull(-theta, -ti, tf);
b = T * a;
targets = b(1:2, :);
end

function T = getAffineTransform(theta)
T = [ ...
  cosd(theta) -sind(theta) 0; ...
  sind(theta) cosd(theta) 0;  ...
  0           0           1 ...
];
end

function T = getAffineTransformFull(theta, ti, tf)
T = [ ...
  cosd(theta) -sind(theta) ti(1)*cosd(theta)-ti(2)*sind(theta) + tf(1); ...
  sind(theta)  cosd(theta) ti(1)*sind(theta)+ti(2)*cosd(theta) + tf(2); ...
  0            0           1 ...
];
end

