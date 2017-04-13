function [I, rec] = lrFlipCDHDDataRecord(I, rec)
% Author: saurabh.me@gmail.com (Saurabh Singh).

im_size = size(I);
im_size = im_size(1:2)';

if size(I, 3) == 1
  I = fliplr(I);
else
  I(:, :, 1) = fliplr(I(:, :, 1));
  I(:, :, 2) = fliplr(I(:, :, 2));
  I(:, :, 3) = fliplr(I(:, :, 3));
end

rec.coords(1, :) = bsxfun(@minus, im_size(2), rec.coords(1, :)) + 1;
rec.imgdims = size(I);

bbox = rec.bbox;
bbox([1 3]) = im_size(2) - bbox([3 1]) + 1;
% rec.bbox = [im_size'-bbox(1:2) im_size'-bbox(3:4)] + 1;
rec.bbox = bbox;
end
