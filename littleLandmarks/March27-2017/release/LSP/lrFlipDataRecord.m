function [I, rec] = lrFlipDataRecord(I, rec)
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
lrflip_order = [
  4 5 6 1 2 3 ... % arms
  10 11 12 7 8 9 ... % legs
  14 13 ... % eyes
  16 15 ... % ears
  17 18 19 20 21 ... % nose, mid pts of sho hip ear torso
  23 22 ... % mid uarm
  25 24 ... % mid larm
  27 26 ... % mid mid uleg
  29 28 ... % mid lleg
  30 31 ...
];
rec.coords = rec.coords(:, lrflip_order);
rec.is_visible = rec.is_visible(:, lrflip_order);
rec.imgdims = size(I);

if isfield(rec, 'torsobox')
  tb = rec.torsobox;
  rec.torsobox = [im_size'-tb(1:2) im_size'-tb(3:4)] + 1;
end
end
