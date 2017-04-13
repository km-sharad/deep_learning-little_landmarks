function patch = clipPatchBoxToBoundary(patchIn, imSize)
% Clip a patch to the image boundary
%
% Author: saurabh.me@gmail.com (Saurabh Singh).
patch = patchIn;
rows = imSize(1);
cols = imSize(2);
doWarn = false;

if patch(1) < 1
  patch(1) = 1;
  doWarn = true;
end
if patch(2) < 1
  patch(2) = 1;
  doWarn = true;
end
if patch(3) > cols
  patch(3) = cols;
  doWarn = true;
end
if patch(4) > rows
  patch(4) = rows;
  doWarn = true;
end

if doWarn
%   fprintf('WARNING: A patch was clipped. P(%d,%d)(%d,%d):I(%d,%d)\n', ...
%     patchIn.x1, patchIn.y1, patchIn.x2, patchIn.y2, rows, cols);
end

end
