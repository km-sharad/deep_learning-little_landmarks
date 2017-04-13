function crops = getCroppingBounds(targets, im_size)
% Returns the border pixels on each side that could be cropped out without
% excluding the targets.
%
% Author: saurabh.me@gmail.com (Saurabh Singh).
min_xy = floor(min(targets, [], 2));
max_xy = ceil(max(targets, [], 2));

crops = [max(0, min_xy' - 1) max(0, im_size([2 1]) - max_xy')]; % l t r b
end
