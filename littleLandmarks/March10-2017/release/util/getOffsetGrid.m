function offset_grid = getOffsetGrid(half_window, num_points)
xx = linspace(-half_window, half_window, num_points);
yy = xx;
[xx, yy] = meshgrid(xx, yy);
offset_grid = [xx(:) yy(:)];
end
