function I = doEigenBasedDistortion(I, eigen_vectors, eigen_values, scale)
% Author: saurabh.me@gmail.com (Saurabh Singh).

offset = eigen_vectors * (eigen_values .* randn(size(eigen_values))) * scale;
I = bsxfun(@plus, I, reshape(offset, 1, 1, []));
end

