function params = getParamsWithDefaults(params, defaults)
% Add default values in the params if not present.
%
% Author: saurabh.me@gmail.com (Saurabh Singh).
if rem(length(defaults), 2) ~= 0
  error('defaults should be a cell array of name value pairs');
end

defaults = reshape(defaults, 1, []);
if isempty(params)
  params = cell2struct(defaults(2:2:end), defaults(1:2:end), 2);
end

num_params = length(defaults) / 2;
for i = 1 : num_params
  if ~isfield(params, defaults{2*i-1})
    params.(defaults{2*i-1}) = defaults{2*i};
  end
end
end
