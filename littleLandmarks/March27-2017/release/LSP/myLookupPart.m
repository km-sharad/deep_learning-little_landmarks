function [inds, numParts] = myLookupPart(varargin)
% Returns the part # in the annotation for a string representation

key = getJointNameKeys();
invkey = fields(key);
assert(key.(invkey{10})==10);


if nargout==2
  numParts = max(cellfun(@(x)key.(x), ...
    fieldnames(rmfield(key,'KEYPOINT_FLIPMAP'))));
end

if nargin == 0, 
    inds = key; 
    return; 
end

if nargin==1 && iscell(varargin{1})
    names = varargin{1};
else
    names = varargin;
end
inds = nan(length(names),1);
for i=1:length(names)
    if isfield(key,names{i});
        inds(i) = key.(names{i});
    end
end
end
