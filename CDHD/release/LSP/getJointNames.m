function names = getJointNames()
keys = getJointNameKeys();
fnames = fieldnames(keys);
names = cell(size(fnames));
for i = 1 : length(fnames)
  names{keys.(fnames{i})} = fnames{i};
end
end
