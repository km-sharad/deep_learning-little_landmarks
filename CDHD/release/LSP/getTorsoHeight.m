function torso = getTorsoHeight(flic, type)
% Author: saurabh.me@gmail.com (Saurabh Singh).
if strcmp(type, 'diagonal')
  top_parts = myLookupPart({'lsho'});
  bottom_parts = myLookupPart({'rhip'});
  
  s = flic.coords(:, top_parts);
  h = flic.coords(:, bottom_parts);
  
  torso = norm(s - h);
elseif strcmp(type, 'midpoint')
  top_parts = myLookupPart({'lsho', 'rsho'});
  bottom_parts = myLookupPart({'lhip', 'rhip'});
  
  top_parts = flic.coords(:, top_parts);
  bottom_parts = flic.coords(:, bottom_parts);
  
  s = mean(top_parts, 2);
  h = mean(bottom_parts, 2);
  torso = norm(s - h);
else
  error('Unsupported');
end
end
