%% Prepare the car door handle dataset.

data_file = [CONFIG.lspDataDir 'metadata_new_oc.mat'];
data = load(data_file, 'lsp_anno_oc');
lsp = data.lsp_anno_oc;

%% Load the metadata for the dataset.

fg_car = load([CONFIG.stanfordFGCarDataDir 'cars_annos.mat']);

%% Load up the annotations.

img_root = CONFIG.stanfordFGCarDataDir;
anno_dir = [CONFIG.procDir 'StanfordFGCarAnnotations/'];
done_categories = false(size(fg_car.class_names));
all_im_classes = [fg_car.annotations.class];
cdhd = cell(size(fg_car.annotations));
annotated = false(size(fg_car.annotations));
num_imgs = length(fg_car.annotations);
for i = 1 : num_imgs
  file_name = sprintf('%s%d.mat', anno_dir, i);
  if ~exist(file_name, 'file')
    continue;
  end
  anno = load(file_name);
  
  annotated(i) = true;
  
  im_info = imfinfo([img_root fg_car.annotations(i).relative_im_path]);
  
  % coords is a 2xN matrix containing 2D locations of N different keypoints.
  % For cars we just have one keypoint - the car door handle.
  cdhd{i}.coords = [anno.x; anno.y];
  cdhd{i}.filepath = fg_car.annotations(i).relative_im_path;
  cdhd{i}.imgdims = [im_info.Height im_info.Width 3];
  cdhd{i}.istrain = ~fg_car.annotations(i).test;
  cdhd{i}.istest = fg_car.annotations(i).test;
  cdhd{i}.is_visible = ~anno.hidden;
  cdhd{i}.class = fg_car.annotations(i).class;
  cdhd{i}.bbox = [ ...
    fg_car.annotations(i).bbox_x1 ...
    fg_car.annotations(i).bbox_y1 ...
    fg_car.annotations(i).bbox_x2 ...
    fg_car.annotations(i).bbox_y2 ...
    ];
%   keyboard;
end
fprintf('Finished\n');

annotated = find(annotated);
cdhd = cdhd(annotated);
cdhd = cat(2, cdhd{:});

%% Save the annotations.

out_file = [CONFIG.stanfordFGCarDataDir 'cdhd_anno.mat'];
save(out_file, 'cdhd', 'annotated');

