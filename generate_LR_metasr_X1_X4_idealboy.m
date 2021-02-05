%%provided by idealboy, https://github.com/idealboy/Meta-SR-Pytorch/edit/master/prepare_dataset/generate_LR_metasr_X1_X4_mt.m
function generate_LR_metasr_X1_X4_idealboy()
%% settings
path_save = 'F:/Flickr2K/';
path_src = 'E:/SuperResolutionDataset/2K_Resolution/Flickr2K/Flickr2K_train_HR';

ext               =  {'*.png'};
filepaths           =  [];
for i = 1 : length(ext)
    filepaths = cat(1,filepaths, dir(fullfile(path_src, ext{i})));
end
nb_im = length(filepaths);

idx_start = 1400;
idx_end = nb_im;

DIV2K_HR = [];

for idx_im = idx_start:idx_end
    fprintf('Read HR :%d\n', idx_im);
    ImHR = imread(fullfile(path_src, filepaths(idx_im).name));
    DIV2K_HR{idx_im} = ImHR;
end

FolderLR_bicubic = fullfile(path_save,'Flickr2k_tarin_LR_bicubic')

if ~exist(FolderLR_bicubic)
    mkdir(FolderLR_bicubic)
end
    
%% generate and save LR via imresize() with Bicubic

scales = 1.0:1.0:4.0;
% scales = 1.0:0.05:8.0;

scale_num = length(scales);

%% look at the original source code, when applying parfor on outter loop, the large variable 'DIV2K_HR' wiil be
%% passed to each workers, and it will be very very slow just for start up the parfor.

%% we just exchange the original outter-loop and inner-loop,
%% so that, when applying parfor on current inner-loop('parfor i = 2:1:scale_num'),
%% it will not cause a heavy load to each workers when passing the shared variable,(image = ImHR;)
%% and, now, for each image, all the resized images (with diffrent scale) will be processed parallelly(speed up)

for IdxIm = idx_start:idx_end
    fprintf('IdxIm=%d\n', IdxIm);
    ImHR = DIV2K_HR{IdxIm};

    parfor i = 2:1:scale_num
        image = ImHR;
        scale = scales(i);
        FolderLR = fullfile(path_save,'LR_bicubic', sprintf('X%.2f',scale));
%         FolderLR = fullfile(path_save,'LR_bicubic', sprintf('X%d',scale));
        
        if ~exist(FolderLR, 'dir')
            mkdir(FolderLR)
        end

        [h,w,~]=size(image);
        while mod(h, scale) ~= 0
            h = h - 1;
        end
        while mod(w, scale) ~= 0
            w = w - 1;
        end
%         if mod(h,scale) ~= 0
%             h = h - mod(h, scale);
%         end
%         if mod(w,scale) ~= 0
%             w = w - mod(w, scale);
%         end
        image = image(1:h,1:w,:);

        image= imresize(image, [round(h/scale), round(w/scale)], 'bicubic');
        % name image
        fileName = filepaths(IdxIm).name
%         fileName = strcat(fileName, sprintf('x%d', scale));
        NameLR = fullfile(FolderLR, fileName);
        % save image
        imwrite(image, NameLR, 'png');
    end
end

