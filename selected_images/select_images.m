% select images contain a given number of classes.
clear;
clc;

% Load data
load('ADE20K_2016_07_26/index_ade20k.mat');

Nimage = 22210;
filenames20 = [];
filenames50 = [];
filenames100 = [];
filenames150 = [];
filenames300 = [];
filenames500 = [];


[~, objectindexes] = sort(index.objectcounts);
objectindexes20 = objectindexes(end - 20 + 1 : end);
objectindexes50 = objectindexes(end - 50 + 1 : end);
objectindexes100 = objectindexes(end - 100 + 1 : end);
objectindexes150 = objectindexes(end - 150 + 1 : end);
objectindexes300 = objectindexes(end - 300 + 1 : end);
objectindexes500 = objectindexes(end - 500 + 1 : end);

for n = 1 : Nimage
    filename = fullfile(index.folder{n}, index.filename{n});
%     filename
    fileseg = strrep(filename,'.jpg', '_seg.png');
    % Read object masks
    seg = imread(fileseg);
    M = size(seg, 1);
    N = size(seg, 2);
    R = seg(:,:,1);
    G = seg(:,:,2);
    ObjectClassMasks = (uint16(R)/10)*256+uint16(G);
    objects = unique(ObjectClassMasks(:));
    if objectsfilter(objects, objectindexes20)
        disp(['find 20, n = ', int2str(n), ', ', int2str(M), ' by ', int2str(N)]);
        filenames20 = [filenames20, filename];
        filenames50 = [filenames50, filename];
        filenames100 = [filenames100, filename];
        filenames150 = [filenames150, filename];
        filenames300 = [filenames300, filename];
        filenames500 = [filenames500, filename];
    elseif objectsfilter(objects, objectindexes50)
        disp(['find 50, n = ', int2str(n), ', ', int2str(M), ' by ', int2str(N)]);
        filenames50 = [filenames50, filename];
        filenames100 = [filenames100, filename];
        filenames150 = [filenames150, filename];
        filenames300 = [filenames300, filename];
        filenames500 = [filenames500, filename];
    elseif objectsfilter(objects, objectindexes100)
        disp(['find 100, n = ', int2str(n), ', ', int2str(M), ' by ', int2str(N)]);
        filenames100 = [filenames100, filename];
        filenames150 = [filenames150, filename];
        filenames300 = [filenames300, filename];
        filenames500 = [filenames500, filename];
    elseif objectsfilter(objects, objectindexes150)
        disp(['find 150, n = ', int2str(n), ', ', int2str(M), ' by ', int2str(N)]);
        filenames150 = [filenames150, filename];
        filenames300 = [filenames300, filename];
        filenames500 = [filenames500, filename];
    elseif objectsfilter(objects, objectindexes300)
        disp(['find 300, n = ', int2str(n), ', ', int2str(M), ' by ', int2str(N)]);
        filenames300 = [filenames300, filename];
        filenames500 = [filenames500, filename];
    elseif objectsfilter(objects, objectindexes500)
        disp(['find 500, n = ', int2str(n), ', ', int2str(M), ' by ', int2str(N)]);
        filenames500 = [filenames500, filename];
    end
end

%%
length(filenames20)
length(filenames50)
length(filenames100)
length(filenames150)
length(filenames300)
length(filenames500)
%%
filenames20 = strsplit(filenames20, '.jpg');
filenames50 = strsplit(filenames50, '.jpg');
filenames100 = strsplit(filenames100, '.jpg');
filenames150 = strsplit(filenames150, '.jpg');
filenames300 = strsplit(filenames300, '.jpg');
filenames500 = strsplit(filenames500, '.jpg');

%% save to txt
filenames = {filenames020, filenames050, filenames100, filenames150, filenames300, filenames500};
nums = [20, 50, 100, 150, 300, 500];
for i = 1 : length(nums)
    if nums(i) < 100
        numstr = ['0', int2str(nums(i))];
    else
        numstr = int2str(nums(i));
    end
%     % random indexes.
%     idx = randperm(length(filenames{i}));
%     % training set.
%     fid = fopen(['top', numstr, '_train.txt'],'w');
%     for j = 1 : length(filenames{i}) * 0.8
%         fprintf(fid, '%s\n', filenames{i}{idx(j)});
%     end
%     fclose(fid);
%     % validation set.
%     fid = fopen(['top', numstr, '_val.txt'],'w');
%     for k = j + 1 : length(filenames{i})
%         fprintf(fid, '%s\n', filenames{i}{idx(k)});
%     end
%     fclose(fid);
    % class index re-encoding
    fid = fopen(['top', numstr, '_class.txt'],'w');
    for idx = objectindexes(end - nums(i) + 1 : end)
        fprintf(fid, '%d\n', idx);
    end
end


%%


function passed = objectsfilter(objects, objectindexes)
    % return true if the file contain only objects belong to objectindexes.
    passed = true;
    for i = 1 : length(objects)
        if objects(i) > 0 && isempty(find(objectindexes == objects(i), 1))
            passed = false;
            break;
        end
    end
end





