clear;
clc;
close all;
rng(13)
addpath('.\functions');

load('Indian_pines_gt.mat');
load('Indian_pines_corrected.mat');

GroundT = get_gt(indian_pines_gt);

specific_index = [2,3,5,6,8,10,11,12,14,15];
selected_indices = ismember(GroundT(:,2),specific_index);
GroundT = GroundT(selected_indices,:);

for i = 1:length(specific_index)
    GroundT(GroundT(:,2) == specific_index(i),2) = i;
end

c = max(GroundT(:,2));

numClass = zeros(c,1);
for i = 1:c
    numClass(i) = sum(GroundT(:,2) == i);
end

train_num = 150;
train_num_list = ones(c,1) * train_num;

small_class_idx = find(numClass < train_num);
train_num_list(small_class_idx) = ceil(numClass(small_class_idx) * 0.5);

img = double(indian_pines_corrected);
[no_lines,no_rows,~] = size(img);

GroundT = GroundT';

alpha   = 1e-2;
beta    = 1e-5;
gamma   = 1e-2;
lambda  = 1e-1;

N1 = 10;
Ng = 10;
N2 = 50;

s = 1;
epsilon = 0.1;

img2 = average_fusion(img,15);
no_bands = size(img2,3);

fimg = reshape(img2,[no_lines*no_rows,no_bands]);
fimg = scale_new(fimg);
fimg = reshape(fimg,[no_lines,no_rows,no_bands]);

fimg1 = spatial_feature(fimg,115,0.6);
fimg2 = spatial_feature(fimg,200,0.9);
fimg3 = spatial_feature(fimg,30,0.3);

f_fimg = cat(3,fimg1,fimg2,fimg3);

fimg = PCA_img(f_fimg,30);

fimg = ToVector(fimg);
fimg = double(fimg');

indexes = [];

for i = 1:c
    current_train_num = train_num_list(i);

    index = find(GroundT(2,:) == i);
    randindex = index(randperm(length(index)));

    index_train = randindex(1:current_train_num);
    indexes = [indexes,index_train];
end

train_SL = GroundT(:,indexes);
train_data = fimg(:,train_SL(1,:));
train_label = train_SL(2,:)';

test_SL = GroundT;
test_SL(:,indexes) = [];

test_data = fimg(:,test_SL(1,:));
test_label = test_SL(2,:)';

H_train = zeros(c,size(train_data,2));

for i = 1:size(train_data,2)
    H_train(train_label(i),i) = 1;
end

class_count = zeros(1,c);

for i = 1:c
    class_count(i) = sum(train_label == i);
end

Train_num = [0,class_count];

[A,A_test] = bls_train(train_data',test_data',s,N1,Ng,N2);

Y = H_train';

[W,M,T,loss] = radbls(A,c,Y,Train_num,gamma,alpha,beta,lambda,epsilon);

T_train = A * W;
T_test = A_test * W;

train_norm = sqrt(sum(T_train .* T_train,2));
train_norm(train_norm == 0) = eps;
T_train = T_train ./ train_norm;

mdl = fitcknn(T_train,train_label);

test_norm = sqrt(sum(T_test .* T_test,2));
test_norm(test_norm == 0) = eps;
T_test = T_test ./ test_norm;

class_test = predict(mdl,T_test);

OA = sum(test_label == class_test) / length(test_label) * 100;

fprintf('\nOA = %.4f %%\n',OA);