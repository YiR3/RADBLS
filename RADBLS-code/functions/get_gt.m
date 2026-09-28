function GroundT = get_gt(class_gt)
[x,y] = size(class_gt);
data = reshape(class_gt,x*y,1);
c = max(max(class_gt));
index_list = [];
gt_list = [];
for i = 1:c
    index = find(class_gt == i);
    index_list = [index_list;index];
    gt_list = [gt_list;data(index)];
end
gt_list = double(gt_list);
GroundT = cat(2,index_list,gt_list);
end
    
    