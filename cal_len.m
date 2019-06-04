function path_length = cal_len(path,topo)

path_length = 0;
for ll = 1:length(path)-1
    path_length = path_length + topo(path(ll),path(ll+1));
end