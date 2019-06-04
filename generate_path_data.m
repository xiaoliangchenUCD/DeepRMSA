%%
%%

clear all
close all
clc

global link

Path = [];
k = 5;
topo = Cost_239;
num_nodes = size(topo,1);
[link,linksum]= topology_link(topo,num_nodes);

for ii = 1:num_nodes
    for jj = 1:num_nodes
        if ii ~= jj
            kPaths = kShortestPath(topo,ii,jj,k);
            path_ids = zeros(length(kPaths),num_nodes);
            link_ids = zeros(length(kPaths),num_nodes);
            path_lens = zeros(1,length(kPaths));
            for tt = 1:length(kPaths)
                path_id = cell2mat(kPaths(tt));
                link_id = calclink(path_id);
                path_ids(tt,1:length(path_id)) = path_id;
                link_ids(tt,1:length(link_id)) = link_id;
                path_lens(tt) = cal_len(path_id,topo);
            end
            Path(ii,jj).path_id = path_ids;
            Path(ii,jj).link_id = link_ids;
            Path(ii,jj).path_len = path_lens;
        end
    end
end

fid = fopen('Src_Dst_Paths_Cost.dat','w');
for ii = 1:num_nodes
    for jj = 1:num_nodes
        if ii == jj
            for tt = 1:k
                fwrite(fid,0,'int');% 路径节点个数
            end
        else
            for tt = 1:k
                if tt <= size(Path(ii,jj).path_id,1)
                    fwrite(fid,sum(Path(ii,jj).path_id(tt,:)~=0),'int');
                else
                    fwrite(fid,0,'int');
                end
            end
            for tt = 1:k
                if tt <= size(Path(ii,jj).path_id,1)
                    for qq = 1:sum(Path(ii,jj).path_id(tt,:)~=0)
                        fwrite(fid,Path(ii,jj).path_id(tt,qq),'int');
                    end
                end
            end
        end            
    end    
end
fclose(fid);