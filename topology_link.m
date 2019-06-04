function [link,linksum]= topology_link(W,num_nodes)

link=zeros(num_nodes);
linksum=0;
for i=1:num_nodes
    for j=1:num_nodes
        if ((W(i,j)~=0)&(W(i,j)~=inf))
            linksum=linksum+1;
            link(i,j)=linksum;
        end
    end
end
end