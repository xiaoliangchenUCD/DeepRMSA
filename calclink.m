function [path_link] = calclink(p)

global link

path_link=[];
for nn=1:(length(p)-1)
   a=p(nn);
   b=p(nn+1);
   path_link=[path_link link(a,b)];  
end