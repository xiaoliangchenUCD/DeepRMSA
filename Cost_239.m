function topology = Cost_239()%%%Europe_cost239
topology=ones(11)*inf;
for ii=1:11
    topology(ii,ii)=0;
end
%%%%%ii=1%%%%%%%%%%%%%%%%
topology(1,2)=450*2;
topology(1,3)=390*2;
topology(1,4)=550*2;
topology(1,8)=1310*2;
%%%%%ii=2%%%%%%%%%%%%%%%%
topology(2,3)=300*2;
topology(2,5)=400*2;
topology(2,6)=600*2;
topology(2,7)=820*2;
topology(2,9)=1090*2;
%%%%%ii=3%%%%%%%%%%%%%%%%
topology(3,4)=210*2;
topology(3,5)=220*2;
topology(3,7)=930*2;
%%%%%ii=4%%%%%%%%%%%%%%%%
topology(4,5)=390*2;
topology(4,8)=760*2;
topology(4,9)=660*2;
%%%%%ii=5%%%%%%%%%%%%%%%%
topology(5,6)=350*2;
topology(5,10)=730*2;
%%%%%ii=6%%%%%%%%%%%%%%%%
topology(6,7)=320*2;
topology(6,10)=565*2;
topology(6,11)=730*2;
%%%%%ii=7%%%%%%%%%%%%%%%%
topology(7,11)=820*2;
%%%%%ii=8%%%%%%%%%%%%%%%%
topology(8,9)=390*2;
topology(8,10)=740*2;
%%%%%ii=9%%%%%%%%%%%%%%%%
topology(9,10)=340*2;
topology(9,11)=660*2;
%%%%%ii=10%%%%%%%%%%%%%%%%
topology(10,11)=320*2;

for ii=1:10
    for jj=ii+1:11
      topology(jj,ii)=topology(ii,jj);
    end
end

end






















    