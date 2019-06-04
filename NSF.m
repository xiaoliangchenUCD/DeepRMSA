%% 2013-7-18: ½«NSFÍØÆË¾àÀëËõÐ¡Ò»°ë£¡
function topology = NSF
topology = [ 0 2100 3000  Inf  Inf  Inf  Inf 4800  Inf  Inf  Inf  Inf  Inf  Inf ;
			2100  0 1200 1500  Inf  Inf  Inf  Inf  Inf  Inf  Inf  Inf  Inf  Inf ;
			3000 1200  0  Inf  Inf 3600  Inf  Inf  Inf  Inf  Inf  Inf  Inf  Inf ;
			 Inf 1500  Inf  0 1200  Inf  Inf  Inf  Inf  Inf 3900  Inf  Inf  Inf ;
			 Inf  Inf  Inf 1200  0 2400 1200  Inf  Inf  Inf  Inf  Inf  Inf  Inf ;
			 Inf  Inf 3600  Inf 2400  0  Inf  Inf  Inf 2100  Inf  Inf  Inf 3600 ;
			 Inf  Inf  Inf  Inf 1200  Inf  0 1500  Inf 2700  Inf  Inf  Inf  Inf ;
			4800  Inf  Inf  Inf  Inf  Inf 1500  0 1500  Inf  Inf  Inf  Inf  Inf ;
			 Inf  Inf  Inf  Inf  Inf  Inf  Inf 1500  0 1500  Inf  600  600  Inf ;
			 Inf  Inf  Inf  Inf  Inf 2100 2700  Inf 1500  0  Inf  Inf  Inf  Inf ;
			 Inf  Inf  Inf 3900  Inf  Inf  Inf  Inf  Inf  Inf  0 1200 1500  Inf ;
			 Inf  Inf  Inf  Inf  Inf  Inf  Inf  Inf  600  Inf 1200 0  Inf  600 ;
			 Inf  Inf  Inf  Inf  Inf  Inf  Inf  Inf  600  Inf 1500  Inf  0  300 ;
			 Inf  Inf  Inf  Inf  Inf 3600  Inf  Inf  Inf  Inf  Inf  600  300  0];
topology = topology/2;
end