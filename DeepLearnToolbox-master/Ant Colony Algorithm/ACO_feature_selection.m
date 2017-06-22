clc;
clear;
mydata=importdata('data.txt');
RHO=corr(mydata);
R=abs(RHO);
n=30; % The number of features without class 无类特征的数量
m=3; % The number of selected features  被选中的特征
CrossOverProb=0.9; % The Cross Over Probability 交叉概率
MutationProb=0.04;  % The Mutation Probability 突变概率
Population = 500;   % The Size of popuation 人口规模 
GenerationNo = 300; % The Number of generation 多少代
%[N, m, TOWNS] = setup16;
%trail = N.*ones(N);
na = 200;
cc = 1;
for i=1:n
    trail(i) = cc;
    deltatrail(i) = 0;
end;    
maxnoiteration = 20;  %最大限度
k = 50;
p = 1;
rho = 0.75;
alpha=1;
beta=1;

counter=1;   % The First iteration 
for j=1:na
     Select = randperm(n);
     s(j,1:m)= Select(1:m);
end    

data = dataset(mydata(:,1:n),mydata(:,n+1));  %
[Train,Test]=gendat(data,0.6);



for j=1:na
    A = [];
    B = [];
    for i=1:m
        A = [A,Train(:,s(j,i))];
        B = [B,Test(:,s(j,i))];
    end    
        
    W=knnc(A);
    D=B*W;
    E(j)=D*testc;
 %   scatterd(A);
 %   plotc({W});
end
        
[MinError,MinErrorIndex]=min(E);
[SO,IX]=sort(E);

for j=1:k
   for l=1:m
      for i=1:n
         deltatrail(i)=0;
         if (s(IX(j),l)==i)
            Errorg = max(SO(1:k))-E(IX(j));
            Errorhg=0;
            for h=1:k
              if ( (max(SO(1:k))-E(IX(h))) > Errorhg ) 
                Errorhg = max(SO(1:k))-E(IX(h));
                end
            end    
            if (Errorhg ~= 0)
              deltatrail(i)=Errorg/Errorhg; 
            else
              deltatrail(i)=1;
            end;  
         end;   
         trail(i)= rho * trail(i) + deltatrail(i);
      end;  
   end;  
end;  


%for j=1:na
%    temp1(1:m)= s(IX(round(rand*(k-1))+1),1:m);
%    r=randperm(m);
%    for l=1:m-p
%      temp2(l)=temp1(r(l));
%    end;        
%    s(j,1:m-p)=temp2(1:m-p);
%end;    
    


%tourlength = [];
%temp_length = [];
%rho = 0.5; % rho = 1 - (evaporation of trail)
%Q = N;
%avgtour = [];
%mintour = [];

while ( (counter < maxnoiteration) & (MinError > 0.001) )

counter = counter+1;
%for exit_tours = 1:40 %Loop back to here
%tabu = zeros(m,N); %generate (N x m) matrix containing zeros
%set initial town for each ant randomly
%for i = 1:m
%tabu(i,1) = ceil(N*rand);
%end
%Use function nexttown.m to determine next town based on visibility and
%trail
%for j = 1:length(TOWNS(1,:))-1 %for each town
%for i = 1:m %for each ant
%sub_tabu = tabu(i,:); %create tabu for each ant
%add new town to tabu, using function nexttown.m
%tabu(i,j+1) = nexttown(sub_tabu, TOWNS, trail);
%end
%end


for j=1:na
    temp1(1:m)= s(IX(round(rand*(k-1))+1),1:m);
    r=randperm(m);
    for l=1:m-p
      temp2(l)=temp1(r(l));
    end;        
    s(j,1:m-p)=temp2(1:m-p);
end;    


for mm = m-p+1:m
  for j=1:na
   for i=1:n
     flag=0;
     for l=1:m-p
        if (s(j,l)==i)
          flag=1;
        end;
     end;   
     if (flag==1)
       USM(i)=0;
     else
       den=0;
       for l=1:m-p
         den=den+R(i,s(j,l));
       end  
       if (den~=0)
           LI(i)=R(i,n+1)/den;
       else
           LI(i)=R(i,n+1);
       end;    
       vis(i)=LI(i)^beta;
       trail_p(i)=trail(i)^alpha;
       sigma=0;
       for ii=1:n
         flag=0;
         for l=1:m-p
           if(s(j,l)==ii)
             flag=1;
           end;
         end;  
         if (flag==0)
           den=0;
           for l=1:m-p
             den=den+R(ii,s(j,l));
           end;  
           if (den~=0)
              LI(ii)=R(ii,n+1)/den;
           else
              LI(ii)=R(ii,n+1);
           end   
           sigma=sigma+ (trail(ii)^alpha) * (LI(ii)^beta);
         end;  
       end;  
       USM(i)=(vis(i)*trail_p(i))/sigma;  
       end  
   end  
   [maxf,maxfindex]=max(USM);
   s(j,mm)=maxfindex;
  end;  
end;       


for j=1:na
    flag=0;
    for k1=1:m
        for k2=k1+1:m
            if(s(j,k1)==s(j,k2))
               flag=1;
            end;
        end;
    end;    
    if (flag==1)
         Select=randperm(n);
         s(j,1:m)=Select(1:m);
     end;    
end;    
        


for i=1:na
  for j=i+1:na 
    if (s(i,1:m)==s(j,1:m))
      Select = randperm(n);
       s(j,1:m)= Select(1:m);
    end;
  end;
end;  
    
for j=1:na
    A = [];
    B = [];
    for i=1:m
        A = [A,Train(:,s(j,i))];
        B = [B,Test(:,s(j,i))];
    end    
        
    W=knnc(A);
    D=B*W;
    E(j)=D*testc;
  %  scatterd(A);
  %  plotc({W});
end
        
[MinError,MinErrorIndex]=min(E);
[SO,IX]=sort(E);

for j=1:k
   for l=1:m
      for i=1:n
         deltatrail(i)=0;
         if (s(IX(j),l)==i)
            Errorg = max(SO(1:k))-E(IX(j));
            Errorhg=0;
            for h=1:k
              if ( (max(SO(1:k))-E(IX(h))) > Errorhg ) 
                Errorhg = max(SO(1:k))-E(IX(h));
                end
            end    
            if (Errorhg ~= 0)
              deltatrail(i)=Errorg/Errorhg; 
            else
              deltatrail(i)=1;
            end;  
         end;   
         trail(i)= rho * trail(i) + deltatrail(i);
      end;  
   end;  
end;  


%for j=1:na
%    temp1(1:m)= s(IX(round(rand*(k-1))+1),1:m);
%    r=randperm(m);
%    for l=1:m-p
%     temp2(l)=temp1(r(l));
%    end;        
%    s(j,1:m-p)=temp2(1:m-p);
%end;    

end;      
      
disp(s(MinErrorIndex,1:m));      

%%Compute and store tour length
%tabu = [tabu, tabu(:,1)]; %augment tabu to complete tour
%for k = 1:m %each ant
%Total distance between each town i,j
%temp_length(k,:) = sqrt((TOWNS(2,tabu(k,1:end - 1)) - ...
%TOWNS(2,tabu(k,2:end))).^2 + ...
%(TOWNS(3,tabu(k,1:end - 1)) - TOWNS(3,tabu(k,2:end))).^2);
%end
%Dynamically increasing vector of total tour length
%tourlength = [tourlength, (sum(temp_length'))'];
%%
%%Exit Condition
%%

%Update trail
%trail = 0.5*rho .* trail;
%for i = 1:m
%for n = 1:N
%for trail(i,j), increase trail by inverse of tour length of
%given ant
%trail(tabu(i,n),tabu(i,n+1)) = trail(tabu(i,n),tabu(i,n+1)) + ...
%Q/tourlength(i);
%end
%end
%trail = trail+trail'; %makes trail symmetric
%figure(1)
%plottown = plot (TOWNS(2,:), TOWNS(3,:), 'ro'); %plot graph of town
%layout
%axis ([0 2*N 0 2*N]);
%hold on
%maxtrail = max(max(trail));
%yheavy = [];
%xheavy = [];
%ymed = [];
%xmed = [];
%for i = 1:N-1
%for j = i:N
%if trail(i,j)>0.5*maxtrail
%yheavy(end+1,1) = TOWNS(3,i);
%yheavy(end,2) = TOWNS(3,j);
%xheavy(end+1,1) = TOWNS(2,i);
%xheavy(end,2) = TOWNS(2,j);
%elseif trail(i,j)>0.01*maxtrail
%ymed(end+1,1) = TOWNS(3,i);
%ymed(end,2) = TOWNS(3,j);
%xmed(end+1,1) = TOWNS(2,i);
%xmed(end,2) = TOWNS(2,j);
%end
%end
%end
%if length(xmed)>2
%for i = 1:length(xmed)
%medplot = plot (xmed(i,:),ymed(i,:),'c-');
%set(medplot,'LineWidth',1)
%end
%end
%set(medplot,'LineWidth',1)
%for i = 1:length(xheavy)
%heavyplot = plot (xheavy(i,:),yheavy(i,:),'b-');
%set(heavyplot,'LineWidth',2)
%end
%set(heavyplot,'LineWidth',3)
%drawnow
%hold off
%end %Loop back to tabu
%avgtour = mean(tourlength);
%mintour(1,:) = min(tourlength(:,:));
%avg_min = [avgtour' mintour'];
%figure(2)
%plotconverge = plot (avg_min); legend ('avg', 'min');