clc;
clf;
clear;
 % X = 测试样本矩阵;
X = load('data.txt');
[N,n]=size(X);      % N =测试样本数;n =测试样本的属性数;
K = 4;              % K = 组数; 
R = 100;            % R = 蚂蚁数; 
t_max = 1000;       % t_max =最大迭代次数;           
% 初始化
c = 10^-2;
tau = ones(N,K) * c;    %信息素矩阵，初始值为0.01的N*K矩阵（样本数*聚类数）
q = 0.9;                % 阈值q
rho = 0.1;              % 蒸发率
best_solution_function_value = inf; % 最佳路径度量值（初值为无穷大，该值越小聚类效果越好）
tic                     %计算程序运行时间
t = 1; 
%=======程序终止条件（下列两个终止条件任选其一）======
% while ((t<=t_max))                             %达到最大迭代次数而终止
% while ((best_solution_function_value>=19727))  %达到一定的聚类效果而终止
while ((best_solution_function_value>=19727))    
%========================= 
    %路径标识字符：标识每只蚂蚁的路径
    solution_string = zeros(R,N+1);     
    for i = 1 : R                       %以信息素为依据确定蚂蚁的路径
        r = rand(1,N);    %随机产生值为0-1随机数的1*51的数组
        for g = 1 : N
            if r(g) < q     %如果r(g)小于阈值
                tau_max = max(tau(g,:));
                Cluster_number = find(tau(g,:)==tau_max);   %聚类标识数，选择信息素最多的路径
                solution_string(i,g) = Cluster_number(1);   %确定第i只蚂蚁对第g个样本的路径标识
            else            %如果r(g)大于阈值,求出各路径信息素占在总信息素的比例，按概率选择路径
                sum_p = sum(tau(g,:)); 
                p = tau(g,:) / sum_p; 
                for u = 2 : K 
                    p(u) = p(u) + p(u-1); 
                end
               rr = rand;          
                for s = 1 : K 
                    if (rr <= p(s)) 
                       Cluster_number = s;
                       solution_string(i,g) = Cluster_number;  
                    break; 
                end 
            end
        end
    end

    % 计算聚类中心
    weight = zeros(N,K);
       for h = 1:N              %给路径做计算标识
           Cluster_index = solution_string(i,h); %类的索引编号          
           weight(h,Cluster_index) = 1;          %对样本选择的类在weight数组的相应位置标1
       end

       cluster_center = zeros(K,n);  %聚类中心（聚类数K个中心）
       for j = 1:K
           for v = 1:n
               sum_wx = sum(weight(:,j).*X(:,v));   %各类样本各属性值之和
               sum_w = sum(weight(:,j));            %各类样本个数
               if sum_w==0                          %该类样本数为0，则该类的聚类中心为0
                 cluster_center(j,v) =0
                  continue;
               else                                 %该类样本数不为0，则聚类中心的值取样本属性值的平均值
               cluster_center(j,v) = sum_wx/sum_w;
               end
            end
       end

    % 计算各样本点各属性到其对应的聚类中心的均方差之和，该值存入solution_string的最后一位
      F = 0;
      for j= 1:K
          for ii = 1:N
              Temp=0;
              if solution_string(i,ii)==j;                
                  for v = 1:n
                      Temp = ((abs(X(ii,v)-cluster_center(j,v))).^2)+Temp;
                  end
                  Temp = sqrt(Temp);
              end
            F = (Temp)+F;
          end        
      end

       solution_string(i,end) = F;                      

    end 
    %根据F值，把solution_string矩阵升序排序
    [fitness_ascend,solution_index] = sort(solution_string(:,end),1);
    solution_ascend = [solution_string(solution_index,1:end-1) fitness_ascend];

    pls = 0.1;        %局部寻优阈值pls（相当于变异率）
    L = 2;              % 在L条路径内局部寻优
    % 局部寻优程序
    solution_temp = zeros(L,N+1);
    k = 1;
    while(k <= L)
           solution_temp(k,:) = solution_ascend(k,:);
           rp = rand(1,N);     %产生一个1*N(51)维的随机数组，某值小于pls则随机改变其对应的路径标识
           for i = 1:N
               if rp(i) <= pls
                   current_cluster_number = setdiff([1:K],solution_temp(k,i));
                   rrr=randint(1,1,[1,K-1]);
                   change_cluster = current_cluster_number(rrr);
                   solution_temp(k,i) = change_cluster;
               end
           end

        % 计算临时聚类中心   
           solution_temp_weight = zeros(N,K);
           for h = 1:N
               solution_temp_cluster_index = solution_temp(k,h);           
               solution_temp_weight(h,solution_temp_cluster_index) = 1;
           end

           solution_temp_cluster_center = zeros(K,n);
           for j = 1:K
               for v = 1:n
                   solution_temp_sum_wx = sum(solution_temp_weight(:,j).*X(:,v));
                   solution_temp_sum_w = sum(solution_temp_weight(:,j));
                   if solution_temp_sum_w==0
                   solution_temp_cluster_center(j,v) =0;
                   continue;
                   else
                       solution_temp_cluster_center(j,v) = solution_temp_sum_wx/solution_temp_sum_w;
                   end
               end
          end
          % 计算各样本点各属性到其对应的临时聚类中心的均方差之和Ft；
          solution_temp_F = 0;
          for j= 1:K
              for ii = 1:N
                  st_Temp=0;
                  if solution_temp(k,ii)==j;                               
                      for v = 1:n
                          st_Temp = ((abs(X(ii,v)-solution_temp_cluster_center(j,v))).^2)+st_Temp;
                      end
                      st_Temp = sqrt(st_Temp);
                  end
                  solution_temp_F = (st_Temp)+solution_temp_F;
              end
          end
        solution_temp(k,end) = solution_temp_F;   
        %根据临时聚类度量调整路径
        % 如果 Ft<Fl 则 Fl=Ft ， Sl=St
          if solution_temp(k,end) <= solution_ascend(k,end)              
              solution_ascend(k,:) = solution_temp(k,:);               
          end

          if solution_ascend(k,end)<=best_solution_function_value
              best_solution = solution_ascend(k,:);
          end
      k = k+1;
      end   

    % 用最好的L条路径更新信息数矩阵
    tau_F = 0;
    for j = 1:L    
       tau_F = tau_F + solution_ascend(j,end);
    end
    for i = 1 : N        
       tau(i,best_solution(1,i)) = (1 - rho) * tau(i,best_solution(1,i)) + 1/ tau_F; 
    %1/tau_F和rho/tau_F效果都很好
    end 
    t=t+1
    best_solution_function_value =  solution_ascend(1,end);
    best_solution_function_value
end
time=toc;       %输出程序运行时间
clc
t 
time
cluster_center
best_solution = solution_ascend(1,1:end-1);
IDY=ctranspose(best_solution)
best_solution_function_value =  solution_ascend(1,end)
%分类结果显示
plot3(cluster_center(:,1),cluster_center(:,2),cluster_center(:,3),'o');grid;box
title('蚁群聚类结果(R=100,t=1000)')
xlabel('X')
ylabel('Y')
zlabel('Z')
YY=[1 2 3 4];
index1 = find(YY(1) == best_solution)
index2 = find(YY(2) == best_solution)
index3 = find(YY(3) == best_solution)
index4 = find(YY(4) == best_solution)
line(X(index1,1),X(index1,2),X(index1,3),'linestyle','none','marker','*','color','g');
line(X(index2,1),X(index2,2),X(index2,3),'linestyle','none','marker','*','color','r');
line(X(index3,1),X(index3,2),X(index3,3),'linestyle','none','marker','+','color','b');
line(X(index4,1),X(index4,2),X(index4,3),'linestyle','none','marker','s','color','b');
rotate3d