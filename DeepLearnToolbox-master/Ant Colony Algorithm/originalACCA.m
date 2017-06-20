clc;
clf;
clear;
tic
 % X = 测试样本矩阵;
%  X = load('data.txt');
X=[
2232.43	3077.87	1298.87;
1580.1	1752.07	2463.04;
1962.4	1594.97	1835.95;
1495.18	1957.44	3498.02;
1125.17	1594.39	2937.73;
24.22	3447.31	2145.01;
1269.07	1910.72	2701.97;
1802.07	1725.81	1966.35;
1817.36	1927.4	2328.79;
1860.45	1782.88	1875.13;
1237.91	2055.13	3405.09;
688.94	2104.72	3198.51;
1675.65	1747.23	1580.39;
1806.02	1810.19	2191.12;
74.56	3288.02	2433.87;
307.35	3363.84	2021.61;
1988.27	1657.51	2069.2;
2173.92	2608.55	1803.57;
372.16	3077.44	2163.46;
576.6	2140.98	3320;
1724.13	1704.49	1798.75;
2501.21	2652.65	984.56;
1656.94	1913.34	2459.07;
362.51	3150.03	2472;
565.74	2284.97	3024.58;
1978.06	1536.13	2375.64;
1661.06	1552.4	2005.05;
790.29	2419.98	3051.16;
1557.27	1746.27	1879.13;
2793.36	3009.26	1073.55;
1766.08	1803.14	1895.18;
1207.88	1600.62	3123.07;
245.75	3373.67	2248.45;
2785.36	3052.81	1035.65;
315.42	3088.29	2187.12;
1243.28	2451.72	3111.99;
829.84	1555.91	3139.21;
1347.07	2364.31	3096.88;
1926.98	1507.34	1626.47;
1808.57	1608.78	1565.95;
1124.1	1840.98	2819.41;
2661	3302.39	1710.32;
1805.55	1899.09	2400.6;
1130.18	1902.42	2753.7;
1355.19	1566.16	2927.81;
1651.14	1774.03	1725.56;
2110.63	3308.04	702.06;
2788.11	3395.23	1684.45;
1807.61	1680.56	2356.65;
1363.58	1729.44	2749.55;
1992.42	1526.9	1581.42;     
]
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
tic
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
   for k=1:R     
              if solution_ascend(k,end)<=best_solution_function_value
              best_solution = solution_ascend(k,:);
              end
      k = k+1;
   end   

    % 用最好的L条路径更新信息数矩阵
    tau_F = 0;
    L=2;
    for j = 1:L    
       tau_F = tau_F + solution_ascend(j,end);
    end
    for i = 1 : N        
       tau(i,best_solution(1,i)) = (1 - rho) * tau(i,best_solution(1,i)) + 1/ tau_F; 
    %1/tau_F和rho/tau_F效果都很好
    end 
    t=t+1
     best_solution_function_value =  solution_ascend(1,end)   
end
time=toc;
clc
t 
time
cluster_center
best_solution = solution_ascend(1,1:end-1);
IDY=ctranspose(best_solution)
best_solution_function_value =  solution_ascend(1,end)
%分类结果显示
plot3(cluster_center(:,1),cluster_center(:,2),cluster_center(:,3),'o');grid;box
title('蚁群聚类结果(R=100,t=10000)')
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
toc