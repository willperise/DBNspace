clc;
clf;
clear;
 % X = ������������;
X = load('data.txt');
[N,n]=size(X);      % N =����������;n =����������������;
K = 4;              % K = ����; 
R = 100;            % R = ������; 
t_max = 1000;       % t_max =����������;           
% ��ʼ��
c = 10^-2;
tau = ones(N,K) * c;    %��Ϣ�ؾ��󣬳�ʼֵΪ0.01��N*K����������*��������
q = 0.9;                % ��ֵq
rho = 0.1;              % ������
best_solution_function_value = inf; % ���·������ֵ����ֵΪ����󣬸�ֵԽС����Ч��Խ�ã�
tic                     %�����������ʱ��
t = 1; 
%=======������ֹ����������������ֹ������ѡ��һ��======
% while ((t<=t_max))                             %�ﵽ��������������ֹ
% while ((best_solution_function_value>=19727))  %�ﵽһ���ľ���Ч������ֹ
while ((best_solution_function_value>=19727))    
%========================= 
    %·����ʶ�ַ�����ʶÿֻ���ϵ�·��
    solution_string = zeros(R,N+1);     
    for i = 1 : R                       %����Ϣ��Ϊ����ȷ�����ϵ�·��
        r = rand(1,N);    %�������ֵΪ0-1�������1*51������
        for g = 1 : N
            if r(g) < q     %���r(g)С����ֵ
                tau_max = max(tau(g,:));
                Cluster_number = find(tau(g,:)==tau_max);   %�����ʶ����ѡ����Ϣ������·��
                solution_string(i,g) = Cluster_number(1);   %ȷ����iֻ���϶Ե�g��������·����ʶ
            else            %���r(g)������ֵ,�����·����Ϣ��ռ������Ϣ�صı�����������ѡ��·��
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

    % �����������
    weight = zeros(N,K);
       for h = 1:N              %��·���������ʶ
           Cluster_index = solution_string(i,h); %����������          
           weight(h,Cluster_index) = 1;          %������ѡ�������weight�������Ӧλ�ñ�1
       end

       cluster_center = zeros(K,n);  %�������ģ�������K�����ģ�
       for j = 1:K
           for v = 1:n
               sum_wx = sum(weight(:,j).*X(:,v));   %��������������ֵ֮��
               sum_w = sum(weight(:,j));            %������������
               if sum_w==0                          %����������Ϊ0�������ľ�������Ϊ0
                 cluster_center(j,v) =0
                  continue;
               else                                 %������������Ϊ0����������ĵ�ֵȡ��������ֵ��ƽ��ֵ
               cluster_center(j,v) = sum_wx/sum_w;
               end
            end
       end

    % ���������������Ե����Ӧ�ľ������ĵľ�����֮�ͣ���ֵ����solution_string�����һλ
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
    %����Fֵ����solution_string������������
    [fitness_ascend,solution_index] = sort(solution_string(:,end),1);
    solution_ascend = [solution_string(solution_index,1:end-1) fitness_ascend];

    pls = 0.1;        %�ֲ�Ѱ����ֵpls���൱�ڱ����ʣ�
    L = 2;              % ��L��·���ھֲ�Ѱ��
    % �ֲ�Ѱ�ų���
    solution_temp = zeros(L,N+1);
    k = 1;
    while(k <= L)
           solution_temp(k,:) = solution_ascend(k,:);
           rp = rand(1,N);     %����һ��1*N(51)ά��������飬ĳֵС��pls������ı����Ӧ��·����ʶ
           for i = 1:N
               if rp(i) <= pls
                   current_cluster_number = setdiff([1:K],solution_temp(k,i));
                   rrr=randint(1,1,[1,K-1]);
                   change_cluster = current_cluster_number(rrr);
                   solution_temp(k,i) = change_cluster;
               end
           end

        % ������ʱ��������   
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
          % ���������������Ե����Ӧ����ʱ�������ĵľ�����֮��Ft��
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
        %������ʱ�����������·��
        % ��� Ft<Fl �� Fl=Ft �� Sl=St
          if solution_temp(k,end) <= solution_ascend(k,end)              
              solution_ascend(k,:) = solution_temp(k,:);               
          end

          if solution_ascend(k,end)<=best_solution_function_value
              best_solution = solution_ascend(k,:);
          end
      k = k+1;
      end   

    % ����õ�L��·��������Ϣ������
    tau_F = 0;
    for j = 1:L    
       tau_F = tau_F + solution_ascend(j,end);
    end
    for i = 1 : N        
       tau(i,best_solution(1,i)) = (1 - rho) * tau(i,best_solution(1,i)) + 1/ tau_F; 
    %1/tau_F��rho/tau_FЧ�����ܺ�
    end 
    t=t+1
    best_solution_function_value =  solution_ascend(1,end);
    best_solution_function_value
end
time=toc;       %�����������ʱ��
clc
t 
time
cluster_center
best_solution = solution_ascend(1,1:end-1);
IDY=ctranspose(best_solution)
best_solution_function_value =  solution_ascend(1,end)
%��������ʾ
plot3(cluster_center(:,1),cluster_center(:,2),cluster_center(:,3),'o');grid;box
title('��Ⱥ������(R=100,t=1000)')
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