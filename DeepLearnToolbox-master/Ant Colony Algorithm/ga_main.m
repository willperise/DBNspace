function ga_main()
% �Ŵ��㷨����
% n-- ��Ⱥ��ģ% ger-- ��������% pc--- �������% pm-- �������
% v-- ��ʼ��Ⱥ����ģΪn��% f-- Ŀ�꺯��ֵ% fit-- ��Ӧ������
% vx-- ������Ӧ��ֵ����% vmfit-- ƽ����Ӧ��ֵ����
clear all;
close all;
clc;%����
tic;%��ʱ����ʼ��ʱ
n=20;ger=100;pc=0.65;pm=0.05;%��ʼ������
%����Ϊ����ֵ�����Ը��ġ�
% ���ɳ�ʼ��Ⱥ
v=init_population(n,22); %�õ���ʼ��Ⱥ��22����������20*22��0-1����
[N,L]=size(v);           %�õ���ʼ��ģ�У���
disp(sprintf('Number of generations:%d',ger)); %���ٴ�
disp(sprintf('Population size:%d',N));
disp(sprintf('Crossover probability:%.3f',pc));
disp(sprintf('Mutation probability:%.3f',pm)); %sprintf���Կ��������ʽ
% ���Ż�����
xmin=0;xmax=9;  %����X��Χ
f='x+10*sin(x.*5)+7*cos(x.*4)';
% ������Ӧ�ȣ���������ʼ��Ⱥͼ��
x=decode(v(:,1:22),xmin,xmax);%22λ�����ƻ���ʮ���ƣ�%ð�ű�ʾ�������н��в�����
fit=eval(f);%evalת������ֵ�͵�  %������Ӧ��
figure(1);%�򿪵�һ������
fplot(f,[xmin,xmax]);%��������ͼ
grid on;hold on;
plot(x,fit,'k*');%��ͼ,����ʼ��Ⱥ����Ӧ��ͼ��
title('(a)Ⱦɫ��ĳ�ʼλ��');%����
xlabel('x');ylabel('f(x)');%�����
% ����ǰ�ĳ�ʼ��
vmfit=[];%ƽ����Ӧ��
vx=[]; %������Ӧ��
it=1; % ����������
% ��ʼ����
while it<=ger %�������� %100��
    %Reproduction(Bi-classist Selection)
    vtemp=roulette(v,fit);%��������  
    %Crossover    
    v=crossover(vtemp,pc);%�������� 
    %Mutation��������
    M=rand(N,L)<=pm;%����������ҵ���0.05С�ķ���
    %M(1,:)=zeros(1,L);
    v=v-2.*(v.*M)+M;%����0-1������˺�M��1�ĵط�V�Ͳ��䣬�ٳ���2. NICE!!ȷʵ�ã�������M��Ϊ1��λ���ϵĵط���ֵ�䷴
    %�����ǵ�� %����  
    %Results 
    x=decode(v(:,1:22),xmin,xmax);%���룬��Ŀ�꺯��ֵ
    fit=eval(f);        %������ֵ
    [sol,indb]=max(fit);% ÿ�ε���������Ŀ�꺯��ֵ������λ��
    v(1,:)=v(indb,:);   %�����ֵ����
    fit_mean=mean(fit); % ÿ�ε�����Ŀ�꺯��ֵ��ƽ��ֵ��mean���ֵ
    vx=[vx sol];        %������Ӧ��ֵ
    vmfit=[vmfit fit_mean];%��Ӧ�Ⱦ�ֵ
    it=it+1;            %������������������
end
%%%% �����
disp(sprintf('\n'));  %��һ��% ��ʾ���Ž⼰����ֵ
disp(sprintf('Maximum found[x,f(x)]:[%.4f,%.4f]',x(indb),sol));
% ͼ����ʾ���Ž��
figure(2);
fplot(f,[xmin,xmax]);
grid on;hold on;
plot(x,fit,'r*');
title('Ⱦɫ�������λ��');
xlabel('x');ylabel('f(x)');
% ͼ����ʾ���ż�ƽ������ֵ�仯����
figure(3);
plot(vx);
%title('����,ƽ������ֵ�仯����');
xlabel('Generations');ylabel('f(x)');hold on;
plot(vmfit,'r');hold off;
runtime=toc%��ʱ����
end
%%
%Decodify bitstrings
function x=decode(v,xymin,xymax)
% x    ----real value(precision:6)
% v    ----binary string(length:22)
v=fliplr(v); %ʵ�����ҷ�ת�ߵ�
[s,c]=size(v); %c�����������У���
aux=0:1:c-1;   %21ά����
aux=ones(s,1)*aux;%Ȩֵ��������
x1=sum((v.*2.^aux)');%Ȩֵ   %ע��ת��   %sum�����к�
x=xymin+(xymax-xymin)*x1./(2^c-1);    %���ֵ4194303;
end
%%
%Crossover
function v=crossover(vtemp,pc)
[N,L]=size(vtemp);
C(:,1)=rand(N,1)<=pc;%ѡ���ӽ��ġ�<=pc����1������0����0-1����
I=find(C(:,1)==1);%�ҷ�������1��Ԫ�أ����±깹��������
I';%���������
j=1;
for i=1:2:size(I)%�������������2Ϊ����
    if i>=size(I)%���������� ������������������һ�в�����
        break;
    end
    site=fix(1+L*rand(1));%fix����ȡ����L=22.%site����1-22.���ȷ���������λ��
    temp=vtemp(I(i,1),:);%�������ݴ������T  ��¼Ҫ����ĵ�һ�л���
    vtemp(I(i,1),site:end)=vtemp(I(i+1,1),site:end);%�����������ֵ
    vtemp(I(i+1,1),site:end)=temp(:,site:end);%����  tempû�б��޸�
end
v=vtemp;%���Ʒ���
end        
%%
%Function init_population
function v=init_population(n1,s1)
v=round(rand(n1,s1));%rand�����������%round��������ȡ��
end
%%
function vtemp=roulette(v,fit)
N=size(v);  %N����
fitmin=abs(min(fit));%��Сֵ�;���ֵ
fit=fitmin+fit; %��Сֵ���ϲ�������֤fit>=0.
%fit
S=sum(fit);%�������ĺ�
for i=1:N
    SI=S*rand(1);%rand�������0-s֮���һ�������
    for j=1:N
        if SI<=sum(fit(1:j))  %�ۼ���ֵ
            vtemp(i,:)=v(j,:);%ѡ�д�����
            break
        end
    end
end
end

