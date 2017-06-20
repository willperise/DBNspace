clear;
clc;
% tic
% load('corrected.txt');
% load('kddcup.data_10_percent_corrected.txt');
% toc
tic
%读取训练数据
[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36,f37,f38,f39,f40,f41,class] = textread('kddcup.data_10_percent_corrected.txt' , '%f%f%f%f%f%f%f%f%f%f %f%f%f%f%f%f%f%f%f%f %f%f%f%f%f%f%f%f%f%f %f%f%f%f%f%f%f%f%f%f %f%f');    
disp(length(class));
%特征值归一化
[input,minI,maxI] = premnmx( [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,f19,f20,f21,f22,f23,f24,f25,f26,f27,f28,f29,f30,f31,f32,f33,f34,f35,f36,f37,f38,f39,f40,f41]')  ;  %

%构造输出矩阵
s = length( class ) ;
output = zeros( s , 35  ) ;%494021*35
%output
for i = 1 : s 
   output( i , class( i )  ) = 1 ;
end


%创建神经网络
net = newff( minmax(input) , [110 35] , { 'logsig' 'purelin' } , 'traingdx' ) ;%minmax求最小最大值，[10,3]定义隐藏层和输出层神经元个数， { 'logsig' 'purelin' }代表每一层传递函数，'traingdx' 代表训练函数

%设置训练参数
net.trainparam.show = 50 ;
net.trainparam.epochs = 1500 ;
net.trainparam.goal = 0.01 ;
net.trainParam.lr = 0.01 ;
net.trainParam.showWindow = false; 
net.trainParam.showCommandLine = false;
%开始训练
net = train( net, input , output' ) ;
%读取测试数据
[t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,f30,t31,t32,t33,t34,t35,t36,t37,t38,t39,t40,t41,c] = textread('corrected.txt' , '%f%f%f%f%f%f%f%f%f%f %f%f%f%f%f%f%f%f%f%f %f%f%f%f%f%f%f%f%f%f %f%f%f%f%f%f%f%f%f%f %f%f');

%测试数据归一化
testInput = tramnmx ( [t1,t2,t3,t4,t5,t6,t7,t8,t9,t10,t11,t12,t13,t14,t15,t16,t17,t18,t19,t20,t21,t22,t23,t24,t25,t26,t27,t28,t29,f30,t31,t32,t33,t34,t35,t36,t37,t38,t39,t40,t41]' , minI, maxI ) ;


%仿真
Y = sim( net , testInput ) ;
% [s1 , s2] = size( Y ) ;
% hitNum = 0 ;
% for i = 1 : s2
%     [m , Index] = max( Y( : ,  i ) ) ;    
%     if( Index  == c(i)   ) 
%         c(i)
%         hitNum = hitNum + 1 ; 
%     end
% end
% toc
%统计识别正确率
 [s1 , s2] = size( Y ) ;
hitNum = 0 ;
for i = 1 : s2
    [m , Index] = max( Y( : ,  i ) ) ;
    
    if( Index  == c(i)   ) 
        hitNum = hitNum + 1 ; 
    end
end
sprintf('识别率是 %3.3f%%',100 * hitNum / s2 )
toc

