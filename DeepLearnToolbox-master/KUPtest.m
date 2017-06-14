tic
%读取训练数据
[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,class] = textread('Modbus_trafficf.txt' , '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f','headerlines',1);    

%特征值归一化
[input,minI,maxI] = premnmx( [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18]')  ;  %

%构造输出矩阵
s = length( class ) ;
output = zeros( s , 3  ) ;
for i = 1 : s 
   output( i , class( i )  ) = 1 ;
end

%创建神经网络
net = newff( minmax(input) , [10 4] , { 'logsig' 'purelin' } , 'traingdx' ) ;%minmax求最小最大值，[10,3]定义隐藏层和输出层神经元个数， { 'logsig' 'purelin' }代表每一层传递函数，'traingdx' 代表训练函数

%设置训练参数
net.trainparam.show = 50 ;
net.trainparam.epochs = 500 ;
net.trainparam.goal = 0.01 ;
net.trainParam.lr = 0.01 ;
net.trainParam.showWindow = false; 
net.trainParam.showCommandLine = false;
%开始训练
net = train( net, input , output' ) ;

%读取测试数据
[f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,c] = textread('test.txt' , '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f',150);

%测试数据归一化
testInput = tramnmx ( [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18]' , minI, maxI ) ;


%仿真
Y = sim( net , testInput ) ;
[s1 , s2] = size( Y ) ;
hitNum = 0 ;
for i = 1 : s2
    [m , Index] = max( Y( : ,  i ) ) ;    
    if( Index  == c(i)   ) 
        c(i)
        hitNum = hitNum + 1 ; 
    end
end
toc

%统计识别正确率
%  [s1 , s2] = size( Y ) ;
% hitNum = 0 ;
% for i = 1 : s2
%     [m , Index] = max( Y( : ,  i ) ) ;
%     
%     if( Index  == c(i)   ) 
%         hitNum = hitNum + 1 ; 
%     end
% end
% sprintf('识别率是 %3.3f%%',100 * hitNum / s2 )

