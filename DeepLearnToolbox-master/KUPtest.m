load('corrected.txt')
% tic%test
% %��ȡѵ������
% [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,class] = textread('Modbus_trafficf.txt' , '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f','headerlines',1);    
% 
% %����ֵ��һ��
% [input,minI,maxI] = premnmx( [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18]')  ;  %
% 
% %�����������
% s = length( class ) ;
% output = zeros( s , 3  ) ;
% for i = 1 : s 
%    output( i , class( i )  ) = 1 ;
% end
% 
% %����������
% net = newff( minmax(input) , [10 4] , { 'logsig' 'purelin' } , 'traingdx' ) ;%minmax����С���ֵ��[10,3]�������ز���������Ԫ������ { 'logsig' 'purelin' }����ÿһ�㴫�ݺ�����'traingdx' ����ѵ������
% 
% %����ѵ������
% net.trainparam.show = 50 ;
% net.trainparam.epochs = 500 ;
% net.trainparam.goal = 0.01 ;
% net.trainParam.lr = 0.01 ;
% net.trainParam.showWindow = false; 
% net.trainParam.showCommandLine = false;
% %��ʼѵ��
% net = train( net, input , output' ) ;
% 
% %��ȡ��������
% [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18,c] = textread('test.txt' , '%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f%f',150);
% 
% %�������ݹ�һ��
% testInput = tramnmx ( [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,f14,f15,f16,f17,f18]' , minI, maxI ) ;
% 
% 
% %����
% Y = sim( net , testInput ) ;
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
% 
% %ͳ��ʶ����ȷ��
% %  [s1 , s2] = size( Y ) ;
% % hitNum = 0 ;
% % for i = 1 : s2
% %     [m , Index] = max( Y( : ,  i ) ) ;
% %     
% %     if( Index  == c(i)   ) 
% %         hitNum = hitNum + 1 ; 
% %     end
% % end
% % sprintf('ʶ������ %3.3f%%',100 * hitNum / s2 )
% 
