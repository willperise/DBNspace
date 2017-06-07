function test_example_DBN
load('data/mnist_uint8');

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rand('state',0) 
dbn.sizes = [100];     % DBN�Ľṹ��v1��Ϊraw pixel/ԭʼͼƬ��h1/v2��Ľڵ���Ϊ100
opts.numepochs =   1;    % opts��ͨ�Ĺ�����  ͬһ��������ִ��RBM�Ĵ���
opts.batchsize = 100;    %  һ�����������ĸ���  numbatches �����ж�����
opts.momentum  =   0;    % ������  ��¼��ǰ�ĸ��·��򣨸�������֮�ã����������ڵķ������£��Ӷ��ӿ�ѧϰ���ٶ�
opts.alpha     =   1;    %  ѧϰ����
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN test
rand('state',0)
%train dbn
dbn.sizes = [100 100];   %DBN�Ľṹ��v1��Ϊraw pixel/ԭʼͼƬ��h1/v2��Ľڵ���Ϊ100��h2/v3��Ľڵ���Ϊ200
opts.numepochs =   1;
opts.batchsize = 100;
opts.momentum  =   0;
opts.alpha     =   1;
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);

%unfold dbn to nn
nn = dbnunfoldtonn(dbn, 10);
nn.activation_function = 'sigm';

%train nn
opts.numepochs =  1;
opts.batchsize = 100;
nn = nntrain(nn, train_x, train_y, opts);
[er, bad] = nntest(nn, test_x, test_y);

assert(er < 0.10, 'Too big error');
