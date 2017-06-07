function test_example_DBN
load('data/mnist_uint8');

train_x = double(train_x) / 255;
test_x  = double(test_x)  / 255;
train_y = double(train_y);
test_y  = double(test_y);

%%  ex1 train a 100 hidden unit RBM and visualize its weights
rand('state',0) 
dbn.sizes = [100];     % DBN的结构，v1层为raw pixel/原始图片，h1/v2层的节点数为100
opts.numepochs =   1;    % opts普通的工具体  同一部分数据执行RBM的次数
opts.batchsize = 100;    %  一批输入样本的个数  numbatches 代表有多少批
opts.momentum  =   0;    % 冲量项  记录以前的更新方向（更新数据之用），并与现在的方向结合下，从而加快学习的速度
opts.alpha     =   1;    %  学习速率
dbn = dbnsetup(dbn, train_x, opts);
dbn = dbntrain(dbn, train_x, opts);
figure; visualize(dbn.rbm{1}.W');   %  Visualize the RBM weights

%%  ex2 train a 100-100 hidden unit DBN and use its weights to initialize a NN test
rand('state',0)
%train dbn
dbn.sizes = [100 100];   %DBN的结构，v1层为raw pixel/原始图片，h1/v2层的节点数为100，h2/v3层的节点数为200
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
