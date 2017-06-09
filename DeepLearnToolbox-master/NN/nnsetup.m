function nn = nnsetup(architecture)
%NNSETUP creates a Feedforward Backpropagate Neural Network
%首先从传入的architecture中获得这个网络的整体结构，nn.n表示这个网络有多少层，可以参照上面的样例调用nnsetup([784 100 10])加以理解 
% nn = nnsetup(architecture) returns an neural network structure with n=numel(architecture)
% layers, architecture being a n x 1 vector of layer sizes e.g. [784 100 10]

    nn.size   = architecture;
    nn.n      = numel(nn.size);
    
    nn.activation_function              = 'tanh_opt';   % 隐藏层的激活功能：'sigma'（sigmoid）或'tanh_opt'函数（最佳tanh）。          Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
    nn.learningRate                     = 2;            % 当使用“sign”激活功能和非归一化输入时，通常学习率值较低。              learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
    nn.momentum                         = 0.5;          %  Momentum
    nn.scaling_learningRate             = 1;            % （每个时期）学习率的缩放因子              Scaling factor for the learning rate (each epoch)
    nn.weightPenaltyL2                  = 0;            %  L2正规化           L2 regularization
    nn.nonSparsityPenalty               = 0;            %  非稀疏性处罚             Non sparsity penalty
    nn.sparsityTarget                   = 0.05;         %  稀疏性目标                  Sparsity target
    nn.inputZeroMaskedFraction          = 0;            %  用于去噪自动编码器          Used for Denoising AutoEncoders
    nn.dropoutFraction                  = 0;            %  退出级别      (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
    nn.testing                          = 0;            %  内部变量.nntest将此设置为一个。          Internal variable. nntest sets this to one.
    nn.output                           = 'sigm';       %  输出单位'sign'（= logistic），'softmax'和'linear'          output unit 'sigm' (=logistic), 'softmax' and 'linear'
    %对每一层的网络结构进行初始化，一共三个参数W,vW，p，其中W是主要的参数    
    %vW是更新参数时的临时参数，p是所谓的sparsity，(等看到代码了再细讲) 
    for i = 2 : nn.n   
        % weights and weight momentum
        nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1)));
        nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
        
        % average activations (for use with sparsity)
        nn.p{i}     = zeros(1, nn.size(i));   
    end
end
