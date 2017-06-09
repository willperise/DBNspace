function nn = nnsetup(architecture)
%NNSETUP creates a Feedforward Backpropagate Neural Network
%���ȴӴ����architecture�л��������������ṹ��nn.n��ʾ��������ж��ٲ㣬���Բ����������������nnsetup([784 100 10])������� 
% nn = nnsetup(architecture) returns an neural network structure with n=numel(architecture)
% layers, architecture being a n x 1 vector of layer sizes e.g. [784 100 10]

    nn.size   = architecture;
    nn.n      = numel(nn.size);
    
    nn.activation_function              = 'tanh_opt';   % ���ز�ļ���ܣ�'sigma'��sigmoid����'tanh_opt'���������tanh����          Activation functions of hidden layers: 'sigm' (sigmoid) or 'tanh_opt' (optimal tanh).
    nn.learningRate                     = 2;            % ��ʹ�á�sign������ܺͷǹ�һ������ʱ��ͨ��ѧϰ��ֵ�ϵ͡�              learning rate Note: typically needs to be lower when using 'sigm' activation function and non-normalized inputs.
    nn.momentum                         = 0.5;          %  Momentum
    nn.scaling_learningRate             = 1;            % ��ÿ��ʱ�ڣ�ѧϰ�ʵ���������              Scaling factor for the learning rate (each epoch)
    nn.weightPenaltyL2                  = 0;            %  L2���滯           L2 regularization
    nn.nonSparsityPenalty               = 0;            %  ��ϡ���Դ���             Non sparsity penalty
    nn.sparsityTarget                   = 0.05;         %  ϡ����Ŀ��                  Sparsity target
    nn.inputZeroMaskedFraction          = 0;            %  ����ȥ���Զ�������          Used for Denoising AutoEncoders
    nn.dropoutFraction                  = 0;            %  �˳�����      (http://www.cs.toronto.edu/~hinton/absps/dropout.pdf)
    nn.testing                          = 0;            %  �ڲ�����.nntest��������Ϊһ����          Internal variable. nntest sets this to one.
    nn.output                           = 'sigm';       %  �����λ'sign'��= logistic����'softmax'��'linear'          output unit 'sigm' (=logistic), 'softmax' and 'linear'
    %��ÿһ�������ṹ���г�ʼ����һ����������W,vW��p������W����Ҫ�Ĳ���    
    %vW�Ǹ��²���ʱ����ʱ������p����ν��sparsity��(�ȿ�����������ϸ��) 
    for i = 2 : nn.n   
        % weights and weight momentum
        nn.W{i - 1} = (rand(nn.size(i), nn.size(i - 1)+1) - 0.5) * 2 * 4 * sqrt(6 / (nn.size(i) + nn.size(i - 1)));
        nn.vW{i - 1} = zeros(size(nn.W{i - 1}));
        
        % average activations (for use with sparsity)
        nn.p{i}     = zeros(1, nn.size(i));   
    end
end
