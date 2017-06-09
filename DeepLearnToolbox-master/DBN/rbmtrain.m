function rbm = rbmtrain(rbm, x, opts)
    assert(isfloat(x), 'x must be a float');    %断言函数，若不满足isfloat(x) 就输出x must be a float
    assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
    m = size(x, 1);   %求x矩阵的行数
    numbatches = m / opts.batchsize;   %  batchsize:一批样本输入数据的个数          numbatchers:代表有多少批
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');  %rem()求余数 与mod()略有不同

    for i = 1 : opts.numepochs
        kk = randperm(m);    %随机打乱一个1-m的数组序列
        err = 0;
        for l = 1 : numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);  %任意取x数组里面的batchsize行，每一行相当于一个输入样本。
            
            v1 = batch;          %batch：输入样本数据，个数batchsize
            h1 = sigmrnd(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W');   %概率大于某个随机数  求P(h1j=1|v1)=sigmoid(bj + sum_i(v1i * Wij));
            v2 = sigmrnd(repmat(rbm.b', opts.batchsize, 1) + h1 * rbm.W);
            h2 = sigm(repmat(rbm.c', opts.batchsize, 1) + v2 * rbm.W');

            c1 = h1' * v1;
            c2 = h2' * v2;

            rbm.vW = rbm.momentum * rbm.vW + rbm.alpha * (c1 - c2)     / opts.batchsize;
            rbm.vb = rbm.momentum * rbm.vb + rbm.alpha * sum(v1 - v2)' / opts.batchsize;
            rbm.vc = rbm.momentum * rbm.vc + rbm.alpha * sum(h1 - h2)' / opts.batchsize;

            rbm.W = rbm.W + rbm.vW;
            rbm.b = rbm.b + rbm.vb;
            rbm.c = rbm.c + rbm.vc;

            err = err + sum(sum((v1 - v2) .^ 2)) / opts.batchsize;   %每一批数据原样本与预测数据的方差和，即重建误差
        end
        
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
        
    end
end
