function rbm = rbmtrain(rbm, x, opts)
    assert(isfloat(x), 'x must be a float');    %���Ժ�������������isfloat(x) �����x must be a float
    assert(all(x(:)>=0) && all(x(:)<=1), 'all data in x must be in [0:1]');
    m = size(x, 1);   %��x���������
    numbatches = m / opts.batchsize;   %  batchsize:һ�������������ݵĸ���          numbatchers:�����ж�����
    
    assert(rem(numbatches, 1) == 0, 'numbatches not integer');  %rem()������ ��mod()���в�ͬ

    for i = 1 : opts.numepochs
        kk = randperm(m);    %�������һ��1-m����������
        err = 0;
        for l = 1 : numbatches
            batch = x(kk((l - 1) * opts.batchsize + 1 : l * opts.batchsize), :);  %����ȡx���������batchsize�У�ÿһ���൱��һ������������
            
            v1 = batch;          %batch�������������ݣ�����batchsize
            h1 = sigmrnd(repmat(rbm.c', opts.batchsize, 1) + v1 * rbm.W');   %���ʴ���ĳ�������  ��P(h1j=1|v1)=sigmoid(bj + sum_i(v1i * Wij));
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

            err = err + sum(sum((v1 - v2) .^ 2)) / opts.batchsize;   %ÿһ������ԭ������Ԥ�����ݵķ���ͣ����ؽ����
        end
        
        disp(['epoch ' num2str(i) '/' num2str(opts.numepochs)  '. Average reconstruction error is: ' num2str(err / numbatches)]);
        
    end
end
