function dbn = dbntrain(dbn, x, opts)
    n = numel(dbn.rbm);   %���RBMѵ����������
fprintf('dbntrain numel()_dbn.rbm = %d',n);
    dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, opts);   %��ʼ����һ��RBMѵ����
    for i = 2 : n    %��n��ѵ����
        x = rbmup(dbn.rbm{i - 1}, x);
        dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, opts);
    end

end
