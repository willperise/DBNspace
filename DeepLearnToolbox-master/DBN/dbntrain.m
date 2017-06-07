function dbn = dbntrain(dbn, x, opts)
    n = numel(dbn.rbm);   %求出RBM训练机的轮数
fprintf('dbntrain numel()_dbn.rbm = %d',n);
    dbn.rbm{1} = rbmtrain(dbn.rbm{1}, x, opts);   %初始化第一个RBM训练机
    for i = 2 : n    %有n轮训练机
        x = rbmup(dbn.rbm{i - 1}, x);
        dbn.rbm{i} = rbmtrain(dbn.rbm{i}, x, opts);
    end

end
