function dbn = dbnsetup(dbn, x, opts)
    n = size(x, 2);
    dbn.sizes = [n, dbn.sizes];  % 列数为 dbn.sizes行向量列数+1
%初始化W,b,c
    for u = 1 : numel(dbn.sizes) - 1   
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;

        dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u));  % 权值向量
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);   % 可视层的偏置bias
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);    % 隐层的偏置bias
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
