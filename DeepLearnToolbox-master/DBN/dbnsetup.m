function dbn = dbnsetup(dbn, x, opts)
    n = size(x, 2);
    dbn.sizes = [n, dbn.sizes];  % ����Ϊ dbn.sizes����������+1
%��ʼ��W,b,c
    for u = 1 : numel(dbn.sizes) - 1   
        dbn.rbm{u}.alpha    = opts.alpha;
        dbn.rbm{u}.momentum = opts.momentum;

        dbn.rbm{u}.W  = zeros(dbn.sizes(u + 1), dbn.sizes(u));  % Ȩֵ����
        dbn.rbm{u}.vW = zeros(dbn.sizes(u + 1), dbn.sizes(u));

        dbn.rbm{u}.b  = zeros(dbn.sizes(u), 1);   % ���Ӳ��ƫ��bias
        dbn.rbm{u}.vb = zeros(dbn.sizes(u), 1);

        dbn.rbm{u}.c  = zeros(dbn.sizes(u + 1), 1);    % �����ƫ��bias
        dbn.rbm{u}.vc = zeros(dbn.sizes(u + 1), 1);
    end

end
