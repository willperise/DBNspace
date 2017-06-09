function [er, bad] = nntest(nn, x, y)   
%����һ��nnpredict���ں�test �ļ���y���бȽ�
    labels = nnpredict(nn, x);
    [dummy, expected] = max(y,[],2);
    bad = find(labels ~= expected);    
    er = numel(bad) / size(x, 1);
end
