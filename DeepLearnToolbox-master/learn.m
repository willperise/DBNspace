v1 = [3,3,3;4,4,4;5,5,5];
v2 = [1,2,3;2,3,4;3,4,4];
a = (v1 - v2) ^ 2;
b = (v1 - v2) .^ 2;
c = sum((v1 - v2) .^ 2);
d = sum(sum((v1 - v2) .^ 2));
%test
disp(a);
disp(b);
disp(c);
disp(d);
%imshow(train_x);
