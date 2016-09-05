function g =  lin_grad(X,Y,w)
% X is a matrix of data as rows
% Y is a vector of labels

N = size(X,1);

g = (1/N)*(2*(X'*X)*w - 2*X'*Y);


end