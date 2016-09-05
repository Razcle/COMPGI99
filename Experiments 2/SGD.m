function errors = SGD(X,Y,W_est,iter)

N = size(X,1);

errors = zeros(1,iter);
w = W_est;
rate = 0.01;

for i = 1:iter
    errors(i) = (1/N)*(X*w-Y)'*(X*w-Y);
    index = randi(N);
    x = X(index,:);
    y = Y(index);
    grad = lin_grad(x,y,w);
    w = w - rate*grad;
end




end