function [errors,taus,indices,w,vs] = BB_AWSGD(iter,tau,W_est,X,Y)


eta = 0.0001;
nu = 0.00000000001;

D = size(X,2) - 1;
N = size(X,1);

w = W_est;
mean = 0;
taus = zeros(1,iter);
indices = zeros(1,iter);
vs = zeros(1,iter);
means = zeros(1,iter);
ws = zeros(D+1,iter);
errors = zeros(1,iter);



for i = 1:iter
    delta = 20*randn() + tau;
    index = max(1,min(N,round(delta)));
    x = X(index,:);
    y = Y(index);
    d = lin_grad(x,y,w)/normpdf(delta,tau,20);
    w = w - eta*d;
    v = norm(d)^2;
    tau = tau - (nu)*(v-mean)*(delta - tau);
    mean = mean - (1/i)*(mean - v);
    means(i) = mean;
    vs(i) = v;
    ws(:,i) = w;
    errors(i) = (1/N)*(X*w-Y)'*(X*w-Y);
    taus(i) = tau;
    indices(i) = index;
end

end
