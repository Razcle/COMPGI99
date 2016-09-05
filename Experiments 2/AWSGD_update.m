function mu = AWSGD_update(mu,prob,x,y,sigmaSq,W_est,X)
% Updates mu, the parameter of the sampling distribution
% see equations 4 and 5 of debug docs 
%     Inputs
%     ------
%         mu: float
%             present value of my
%        prob:float
%             weight of sampled point
%        x,y: floats
%             sample from the data-set
%        sigmaSq: float
%                 variance
%        W_est: float
%               Present value of W
%        X: matrix
%           Data with rows as examples
%     return
%     -------
%         mu: float
%             updated value of my
       

rate = 0.001; % Learning rate

d = (x*W_est - y)*x/prob;

expectation = sum(((X(:,1)-mu)/(sigmaSq)).*exp(-0.5*(X(:,1)-mu).^2/sigmaSq));
expectation = expectation/sum(exp(-0.5*(X(:,1)-mu).^2/sigmaSq));
gradlog = (x(1)-mu)/(sigmaSq) - expectation;

delta = (d*d')*gradlog;
mu = mu + rate*delta;

end