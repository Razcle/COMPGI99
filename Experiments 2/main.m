%% Experiment 1
% In this script we will attempt to demnstrate some intuition on AW-SGD
clc;clear;close all;

%% Generate some Linear Noisy Data
close all; clear;clc; hold off;
N = 50; D = 1; sigma = 0.2; 
X = [randn(N,D),ones(N,1)]; W = randn(D+1,1);
X = sort(X,1);
Y_true = X*W;
noise = sigma*randn(N,1);
Y = Y_true + noise;
plot(X(:,1),Y,'o')
hold on
plot(X(:,1),Y_true,'r-')
hold off

%% Pick a random starting Point for W and tau and calculate the probability
% distribution 

W_est = randn(D+1,1) ; %random initial starting point
Y_est = X*W_est;

%Define the sampling distribution with its inital parameters
mu = 0;
sigmaSq = 1.5;
prob = exp(-0.5*(X(:,1)-mu).^2/sigmaSq);
prob = prob/sum(prob);

%Calculate the residuals and the true gradient for plotting later
residuals =Y_true - Y_est;
true_grad = (1/N)*(2*(X'*X)*W_est - 2*X'*Y);

%% Determine which of the points give the closes aproximation to the true
% gradient. Will use Cosine similarity for this.

%function to calculate apporximate gradient
grad_est = @(x,y) 2*(x*W_est - y)*x;

%place holders for the similarities and the gradient estimate from each
%point.
cos_similarities = zeros(N,1);
grad_estimates = zeros(N,2);

for i = 1:N
    grad_estimates(i,:) = grad_est(X(i,:),Y(i));
    cos_similarities(i) = cosine_sim(grad_estimates(i,:),true_grad);
end

% sort the gradient estimates by cos similarity so that we can see which is
% best.
estimatesBySimilarity = [grad_estimates,cos_similarities,(1:N)'];
[~,idx] = sortrows(estimatesBySimilarity,3);
estimatesBySimilarity = estimatesBySimilarity(idx,:);

%% Plots

figure()
title('Estimated Gradients and true Gradient (red) centred on w')
hold on
compass(grad_estimates(:,1),grad_estimates(:,2))
compass(true_grad(1),true_grad(2),'r')
hold off

figure()
title('Best Estimated Gradients and true Gradient (red) centred on w')
hold on
compass(estimatesBySimilarity(end-10:end,1),estimatesBySimilarity(end-10:end,2))
compass(true_grad(1),true_grad(2),'r')
hold off

figure()
title('Worst Estimated Gradients and true Gradient (red) centred on w')
hold on
compass(estimatesBySimilarity(1:10,1),estimatesBySimilarity(1:10,2))
compass(true_grad(1),true_grad(2),'r')
hold off

%%
figure()
subplot(4,1,1)
plot(X(:,1),Y,'o')
title('Present best fit line')
hold on
plot(X(:,1),Y_est,'r-');
hold off
subplot(4,1,2)
bar(residuals);
title('residuals')
subplot(4,1,3)
bar(cos_similarities);
title('Cosine Similarity of Gradient estimate') 
subplot(4,1,4)
a = linspace(0,50,200);
b = normpdf(a,25,20);
plot(a,b)
title('Initial Sampling Distribution')


%%  Perfrom BB-AWSGD


iter = 10;
average_BB_error = zeros(1,iter);
average_sgd_error = zeros(1,iter);
average_taus = zeros(1,iter);
average_index = zeros(1,iter);


repeats = 1;
tau = 25;

for i =1:repeats
    [errors_BB_AWSGD,taus,indices,W_est,vs] = BB_AWSGD(iter,tau,W_est,X,Y);
    average_taus = average_taus + taus;
    average_index = average_index + indices;
    average_BB_error = average_BB_error + errors_BB_AWSGD;
    [errors_SGD] = SGD(X,Y,W_est,iter);
    average_sgd_error = average_sgd_error + errors_SGD;
end

average_BB_error = average_BB_error/repeats;
average_sgd_error = average_sgd_error/repeats;
average_taus = average_taus/repeats;
average_index = average_index/repeats;

plot(average_BB_error);hold on; plot(average_sgd_error,'r')
figure()
plot(average_index,'o')
figure()
plot(average_taus)



