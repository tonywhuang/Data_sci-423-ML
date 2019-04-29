clear
clc
close all

% Load data into X and y
data = csvread('breast_cancer_data.csv');
X = data(:,1:end-1);
y = data(:,end);

% Transform X into (N+1)*P matrix
X = [ones(size(X,1),1) X]';

% Softmax method
Softmaxmis = [];
w = zeros(9,1);
for ii = 1:10
    % Check for number of misclassification
    mis = 0;
    for jj = 1:699
        p = X(:,jj)'*w*y(jj);
        if p < 0
            mis = mis + 1;
        end
    end
    Softmaxmis(ii) = mis;
    
    % Update w with Newton's method
    r1 = [];
    for jj = 1:699
        sigmoid = 1/(1+exp(y(jj)*(X(:,jj)'*w)));
        r1(jj,1) = (-1)*sigmoid*y(jj);
    end
    grad = X*r1;
    
    Hess = zeros(9,9);
    for jj = 1:699
        sigmoid = 1/(1+exp(y(jj)*(X(:,jj)'*w)));
        Hes = sigmoid*(1-sigmoid)*X(:,jj)*X(:,jj)';
        Hess = Hess+Hes;
    end
    
    w = w - pinv(Hess)*grad;
end

% Squared max method
Sqrdmaxmis = [];
w = zeros(9,1);
for ii = 1:10
    % Count number of misclassifications
    mis = 0;
    for jj = 1:699
        p = X(:,jj)'*w*y(jj);
        if p < 0
            mis = mis + 1;
        end
    end
    Sqrdmaxmis(ii) = mis;
    
    grad = zeros(9,1);
    for jj = 1:699
        gra = max(0,1-y(jj)*(X(:,jj)'*w))*y(jj)*X(:,jj);
        grad = grad + gra;
    end
    grad = grad*(-2);
    
    Hess = zeros(9,9);
    for jj = 1:699
        if (1 - y(jj)*(X(:,jj)'*w)) > 0
            Hess = Hess + X(:,jj)*X(:,jj)';
        end
    end
    Hess = Hess*2;
    
    w = w - pinv(Hess)*grad;
end

hold on
plot([1:10],Softmaxmis)
plot([1:10],Sqrdmaxmis)
axis([2 10 26 32])
xlabel('iteration')
ylabel('number of misclassifications')
legend('softmax','squared max')
title('breast cancer dataset')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    