clear;
close all;
clc;

[X_train, y_train] = load_MNIST('MNIST_train_data.csv');
learning_rate = 0.001;
num_iter = 500;
%num_iter = [100,200,300,500,800,1000,1200,1500,2000];
W = train(X_train, y_train, num_iter, learning_rate);

y_pred = predict(W, X_train);
fprintf('The training accuracy is %f\n', mean(y_pred == y_train))

[X_test, y_test] = load_MNIST('MNIST_test_data.csv');
y_pred = predict(W, X_test);
fprintf('The test accuracy is %f\n', mean(y_pred == y_test))

% train_accuracy = [];
% test_accuracy = [];
% for ii = 1:numel(num_iter)
%     W = train(X_train, y_train, num_iter(ii), learning_rate);
% 
%     y_pred = predict(W, X_train);
%     %fprintf('The training accuracy is %f\n', mean(y_pred == y_train))
%     train_accuracy(ii) = mean(y_pred == y_train);
% 
%     
%     y_pred = predict(W, X_test);
%     test_accuracy(ii) = mean(y_pred == y_test);
%     %fprintf('The test accuracy is %f\n', mean(y_pred == y_test))
% end
% figure(1)
% title('training accuracy')
% plot(num_iter,train_accuracy)
% 
% figure(2)
% title('test accuracy')
% plot(num_iter,test_accuracy)

function [X_data, y_data] = load_MNIST(filename)
% Load the MNIST dataset
% X_data: N by 785 matrix
% y_data: N by 1 matrix
data = csvread(filename);
y_data = data(:, end);
X_data = append_bias(data(:, 1:end-1));
end

function X_new = append_bias(X_old)
% Append an extra bias dimension to X_old as the FIRST column
% X_old: N by D matrix
% X_new: N by (D+1) matrix
% hint: you may find size() and ones() useful

X_new = [ones(size(X_old,1),1) X_old];

end


function W = train(X_train, y_train, num_iter, lr)
% Train the One-versus-All classifier
% X_train: N by 785 matrix
% y_train: N by 1 matrix
% num_iter: number of iterations
% lr: learning rate
% W: 785 by 10 matrix

D = size(X_train, 2); % Number of features
C = 10; % Number of classes
W =  zeros(D, C);

y_converted = convert_to_binary_class(y_train, C);
for i = 1: num_iter
    grad = gradient_descent(W, X_train, y_converted);
    W = W - lr * grad;
end


end

function y_out = convert_to_binary_class(y_in, C)
% Convert y from a multiclass problem to a binary class one
% y_in: N by 1 matrix
% C: Number of class
% y_out: N by 10 matrix, each column consists of +1 or -1
% hint: you can use for loop for this part. And you may need logical indexing
y_out = ones(size(y_in,1),C);
y_out = y_out*(-1);
for ii = 1:size(y_in,1)
    col = y_in(ii,1);
    y_out(ii,col) = 1;
end

end

function grad = gradient_descent(W, X, y)
% Gradient descent for c-th classifier
% way to computing gradient on Canvas
%
% W: 785 by 10 matrix
% X: N by 785 matrix
% y: N by 10 matrix
% grad: gradient of W, 785 by 10 matrix
% hint: you may find sigmoid() below useful
X = X';
grad = zeros(size(X,1),10);
for ii = 1:10
    r1 = zeros(size(X,2),1);
    for jj = 1:size(X,2)
        r1(jj,1) = -1*sigmoid(-y(jj,ii)*(X(:,jj)'*W(:,ii)))*y(jj,ii);
    end
    grad(:,ii) = X*r1;
end

end


function y = sigmoid(z)
% Sigmoid function
y = zeros(size(z));
mask = (z >= 0.0);
y(mask) = 1./(1 + exp(-z(mask)));
y(~mask) = exp(z(~mask))./(1 + exp(z(~mask)));
end

function y_pred = predict(W, X)
% Calculate the scores in each class and predict the class label
% W: weight matrix, 785 by 10
% X: N by 785
% hint: you may find max(A, [], 2) very useful
y_pred = zeros(size(X,1),1);
for ii = 1:size(X,1)
    ind = zeros(1,10);
    for jj = 1:10
        ind(jj) = X(ii,:)*W(:,jj);
    end
    [~,y_pred(ii,1)] = max(ind);

end

