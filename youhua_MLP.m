%%
clear; clc; close all;
%%
data = readmatrix('数据.xlsx', 'Sheet', 1);   % 读取 Excel

% ---- 划分输入和输出 ----
input_data  = data(1:8, :);      % 前8行输入
output_data = data(9:end, :);    % 其余输出

train_columns = 1:1003;  % 训练集列索引
test_columns = 1004:size(data,2);  % 测试集列索引

train_x = input_data(:, train_columns);
train_y = output_data(:, train_columns);
test_x  = input_data(:, test_columns);
test_y  = output_data(:, test_columns);


XTrain = train_x.';   % [Ntrain × 8]
YTrain = train_y.';   % [Ntrain × 40]
XTest  = test_x.';    % [Ntest  × 8]
YTest  = test_y.';    % [Ntest  × 40]   ← 注意：这里直接转置，不要再单独加 '
% ===========================================

% ---- 读取 MLP 训练保存的模型 ----

load('D:\LSTM\model.mat');

% ---- 预测 ----
result = predict(net, XTest);   % [Ntest × 40]


row_means  = mean(result, 1);    % 1×40，每个输出维度的预测均值
YTest_means = mean(YTest, 1);    % 1×40，每个输出维度的真实均值

rmse = sqrt(mean((row_means - YTest_means).^2));
mae  = mean(abs(row_means - YTest_means));
r2   = 1 - sum((YTest_means - row_means).^2, 'all') / ...
           sum((YTest_means - mean(YTest_means, 'all')).^2, 'all');

metrics_table = table(rmse, mae, r2, ...
    'VariableNames', {'RMSE','MAE','R2'});
disp(metrics_table);


