%%
clear; clc; close all;
%%
data = readmatrix('数据.xlsx', 'Sheet', 1);  % 读取 Excel 文件


% 划分输入和输出
input_data = data(1:8, :);     % 前8行是输入
output_data = data(9:end, :);  % 其余是输出

% 划分训练集和测试集
train_columns = 1:1003;  % 训练集列索引
test_columns = 1004:size(data,2);  % 测试集列索引

train_x = input_data(:, train_columns);
train_y = output_data(:, train_columns);
test_x = input_data(:, test_columns);
test_y = output_data(:, test_columns);


XTrain=(train_x);
YTrain=(train_y);
XTest=(test_x);
YTest=(test_y)';

row_means1 = mean(test_y, 2); % 40×1

load  D:\LSTM\model.mat


result=predict(net,XTest);
row_means = mean(result, 2); % 40×1


YTest_means = mean(YTest, 1)';  % 1×40

rmse = sqrt(mean((row_means - YTest_means).^2));
% 计算 MAE
mae = mean(abs(row_means - YTest_means));

r2  = 1 - sum((YTest_means - row_means).^2, 'all') / sum((YTest_means - mean(YTest_means, 'all')).^2, 'all');

metrics_vector = [rmse, mae, r2];
% 或者创建一个表格以便查看
metrics_table = table(rmse, mae, r2, 'VariableNames', {'RMSE', 'MAE', 'R2'});
disp(metrics_table);

