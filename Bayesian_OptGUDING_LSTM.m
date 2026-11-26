%%
clear; clc; close all;
%% 准备数
data = readmatrix('数据.xlsx', 'Sheet', 1); 

% 划分输入和输出
input_data = data(1:8, :);     % 前8行是输入
output_data = data(9:end, :);  % 其余是输出

% 划分训练集和测试集
train_columns = 1:965;  % 训练集列索引
test_columns = 966:size(data, 2);  % 测试集列索引

train_x = input_data(:, train_columns);
train_y = output_data(:, train_columns);
test_x = input_data(:, test_columns);
test_y = output_data(:, test_columns);

XTrain = (train_x);
YTrain = (train_y);
XTest = (test_x);
YTest = (test_y);

dirname = ['LSTM.8661']; % 新的文件夹名,用于存储优化结果
mkdir(dirname);          % 创建文件夹
cd(dirname);

% 确定超参数
numHiddenUnits = 64;    % 假设的隐含层单元数量
SectionDepth = 1;       % 假设的LSTM层数
InitialLearnRate = 0.003; % 假设的学习率
Epoch = 350;            % 假设的训练轮数
miniBatchSize = 32;     % 假设的最小批尺寸

% 创建网络结构
numFeatures = 8;         % 输入的尺寸
numResponses = 40;       % 输出的尺寸

layers = [ ...
    sequenceInputLayer(numFeatures)
    convBlock(numHiddenUnits, SectionDepth) % 根据定义的隐含层单元数量与层数建立网络
    fullyConnectedLayer(numResponses)
    regressionLayer];

% 网络参数
options = trainingOptions('adam', ...
    'MaxEpochs', Epoch, ...
    'GradientThreshold', 1, ...
    'InitialLearnRate', InitialLearnRate, ...
    'MiniBatchSize', miniBatchSize, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropPeriod', 125, ...
    'LearnRateDropFactor', 0.2, ...
    'SequenceLength', 'longest', ...
    'Verbose', 0);

% 网络训练
[net, training_process] = trainNetwork(XTrain, YTrain, layers, options);

% 测试网络
result = predict(net, XTest);

% 计算误差
valError = mean(mean(abs(result - YTest) ./ abs(YTest))); % 计算平均相对误差
fileName = "model" + num2str(valError) + ".mat"  ; % 将相关结果保存到mat文件
save(fileName, 'training_process', 'net', 'valError', 'options', 'layers');

% LSTM层建立函数
function layers = convBlock(numHiddenUnits, numLstmLayers)  % 建立LSTM的层
    layers = [];
    for Num_layer = 1:numLstmLayers
        layers_single = [
            batchNormalizationLayer
            lstmLayer(numHiddenUnits)
            leakyReluLayer];
        layers = [layers; layers_single];
    end
end