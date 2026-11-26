%%
clear; clc; close all;
%%
data = readmatrix('数据.xlsx', 'Sheet', 1); 

% 划分输入和输出
input_data  = data(1:8, :);      % 前8行是输入
output_data = data(9:end, :);    % 其余是输出

% 划分训练集和测试集
train_columns = 1:1003;
test_columns  = 1004:size(data, 2);

train_x = input_data(:, train_columns);
train_y = output_data(:, train_columns);
test_x  = input_data(:, test_columns);
test_y  = output_data(:, test_columns);

% ---- MLP 需要样本在行方向 ----
XTrain = train_x.';   % [Ntrain × 8]
YTrain = train_y.';   % [Ntrain × 40]
XTest  = test_x.';    % [Ntest  × 8]
YTest  = test_y.';    % [Ntest  × 40]

dirname = 'MLP';
mkdir(dirname);
cd(dirname);

% 确定超参数
NumLayers        = 1;     % 隐藏层数
HiddenUnits      = 64;    % 每层神经元数
InitialLearnRate = 0.003; % 学习率
Epoch            = 350;   % 训练轮数
miniBatchSize    = 32;    % 最小批尺寸
DropoutProb      = 0;   % Dropout 概率

% 创建网络结构
numFeatures  = 8;    % 输入特征数
numResponses = 40;   % 输出响应数

layers = [ ...
    featureInputLayer(numFeatures, 'Normalization', 'zscore')
    mlpBlock(HiddenUnits, NumLayers, DropoutProb)  % 根据定义的神经元数与层数建立网络
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
    'Shuffle', 'every-epoch', ...
    'Verbose', 0);


% 网络训练
[net, training_process] = trainNetwork(XTrain, YTrain, layers, options);

% 测试网络
result = predict(net, XTest);

% 计算误差
valError = mean(mean(abs(result - YTest) ./ abs(YTest)));  % 平均相对误差
fileName ="model"+ num2str(valError) + ".mat";              % 将相关结果保存到 mat 文件
save(fileName, 'training_process', 'net', 'valError', 'options', 'layers');

% ================= MLP 块建立函数 =================
function layers = mlpBlock(numHiddenUnits, numLayers, dropoutProb)
    layers = [];
    for k = 1:numLayers
        layers_single = [
            fullyConnectedLayer(numHiddenUnits)
            batchNormalizationLayer
            reluLayer];
        if dropoutProb > 0
            layers_single = [layers_single; dropoutLayer(dropoutProb)];
        end
        layers = [layers; layers_single];
    end
end