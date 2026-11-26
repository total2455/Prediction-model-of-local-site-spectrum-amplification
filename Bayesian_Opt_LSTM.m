%%
clear; clc; close all;
%% 准备数
data = readmatrix('数据.xlsx'); 

% 划分输入和输出
input_data = data(1:8, :);     % 前8行是输入
output_data = data(9:end, :);  % 其余是输出

% 划分训练集和测试集
train_columns = 1:1003;  % 训练集列索引
test_columns = 1004:size(data,2);  % 测试集列索引

%numSamples = size(data, 2);          % 总样本数
%numTrain   = round(0.8 * numSamples);% 训练集样本数


%train_columns = permIdx(1:numTrain);          % 训练集列索引
%test_columns  = permIdx(numTrain+1:end);      % 测试集列索引


train_x = input_data(:, train_columns);
train_y = output_data(:, train_columns);
test_x = input_data(:, test_columns);
test_y = output_data(:, test_columns);

XTrain=(train_x);
YTrain=(train_y);
XTest=(test_x);
YTest=(test_y);

%XTrain=mapminmax(train_x,0,1);
%YTrain=mapminmax(train_y,0,1);
%XTest=mapminmax(test_x,0,1);
%[YTest,PS]=mapminmax(test_y,0,1);

dirname=['LSTM.8661'];%新的文件夹名,用于存储优化结果
a=['mkdir ' dirname];%创建命令
system(a); %创建文件夹
cd(dirname);%进入新的文件夹

%ObjFcn = makeObjFcn1(XTrain,YTrain,XTest,YTest);

optimVars = [
    optimizableVariable('SectionDepth',[1 2],'Type','integer')%lstm层数（设置取值范围为1-5，可自行修改）
    optimizableVariable('InitialLearnRate',[1e-5 1],'Transform','log')%学习率（设置取值范围为0.0001-1，可自行修改）
    optimizableVariable('Epoch',[100 500],'Type','integer')%训练轮数（设置取值范围为50-100，可自行修改）
    optimizableVariable('miniBatchSize',[10 90],'Type','integer')%设置最小批尺寸（设置取值范围为20-80，可自行修改）
    optimizableVariable('numHiddenUnits',[20 120],'Type','integer')];%隐含层单元数量（设置取值范围为50-100，可自行修改）

ObjFcn = makeObjFcn(XTrain,YTrain,XTest,YTest);

BayesObject = bayesopt(ObjFcn,optimVars,...
    'MaxTime',10*60*60, ...   %设置优化的最长时间
    'MaxObjectiveEvaluations',200, ...%设置优化的次数
    'IsObjectiveDeterministic',false, ...
    'UseParallel',false);

%贝叶斯优化的目标函数（本课题是使得测试数据的相对误差最小）
function ObjFcn = makeObjFcn(XTrain,YTrain,XTest,YTest)
ObjFcn = @valErrorFun;
    function [valError,cons,fileName] = valErrorFun(optVars)
        %层设计
        numFeatures = 8;         %输入的尺寸（按最长降雨时间）
        numResponses = 40;     %输出的尺寸（231×125）的场地
        layers = [ ...
            sequenceInputLayer(numFeatures)
            convBlock(optVars.numHiddenUnits,optVars.SectionDepth)%根据定义的隐含层单元数量与层数建立网络
            fullyConnectedLayer(numResponses)
            regressionLayer];
        %网络参数
        options = trainingOptions('adam', ...
            'MaxEpochs',optVars.Epoch, ...
            'GradientThreshold',1, ...
            'InitialLearnRate',optVars.InitialLearnRate, ...
            'MiniBatchSize',optVars.miniBatchSize, ...
            'LearnRateSchedule','piecewise', ...
            'LearnRateDropPeriod',125, ...
            'LearnRateDropFactor',0.2, ...
            'SequenceLength','longest',...
            'Verbose',0);
        %网路训练
        [net,training_process] = trainNetwork(XTrain,YTrain,layers,options);
        %测试网络
        result=predict(net,XTest);
        r2   = 1 - sum((YTest - result).^2, 'all') / sum((YTest - mean(YTest, 'all')).^2, 'all');
        valError=mean(mean(abs(result-YTest)./abs(YTest)));
        fileName = ['r2', num2str(r2, '%.4f'), '_err_', num2str(valError, '%.4f'), '.mat'];
        save(fileName,'training_process','net','valError','options','layers');
        cons = [];
    end
end

function layers = convBlock(numHiddenUnits,numLstmLayers)  %建立lstm的层
layers=[];
for Num_layer=1:numLstmLayers
    layers_single = [
        batchNormalizationLayer
        lstmLayer(numHiddenUnits)
        leakyReluLayer];
        layers=[layers;layers_single];
end
%layers = repmat(layers,numConvLayers,1);
end