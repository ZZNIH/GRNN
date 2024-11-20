clear all
[data, headers] = xlsread('GD_REYB1.xlsx', 'sheet1');
disp(headers);
disp(data);
save('GD_REYB1.mat', 'data');
load('GD_REYB1.mat');

features = data(:, 1:end-1);  % 提取特征列
targets = data(:, end);  % 提取目标列
rng('default');  % 设置随机数生成器的种子，以确保可重复性
splitRatio = 0.7;  % 70% 的数据用于训练，30% 用于测试
idx = randperm(size(features, 1));

trainIdx = idx(1:round(splitRatio * end));
testIdx = idx(round(splitRatio * end) + 1:end);

% 获取训练集和测试集
trainFeatures = features(trainIdx, :);
trainTargets = targets(trainIdx, :);
testFeatures = features(testIdx, :);
testTargets = targets(testIdx, :);

desired_spread = [];
mse_max = 10e20;
desired_input = [];
desired_output = [];
result_perfp = [];
indices = crossvalind('Kfold', length(trainFeatures), 4);
h = waitbar(0, '正在寻找最优化参数....');
k = 1;

%寻找最佳平滑参数
for i = 1:4
    perfp = [];
    disp(['以下为第', num2str(i), '次交叉验证结果'])
    test = (indices == i); 
    train = ~test;
    p_cv_train = trainFeatures(train, :);
    t_cv_train = trainTargets(train, :);
    p_cv_test = trainFeatures(test, :);
    t_cv_test = trainTargets(test, :);

    for spread = 0.1:0.1:2
        net = newgrnn(p_cv_train', t_cv_train', spread);
        waitbar(k/80, h);
        disp(['当前spread值为', num2str(spread)]);
        test_Out = sim(net, p_cv_test');
        error = t_cv_test - test_Out';
        disp(['当前网络的mse为', num2str(mse(error))])
        perfp = [perfp mse(error)];

        % 选择最佳参数
        if mse(error) < mse_max
            mse_max = mse(error);
            desired_spread = spread;
            desired_input = p_cv_train';
            desired_output = t_cv_train';
        end
        k = k + 1;
    end

    result_perfp(i, :) = perfp;
end;

close(h)
disp(['最佳spread值为', num2str(desired_spread)])
disp(['此时最佳输入值为'])
disp(desired_input)
disp(['此时最佳输出值为'])
disp(desired_output)

% 使用最佳参数构建 GRNN 模型并训练
best_net = newgrnn(desired_input, desired_output, desired_spread);

% 预测测试集结果
predictedTargets = sim(best_net, testFeatures');

% 进行模型性能评估
mseValue = mse(testTargets - predictedTargets);
disp(['测试集的均方误差 (MSE): ', num2str(mseValue)]);

% 计算相关系数
correlation = corr(testTargets, predictedTargets');
disp(['模型预测结果与实际目标的相关系数: ', num2str(correlation)]);

% 导入新数据，假设数据存储在'NewData.xlsx'文件的'Sheet1'表中
[newData, newHeaders] = xlsread('Z_GD.xlsx', 'Sheet1');

% 将新数据保存为MAT文件
save('Z_GD.mat', 'newData', 'newHeaders');
load('Z_GD.mat');
% 预测新的结果
predictedTargetsNew = sim(best_net, newData');
disp(predictedTargetsNew);

% 四舍五入为最近的整数
roundedPredictions = round(predictedTargetsNew);

% 将结果限制在1到4的范围内
roundedPredictions = max(1, min(4, roundedPredictions));

% 显示处理后的预测结果
disp(roundedPredictions);

%行向量转换为列向量
roundedPredictions = roundedPredictions(:);

% 将转换后的列向量保存到 Excel 表格
writematrix(roundedPredictions, 'roundedPredictions.xlsx', 'Sheet', 1);
