## 分支名称

exp_mamba_try1

## 实验idea

将专家改为manba专家，对非线性的刻画更加深刻

## 实验方案

1：

将expert改为mamba

在duet中添加了静态方法：优化器分组函数，即get_optimizer_param_groups辅助函数

在训练中创建了添加差异化权重衰减的逻辑：补充了差异化权重衰减超参数的获取，调用新增的静态方法，使用了更好的优化器

将seq_len，和win_size转为由超参数调整后确定而不是完全由一开始赋值确定（虽然我感觉没必要）：在init部分初始化为None,在multi_forecasting_hyper_param_tune方法结束后，再将config.seq_len确定为self.seq_len（win_size同理）



## 实验结果

## 实验结果分析与改进

## 实验方案改进

## 改进bug积累
