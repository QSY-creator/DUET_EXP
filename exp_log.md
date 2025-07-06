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

1:

2025-07-06 13:20:43 [INFO] ts_benchmark.data.suites.global_storage(46): Notifying all workers to sync data from the global storage.
2025-07-06 13:20:43 [INFO] ts_benchmark.data.suites.global_storage(49): Data server started.
2025-07-06 13:20:43 [INFO] ts_benchmark.models.model_loader(98): Trying to load model ts_benchmark.baselines.duet.DUET
scheduling DUET:   0%|                                                                                                                                               | 0/1 [00:00<?, ?it/s]---------------------------------------------------------- DUET
优化器分组完成:

- 36 个参数在 'ssm_experts' 组 (weight_decay=0.05)
- 5 个参数在 'gating_system' 组 (weight_decay=0.0001)
- 39 个参数在 'other_parts' 组 (weight_decay=0.0001)
  Total trainable parameters: 933524847
  scheduling DUET: 100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:10<00:00, 10.21s/it]
  collecting DUET: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 5722.11it/s]
  2025-07-06 13:20:55 [INFO] ts_benchmark.recording(148): Traceback (most recent call last):
  File "/home/featurize/work/DUET_EXP/ts_benchmark/evaluation/strategy/forecasting.py", line 54, in execute
  single_series_results = self._execute(
  ^^^^^^^^^^^^^^
  File "/home/featurize/work/DUET_EXP/ts_benchmark/evaluation/strategy/rolling_forecast.py", line 199, in _execute
  return self._eval_batch(series, meta_info, model, series_name)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/DUET_EXP/ts_benchmark/evaluation/strategy/rolling_forecast.py", line 309, in _eval_batch
  fit_method(train_valid_data, train_ratio_in_tv=train_ratio_in_tv)
  File "/home/featurize/work/DUET_EXP/ts_benchmark/baselines/duet/duet.py", line 366, in forecast_fit
  output, loss_importance = self.model(input)
  ^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
  return self._call_impl(*args, **kwargs)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
  return forward_call(*args, **kwargs)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/DUET_EXP/ts_benchmark/baselines/duet/models/duet_model.py", line 45, in forward
  reshaped_output, L_importance = self.cluster(channel_independent_input)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
  return self._call_impl(*args, **kwargs)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
  return forward_call(*args, **kwargs)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/DUET_EXP/ts_benchmark/baselines/duet/layers/linear_extractor_cluster.py", line 285, in forward
  expert_outputs = [
  ^
  File "/home/featurize/work/DUET_EXP/ts_benchmark/baselines/duet/layers/linear_extractor_cluster.py", line 286, in `<listcomp>`
  self.experts[i](expert_inputs[i]) for i in range(self.num_experts)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
  return self._call_impl(*args, **kwargs)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
  return forward_call(*args, **kwargs)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/DUET_EXP/ts_benchmark/baselines/duet/layers/linear_pattern_extractor.py", line 46, in forward
  x_proj = self.input_proj(x) # [B, L, d_model]
  ^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
  return self._call_impl(*args, **kwargs)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
  return forward_call(*args, **kwargs)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/linear.py", line 117, in forward
  return F.linear(input, self.weight, self.bias)
  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  RuntimeError: mat1 and mat2 shapes cannot be multiplied (88368x1 and 7x512)

mat1 and mat2 shapes cannot be multiplied (88368x1 and 7x512)

## 实验进度

现在在第一个实验，出现了bug,问了ai,还没调
