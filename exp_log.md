# exp_log

## 分 支名称

    exp_multiscale_try2

## 实验内容/核心改动

尝试插入多尺度部分

### 具体为方案二

    在每个专家内部添加多尺度

## 实验配置

## 实验过程

## 实验结果


过程报错


(base) ➜ DUET_EXP bash ./scripts/multivariate_forecast/ETTh1_script/DUET.sh
2025-06-20 14:44:12 [INFO] ts_benchmark.data.data_source(124): Start loading 1 series in parallel
2025-06-20 14:44:12 [INFO] ts_benchmark.data.data_source(133): Data loading finished.
2025-06-20 14:44:12 [INFO] ts_benchmark.data.suites.global_storage(40): Data server starting...
2025-06-20 14:44:12 [INFO] ts_benchmark.data.suites.global_storage(41): Start sending data to the global storage.
2025-06-20 14:44:12 [INFO] ts_benchmark.data.suites.global_storage(46): Notifying all workers to sync data from the global storage.
2025-06-20 14:44:12 [INFO] ts_benchmark.data.suites.global_storage(49): Data server started.
2025-06-20 14:44:12 [INFO] ts_benchmark.models.model_loader(98): Trying to load model ts_benchmark.baselines.duet.DUET
2025-06-20 14:44:13 [WARNING] root(135): Unknown options: num_rollings, stride, tv_ratio
scheduling DUET:   0%|                                               | 0/1 [00:00<?, ?it/s]column_num为: 7
DEBUG: configs.c_out = 7
DEBUG: configs.c_out = 7
---------------------------------------------------------- DUET
Total trainable parameters: 4950117
scheduling DUET: 100%|███████████████████████████████████████| 1/1 [00:02<00:00,  2.19s/it]
collecting DUET: 100%|█████████████████████████████████████| 1/1 [00:00<00:00, 3968.12it/s]
2025-06-20 14:44:15 [INFO] ts_benchmark.recording(148): Traceback (most recent call last):
  File "/home/featurize/work/DUET_EXP/ts_benchmark/evaluation/strategy/forecasting.py", line 54, in execute
    single_series_results = self._execute(
                            ^^^^^^^^^^^^^^
  File "/home/featurize/work/DUET_EXP/ts_benchmark/evaluation/strategy/fixed_forecast.py", line 66, in _execute
    fit_method(train_valid_data, train_ratio_in_tv=train_ratio_in_tv)
  File "/home/featurize/work/DUET_EXP/ts_benchmark/baselines/duet/duet.py", line 431, in forecast_fit
    output, loss_importance = self.model(input)#输入数据进入模型，开始训练，输出结果和重要性损失权重
                              ^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/DUET_EXP/ts_benchmark/baselines/duet/models/duet_model.py", line 66, in forward
    channel_group_feature, attention = self.Channel_transformer(x=temporal_feature, attn_mask=channel_mask)
                                       ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/DUET_EXP/ts_benchmark/baselines/duet/utils/masked_attention.py", line 62, in forward
    x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/DUET_EXP/ts_benchmark/baselines/duet/utils/masked_attention.py", line 28, in forward
    new_x, attn = self.attention(
                  ^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/DUET_EXP/ts_benchmark/baselines/duet/utils/masked_attention.py", line 186, in forward
    queries = self.query_projection(queries).view(B, L, H, -1)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/linear.py", line 117, in forward
    return F.linear(input, self.weight, self.bias)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: mat1 and mat2 shapes cannot be multiplied (224x96 and 512x512)

mat1 and mat2 shapes cannot be multiplied (224x96 and 512x512)

解决方法：在主代码增加了一个线性层，将维度转为对应匹配的特征



应该是pl1,DEBUG: enc_out_list[0].shape = torch.Size([4, 512, 512])
在经过predict_layers后,DEBUG: dec_out.shape = torch.Size([4, 720, 512])
应该是pl1,经过投影工程处理后DEBUG: dec_out.shape = torch.Size([4, 720, 1])
在经过predict_layers后,DEBUG: dec_out.shape = torch.Size([4, 720, 512])
应该是pl1,经过投影工程处理后DEBUG: dec_out.shape = torch.Size([4, 720, 1])
在经过predict_layers后,DEBUG: dec_out.shape = torch.Size([4, 720, 512])
应该是pl1,经过投影工程处理后DEBUG: dec_out.shape = torch.Size([4, 720, 1])
在经过predict_layers后,DEBUG: dec_out.shape = torch.Size([4, 720, 512])
应该是pl1,经过投影工程处理后DEBUG: dec_out.shape = torch.Size([4, 720, 1])
形状应该是pl1,下采样完每一个尺度的样本形状为 torch.Size([6, 512, 1])
形状应该是pl1,下采样完每一个尺度的样本形状最终变形结果为 torch.Size([6, 512, 1])
形状应该是pl1,下采样完每一个尺度的样本形状为 torch.Size([6, 256, 1])
形状应该是pl1,下采样完每一个尺度的样本形状最终变形结果为 torch.Size([6, 256, 1])
形状应该是pl1,下采样完每一个尺度的样本形状为 torch.Size([6, 128, 1])
形状应该是pl1,下采样完每一个尺度的样本形状最终变形结果为 torch.Size([6, 128, 1])
形状应该是pl1,下采样完每一个尺度的样本形状为 torch.Size([6, 64, 1])
形状应该是pl1,下采样完每一个尺度的样本形状最终变形结果为 torch.Size([6, 64, 1])
大概是pl1,DEBUG: out_season_list[0].shape = torch.Size([6, 512, 512])
大概是pl1,DEBUG: out_trend_list[0].shape = torch.Size([6, 512, 512])
应该是什么样的形状,大概是pl1,DEBUG: out.shape = torch.Size([6, 512, 512])
应该大概是pl1,DEBUG: out.shape = torch.Size([6, 512, 512])
应该是什么样的形状,大概是pl1,DEBUG: out.shape = torch.Size([6, 256, 512])
应该大概是pl1,DEBUG: out.shape = torch.Size([6, 256, 512])
应该是什么样的形状,大概是pl1,DEBUG: out.shape = torch.Size([6, 128, 512])
应该大概是pl1,DEBUG: out.shape = torch.Size([6, 128, 512])
应该是什么样的形状,大概是pl1,DEBUG: out.shape = torch.Size([6, 64, 512])
应该大概是pl1,DEBUG: out.shape = torch.Size([6, 64, 512])
应该是pl1,DEBUG: enc_out_list[0].shape = torch.Size([6, 512, 512])
大概是pl1,DEBUG: out_season_list[0].shape = torch.Size([6, 512, 512])
大概是pl1,DEBUG: out_trend_list[0].shape = torch.Size([6, 512, 512])
应该是什么样的形状,大概是pl1,DEBUG: out.shape = torch.Size([6, 512, 512])
应该大概是pl1,DEBUG: out.shape = torch.Size([6, 512, 512])
应该是什么样的形状,大概是pl1,DEBUG: out.shape = torch.Size([6, 256, 512])
应该大概是pl1,DEBUG: out.shape = torch.Size([6, 256, 512])
应该是什么样的形状,大概是pl1,DEBUG: out.shape = torch.Size([6, 128, 512])
应该大概是pl1,DEBUG: out.shape = torch.Size([6, 128, 512])
应该是什么样的形状,大概是pl1,DEBUG: out.shape = torch.Size([6, 64, 512])
应该大概是pl1,DEBUG: out.shape = torch.Size([6, 64, 512])
应该是pl1,DEBUG: enc_out_list[0].shape = torch.Size([6, 512, 512])
在经过predict_layers后,DEBUG: dec_out.shape = torch.Size([6, 720, 512])
应该是pl1,经过投影工程处理后DEBUG: dec_out.shape = torch.Size([6, 720, 1])
在经过predict_layers后,DEBUG: dec_out.shape = torch.Size([6, 720, 512])
应该是pl1,经过投影工程处理后DEBUG: dec_out.shape = torch.Size([6, 720, 1])
在经过predict_layers后,DEBUG: dec_out.shape = torch.Size([6, 720, 512])
应该是pl1,经过投影工程处理后DEBUG: dec_out.shape = torch.Size([6, 720, 1])
在经过predict_layers后,DEBUG: dec_out.shape = torch.Size([6, 720, 512])
应该是pl1,经过投影工程处理后DEBUG: dec_out.shape = torch.Size([6, 720, 1])
形状应该是pl1,下采样完每一个尺度的样本形状为 torch.Size([4, 512, 1])
形状应该是pl1,下采样完每一个尺度的样本形状最终变形结果为 torch.Size([4, 512, 1])
形状应该是pl1,下采样完每一个尺度的样本形状为 torch.Size([4, 256, 1])
形状应该是pl1,下采样完每一个尺度的样本形状最终变形结果为 torch.Size([4, 256, 1])
形状应该是pl1,下采样完每一个尺度的样本形状为 torch.Size([4, 128, 1])
形状应该是pl1,下采样完每一个尺度的样本形状最终变形结果为 torch.Size([4, 128, 1])
形状应该是pl1,下采样完每一个尺度的样本形状为 torch.Size([4, 64, 1])
形状应该是pl1,下采样完每一个尺度的样本形状最终变形结果为 torch.Size([4, 64, 1])
大概是pl1,DEBUG: out_season_list[0].shape = torch.Size([4, 512, 512])
大概是pl1,DEBUG: out_trend_list[0].shape = torch.Size([4, 512, 512])
应该是什么样的形状,大概是pl1,DEBUG: out.shape = torch.Size([4, 512, 512])
应该大概是pl1,DEBUG: out.shape = torch.Size([4, 512, 512])
应该是什么样的形状,大概是pl1,DEBUG: out.shape = torch.Size([4, 256, 512])
应该大概是pl1,DEBUG: out.shape = torch.Size([4, 256, 512])
应该是什么样的形状,大概是pl1,DEBUG: out.shape = torch.Size([4, 128, 512])
应该大概是pl1,DEBUG: out.shape = torch.Size([4, 128, 512])
应该是什么样的形状,大概是pl1,DEBUG: out.shape = torch.Size([4, 64, 512])
应该大概是pl1,DEBUG: out.shape = torch.Size([4, 64, 512])
应该是pl1,DEBUG: enc_out_list[0].shape = torch.Size([4, 512, 512])
大概是pl1,DEBUG: out_season_list[0].shape = torch.Size([4, 512, 512])
大概是pl1,DEBUG: out_trend_list[0].shape = torch.Size([4, 512, 512])
应该是什么样的形状,大概是pl1,DEBUG: out.shape = torch.Size([4, 512, 512])
应该大概是pl1,DEBUG: out.shape = torch.Size([4, 512, 512])
应该是什么样的形状,大概是pl1,DEBUG: out.shape = torch.Size([4, 256, 512])
应该大概是pl1,DEBUG: out.shape = torch.Size([4, 256, 512])
应该是什么样的形状,大概是pl1,DEBUG: out.shape = torch.Size([4, 128, 512])
应该大概是pl1,DEBUG: out.shape = torch.Size([4, 128, 512])
应该是什么样的形状,大概是pl1,DEBUG: out.shape = torch.Size([4, 64, 512])
应该大概是pl1,DEBUG: out.shape = torch.Size([4, 64, 512])
应该是pl1,DEBUG: enc_out_list[0].shape = torch.Size([4, 512, 512])
在经过predict_layers后,DEBUG: dec_out.shape = torch.Size([4, 720, 512])
应该是pl1,经过投影工程处理后DEBUG: dec_out.shape = torch.Size([4, 720, 1])
在经过predict_layers后,DEBUG: dec_out.shape = torch.Size([4, 720, 512])
应该是pl1,经过投影工程处理后DEBUG: dec_out.shape = torch.Size([4, 720, 1])
在经过predict_layers后,DEBUG: dec_out.shape = torch.Size([4, 720, 512])
应该是pl1,经过投影工程处理后DEBUG: dec_out.shape = torch.Size([4, 720, 1])
在经过predict_layers后,DEBUG: dec_out.shape = torch.Size([4, 720, 512])
应该是pl1,经过投影工程处理后DEBUG: dec_out.shape = torch.Size([4, 720, 1])
scheduling DUET: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [26:06<00:00, 1566.96s/it]
collecting DUET: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 3379.78it/s]
2025-06-21 10:12:21 [INFO] ts_benchmark.recording(148): Traceback (most recent call last):
  File "/home/featurize/work/DUET_EXP/ts_benchmark/evaluation/strategy/forecasting.py", line 54, in execute
    single_series_results = self._execute(
                            ^^^^^^^^^^^^^^
  File "/home/featurize/work/DUET_EXP/ts_benchmark/evaluation/strategy/fixed_forecast.py", line 68, in _execute
    predicted = model.forecast(horizon, train_valid_data)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/DUET_EXP/ts_benchmark/baselines/duet/duet.py", line 506, in forecast
    output, _ = self.model(input)
                ^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/DUET_EXP/ts_benchmark/baselines/duet/models/duet_model.py", line 54, in forward
    reshaped_output, L_importance = self.cluster(channel_independent_input)
                                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1553, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/torch/nn/modules/module.py", line 1562, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/DUET_EXP/ts_benchmark/baselines/duet/layers/linear_extractor_cluster.py", line 300, in forward
    y = dispatcher.combine(expert_outputs)
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/featurize/work/DUET_EXP/ts_benchmark/baselines/duet/layers/linear_extractor_cluster.py", line 94, in combine
    stitched = torch.cat(expert_out, 0)
               ^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 1 but got size 7 for tensor number 1 in the list.

Sizes of tensors must match except in dimension 0. Expected size 1 but got size 7 for tensor number 1 in the list.
2025-06-21 10:12:21 [INFO] ts_benchmark.recording(103): loading log file /home/featurize/work/DUET_EXP/result/ETTh1/DUET/DUET.1750500741.featurize.4371.csv.tar.gz
2025-06-21 10:12:21 [INFO] ts_benchmark.report.utils.leaderboard(162): There are 7 NaN values in the leaderboard due to a higher-than-threshold NaN ratio in the corresponding model+algorithm pairs.

尚未解决

并且这样的修改，使得模型有点复杂，显存有些不足
