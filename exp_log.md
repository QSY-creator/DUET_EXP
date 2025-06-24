分支名称：exp_ccm_try1

实验方案：把时间聚类部分换为ccm，具体而言是将expert的季节特征提取/趋势特征提取部分换成ccm，后面然后同样采取掩码，同样进行transformer

一些思考：

1.不知道ccm和原本的有什么不一样，ccm强调说是面向通道的，但是原本的不也是btn->(bn)l1?

2.另外，这个不知道ccm要不要用transformer,好像不用？

遇到的报错：

cluster_emb.shape torch.Size([7, 4, 512])
EarlyStopping counter: 5 out of 5
cluster_emb.shape torch.Size([1, 4, 512])
scheduling DUET: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [02:24<00:00, 144.78s/it]
collecting DUET: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 7653.84it/s]
2025-06-23 13:16:07 [INFO] ts_benchmark.recording(148): Traceback (most recent call last):
  File "/home/featurize/work/DUET_EXP/ts_benchmark/evaluation/strategy/forecasting.py", line 54, in execute
    single_series_results = self._execute(
                            ^^^^^^^^^^^^^^
  File "/home/featurize/work/DUET_EXP/ts_benchmark/evaluation/strategy/fixed_forecast.py", line 78, in _execute
    inference_data = pd.DataFrame(
                     ^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/pandas/core/frame.py", line 722, in __init__
    mgr = ndarray_to_mgr(
          ^^^^^^^^^^^^^^^
  File "/home/featurize/work/.local/lib/python3.11/site-packages/pandas/core/internals/construction.py", line 349, in ndarray_to_mgr
    _check_values_indices_shape_match(values, index, columns)
  File "/home/featurize/work/.local/lib/python3.11/site-packages/pandas/core/internals/construction.py", line 420, in _check_values_indices_shape_match
    raise ValueError(f"Shape of passed values is {passed}, indices imply {implied}")
ValueError: Shape of passed values is (96, 7), indices imply (3116, 7)

Shape of passed values is (96, 7), indices imply (3116, 7)
2025-06-23 13:16:07 [INFO] ts_benchmark.recording(103): loading log file /home/featurize/work/DUET_EXP/result/ETTh1/DUET/DUET.1750684567.featurize.2988.csv.tar.gz
2025-06-23 13:16:07 [INFO] ts_benchmark.report.utils.leaderboard(162): There are 7 NaN values in the leaderboard due to a higher-than-threshold NaN ratio in the corresponding model+algorithm pairs.


解决了，估计是rolling_config的问题，设置为rollinger而不是fixed


实验结果：

etth1_96_mae：0.3926

etth1_96_mse:0.363

反思：模型应该是太复杂了，两层聚类
