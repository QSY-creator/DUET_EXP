分支名称：exp_ccm_try1

实验方案：把时间聚类部分换为ccm，具体而言是将expert的季节特征提取/趋势特征提取部分换成ccm，后面然后同样采取掩码，同样进行transformer

一些思考：

1.不知道ccm和原本的有什么不一样，ccm强调说是面向通道的，但是原本的不也是btn->(bn)l1?

2.另外，这个不知道ccm要不要用transformer,好像不用？


遇到的报错：


cluster_emb.shape torch.Size([10, 4, 512])

cluster_emb.shape torch.Size([17, 4, 512])

cluster_emb.shape torch.Size([15, 4, 512])

cluster_emb.shape torch.Size([16, 4, 512])

cluster_emb.shape torch.Size([16, 4, 512])

cluster_emb.shape torch.Size([21, 4, 512])

cluster_emb.shape torch.Size([11, 4, 512])

cluster_emb.shape torch.Size([19, 4, 512])

cluster_emb.shape torch.Size([13, 4, 512])

cluster_emb.shape torch.Size([21, 4, 512])

cluster_emb.shape torch.Size([11, 4, 512])

cluster_emb.shape torch.Size([20, 4, 512])

cluster_emb.shape torch.Size([12, 4, 512])

cluster_emb.shape torch.Size([22, 4, 512])

cluster_emb.shape torch.Size([10, 4, 512])

cluster_emb.shape torch.Size([18, 4, 512])

cluster_emb.shape torch.Size([7, 4, 512])

EarlyStopping counter: 5 out of 5

cluster_emb.shape torch.Size([1, 4, 512])

scheduling DUET: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [01:54<00:00, 114.06s/it]

collecting DUET: 100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 6502.80it/s]

2025-06-23 09:28:12 [INFO] ts_benchmark.recording(148): Traceback (most recent call last):

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
