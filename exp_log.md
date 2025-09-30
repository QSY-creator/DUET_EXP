实验记录

实验方案：

1.把时序部分换成time_pro 的mamba

实验报错：

1.**(base) ➜ DUET_EXP bash ./scripts/multivariate_forecast/ETTh1_script/DUET.sh**
2025-09-30 07:49:21 [INFO] ts_benchmark.data.data_source(124): Start loading 1 series in parallel
2025-09-30 07:49:21 [INFO] ts_benchmark.data.data_source(133): Data loading finished.
2025-09-30 07:49:21 [INFO] ts_benchmark.data.suites.global_storage(40): Data server starting...
2025-09-30 07:49:21 [INFO] ts_benchmark.data.suites.global_storage(41): Start sending data to the global storage.
2025-09-30 07:49:21 [INFO] ts_benchmark.data.suites.global_storage(46): Notifying all workers to sync data from the global storage.
2025-09-30 07:49:21 [INFO] ts_benchmark.data.suites.global_storage(49): Data server started.
2025-09-30 07:49:21 [INFO] ts_benchmark.models.model_loader(98): Trying to load model ts_benchmark.baselines.duet.DUET
2025-09-30 07:49:21 [INFO] ts_benchmark.models.model_loader(101): Loading model ts_benchmark.baselines.duet.DUET failed
2025-09-30 07:49:21 [INFO] ts_benchmark.models.model_loader(102): Error: No module named 'TimePro'
2025-09-30 07:49:21 [INFO] ts_benchmark.models.model_loader(98): Trying to load model duet.DUET
2025-09-30 07:49:21 [INFO] ts_benchmark.models.model_loader(101): Loading model duet.DUET failed
2025-09-30 07:49:21 [INFO] ts_benchmark.models.model_loader(102): Error: No module named 'duet'
Traceback (most recent call last):
File "/home/featurize/work/DUET_EXP/./scripts/run_benchmark.py", line 340, in **`<module>`**
log_filenames = pipeline(
^^^^^^^^^
File "/home/featurize/work/DUET_EXP/ts_benchmark/pipeline.py", line 139, in pipeline
model_factory_list = get_models(model_config)
^^^^^^^^^^^^^^^^^^^^^^^^
File "/home/featurize/work/DUET_EXP/ts_benchmark/models/model_loader.py", line 221, in get_models
raise ValueError(f"Unexpected model info type {type(model_info).name}")
