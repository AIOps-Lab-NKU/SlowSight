import pandas as pd
df = pd.read_csv("/home/sunyuxin/Huawei-SlowSight/slowsight/backEnd/data/D3_metric_1.csv")
df = df[[col for col in df.columns if "orphan" not in col]]
df.to_csv("/home/sunyuxin/Huawei-SlowSight/slowsight/backEnd/data/D3_metric.csv", index = False)