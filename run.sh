# 从头训练
python3 -u train.py --output "../models/model_1100" >> output.log
# 接续训练
python3 -u train.py --resume true --from_state "../models/model_1100/checkpoint/latest.pth" --output "../models/model_1100" >> output.log
# 验证
python3 eval.py --model "../models/model_1100/checkpoint/latest.pth" --output "../results/model_1100"