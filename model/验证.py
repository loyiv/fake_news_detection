import torch
from config import parser
import sys
sys.path.append("/root/fake_news/PrefFEND-master-master")

from PrefFEND import PrefFEND
# 初始化参数（确保这些参数与训练时一致）
args = parser.parse_args([])
# 根据需要手动设置一些参数，例如：
args.dataset = 'gossip'
args.use_preference_map = False
args.use_pattern_based_model = True
args.use_fact_based_model = True
args.pattern_based_model = 'BiLSTM'
args.fact_based_model = 'DeClarE'
# 其它必要参数也请设置，确保和训练时一致
args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化模型
model = PrefFEND(args)
model = model.to(args.device)

# 加载 checkpoint
checkpoint_path = 'fake_news/Pref-FEND-master-master/model/ckpts/BiLSTM+DeClarE/4.pt'
checkpoint = torch.load(checkpoint_path, map_location=args.device)
model.load_state_dict(checkpoint['state_dict'])
model.eval()  # 设置评估模式
# 现在可以使用 model 进行推理或验证，例如：
# outputs = model(输入数据)
