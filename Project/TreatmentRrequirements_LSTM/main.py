import warnings

# 1、设置种子
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)
torch.cuda.manual_seed_all(config.seed)
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpus  # 指定GPU训练
torch.backends.cudnn.benchmark = True  # 加快卷积计算速度