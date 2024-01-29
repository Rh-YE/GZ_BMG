import torchvision.transforms as transforms


def get_data_config():
    return {
        'save_dir': "/data/public/renhaoye/mgs/paper_model/swin_mapping/",  # 20231031_训练了BGS的数据集，加了mapping layer
        'train_file': "/data/public/renhaoye/mgs/train_1009.txt", # 20231009_训练了BGS的数据集
        'valid_file': "/data/public/renhaoye/mgs/valid_1009.txt",
        'epochs': 100,
        'batch_size': 256,
        'patience': 15,
        'dropout_rate': 0.3,
        'WORKERS': 128,
        'transfer': transforms.Compose([
            transforms.RandomRotation(degrees=(0, 180)),
            transforms.RandomHorizontalFlip(p=0.5),
            # transforms.RandomResizedCrop(size=256, scale=(1.0, 2.0)),
            # transforms.AugMix,
            transforms.ToTensor(),
        ]),
        'lr': 0.01,
        'weight_decay': 0.01,
        'betas': (0.9, 0.999),
        'phase': "training",
        'sample': 1,
        "T_max": 100
    }
