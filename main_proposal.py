import datetime
import os
import random
from zoneinfo import ZoneInfo

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from models.efficientnetv2_P1 import (
    EfficientNetV2ForImageClassification,
    EfficientNetV2ForImageClassification_v3,
)
from utils.data_utils import get_args, load_image_paths, make_image_loader
from utils.train_loop import train_full_loop


def set_seed(seed=1007):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


set_seed(1007)
os.makedirs("./model_save", exist_ok=True)

now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
print("=" * 30)
print(now.strftime("%Y-%m-%d %H:%M:%S"))
print("=" * 30)

args = get_args()
class_name = args.class_name
print(f"now training for class: {class_name}")

image_paths, y = load_image_paths(class_name, data_root=args.data_root)

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=1007)
kf.get_n_splits(image_paths)

for i, (train_index, test_index) in enumerate(kf.split(image_paths, y), start=1):
    train_idx, validation_idx = train_test_split(
        train_index,
        test_size=0.25,
        shuffle=True,
        random_state=1007 + i,
        stratify=y[train_index],
    )

    print(len(train_idx), len(validation_idx), len(test_index))

    train_loader = make_image_loader(
        image_paths,
        y,
        train_idx,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = make_image_loader(
        image_paths,
        y,
        validation_idx,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = make_image_loader(
        image_paths,
        y,
        test_index,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model_dicts = {
        "efficientnetv2_pspga": EfficientNetV2ForImageClassification_v3(
            num_labels=2,
            img_size=224,
            patch_size=16,
            hidden_dim=512,
            model_variant="s",
        ),
        "efficientnetv2": EfficientNetV2ForImageClassification(
            num_labels=2,
            img_size=224,
            patch_size=16,
            hidden_dim=512,
            model_variant="s",
        ),
    }
    train_full_loop(train_loader, val_loader, test_loader, model_dicts, class_name, i, lr=1e-4)

print("finished")
