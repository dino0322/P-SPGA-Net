import os
import argparse
import random
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split


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

def get_args():
    parser = argparse.ArgumentParser(description='P-SPGA ablation runner')
    parser.add_argument('--class_name', type=str, default='abnormal', help='target class name')
    parser.add_argument('--data_root', type=str, default='./data', help='normal/abnormal data root')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader workers')
    parser.add_argument(
        '--ablation', '--models',
        dest='ablation',
        type=str,
        default='B0,A0,A1,A2,A3,A5',
        help='ablation codes to run. example: B0,A0,A1,A2,A3,A5,M3,DT,A5DT',
    )
    return parser.parse_args()


def parse_ablation_codes(value):
    codes = [code.strip().upper() for code in value.replace(';', ',').split(',') if code.strip()]
    valid_codes = {'B0', 'A0', 'A1', 'A2', 'A3', 'A4', 'A5', 'M3', 'DT', 'A5DT'}
    unknown = [code for code in codes if code not in valid_codes]
    if unknown:
        raise ValueError(f'Unknown ablation code(s): {unknown}. Valid codes: {sorted(valid_codes)}')
    return codes

from utils.data_utils import load_image_paths, make_image_loader
from utils.train_loop import train_full_loop
from models.efficientnetv2_P1 import EfficientNetV2ForImageClassification, EfficientNetV2ForImageClassification_v3
from models.efficientnetv2_m3 import EfficientNetV2PSPGA_AttnPool

import datetime
from zoneinfo import ZoneInfo


os.makedirs('./model_save', exist_ok=True)

now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
print('='*30)
print(now.strftime('%Y-%m-%d %H:%M:%S'))
print('='*30)

args = get_args()
class_name = args.class_name
selected_ablation_codes = parse_ablation_codes(args.ablation)
print(f'now ablation training for class: {class_name}')
print(f'selected ablation codes: {selected_ablation_codes}')

image_paths, y = load_image_paths(class_name, data_root=args.data_root)

kf = StratifiedKFold(n_splits=2, shuffle=True, random_state=1007)
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

    model_factories = {
        'B0': lambda: ('efficientnetv2_baseline', EfficientNetV2ForImageClassification(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='s'
        )),
        'A0': lambda: ('A0_prior_only', EfficientNetV2ForImageClassification_v3(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='s',
            use_branch=False, use_stat_gate=False, use_prior_map=True,
            use_learnable_attention=False, use_progressive_fusion=False,
        )),
        'A1': lambda: ('A1_channel_only', EfficientNetV2ForImageClassification_v3(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='s',
            use_branch=False, use_stat_gate=True, use_prior_map=False,
            use_learnable_attention=False, use_progressive_fusion=False,
        )),
        'A2': lambda: ('A2_stat_only', EfficientNetV2ForImageClassification_v3(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='s',
            use_branch=False, use_stat_gate=True, use_prior_map=True,
            use_learnable_attention=False, use_progressive_fusion=False,
        )),
        'A3': lambda: ('A3_chunk_multiscale_s01', EfficientNetV2ForImageClassification_v3(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='s',
            use_branch=True, branch_type='chunk', use_stat_gate=True, use_prior_map=True,
            use_learnable_attention=False, use_progressive_fusion=False,
            residual_scale=0.1,
        )),
        'A4': lambda: ('A4_stat_learnable', EfficientNetV2ForImageClassification_v3(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='s',
            use_branch=False, use_stat_gate=True, use_prior_map=True,
            use_learnable_attention=True, use_progressive_fusion=True,
        )),
        'A5': lambda: ('A5_chunk_full_s01', EfficientNetV2ForImageClassification_v3(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='s',
            use_branch=True, branch_type='chunk', use_stat_gate=True, use_prior_map=True,
            use_learnable_attention=True, use_progressive_fusion=True,
            residual_scale=0.1,
        )),
        'M3': lambda: ('M3_pspga_attnpool', EfficientNetV2PSPGA_AttnPool(
            num_labels=2, hidden_dim=512, model_variant='s'
        )),
        'DT': lambda: ('DT_pspga_domain_blur', EfficientNetV2ForImageClassification_v3(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='s',
            use_branch=True, branch_type='chunk', use_stat_gate=True, use_prior_map=True,
            use_learnable_attention=True, use_progressive_fusion=True,
            residual_scale=0.1,
        )),
        'A5DT': lambda: ('A5_DTL_chunk_full_s01_domain_blur_k3_s04', EfficientNetV2ForImageClassification_v3(
            num_labels=2, img_size=224, patch_size=16, hidden_dim=512, model_variant='s',
            use_branch=True, branch_type='chunk', use_stat_gate=True, use_prior_map=True,
            use_learnable_attention=True, use_progressive_fusion=True,
            residual_scale=0.1,
        )),
    }

    for code in selected_ablation_codes:
        model_name, model = model_factories[code]()
        domain_blur = code in {'DT', 'A5DT'}
        blur_kernel = 3 if code == 'A5DT' else 5
        blur_sigma = 0.4 if code == 'A5DT' else 0.8
        train_loader1 = make_image_loader(
            image_paths, y, train_idx, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, domain_blur=domain_blur,
            blur_kernel=blur_kernel, blur_sigma=blur_sigma
        )
        val_loader1 = make_image_loader(
            image_paths, y, validation_idx, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, domain_blur=domain_blur,
            blur_kernel=blur_kernel, blur_sigma=blur_sigma
        )
        test_loader1 = make_image_loader(
            image_paths, y, test_index, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, domain_blur=domain_blur,
            blur_kernel=blur_kernel, blur_sigma=blur_sigma
        )
        print(f"selected models: {[model_name]} | domain_blur={domain_blur} | blur_kernel={blur_kernel} | blur_sigma={blur_sigma}")
        train_full_loop(train_loader1, val_loader1, test_loader1, {model_name: model}, class_name, i, lr=1e-4)

print('finished')
