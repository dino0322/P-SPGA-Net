import os
import random  # 추가
import numpy as np # 추가
import torch # 추가
from sklearn.model_selection import StratifiedKFold

def set_seed(seed=1007):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# seed settings
set_seed(1007)

from utils.data_utils import get_args, load_image_paths, make_image_loader
from utils.train_loop import train_full_loop
from models.models.deep_model import DeepModel, DeepSAFEModel
from models.models.deep_model_3 import AttentionEnsemble
#from models.swin_transformer import SwinTransformer, SwinTransformerSAFE
 
from sklearn.model_selection import train_test_split  

#from models.mambavision import MambaVisionForImageClassification, MambaVisionForImageClassification_v2
#from models.nextvit import NextViTForImageClassification, NextViTForImageClassification_v2
#from models.swintransformerv2 import SwinTransformerV2ForImageClassification, SwinTransformerV2ForImageClassification_v2
#from models.convnextv2 import ConvNeXtV2ForImageClassification, ConvNeXtV2ForImageClassification_v2
#from models.efficientnetv2 import EfficientNetV2ForImageClassification, EfficientNetV2ForImageClassification_v2
from models.efficientnetv2_P1 import EfficientNetV2ForImageClassification, EfficientNetV2ForImageClassification_v3

#from models.gpt_vit import GPT2ForImageClassification
#from models.bert_vit import BertForImageClassification,BertForImageClassification_v2

import datetime
from zoneinfo import ZoneInfo  



os.makedirs('./model_save', exist_ok=True)

now = datetime.datetime.now(ZoneInfo("Asia/Seoul"))
print('='*30)
print(now.strftime('%Y-%m-%d %H:%M:%S'))
print('='*30)
 
args = get_args()
class_name = args.class_name
print(f'now training for class: {class_name}')

image_paths, y = load_image_paths(class_name, data_root=args.data_root)

from sklearn.model_selection import KFold

kf = StratifiedKFold(n_splits=3,shuffle=True,random_state=1007)
kf.get_n_splits(image_paths)
for i, (train_index, test_index) in enumerate(kf.split(image_paths, y), start =1):
    train_idx, validation_idx = train_test_split(
        train_index,
        test_size=0.25,
        shuffle=True,
        random_state=1007 + i,
        stratify=y[train_index],
    )

    print(len(train_idx), len(validation_idx), len(test_index))

    train_loader1 = make_image_loader(
        image_paths, y, train_idx, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader1 = make_image_loader(
        image_paths, y, validation_idx, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader1 = make_image_loader(
        image_paths, y, test_index, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    model_dicts = {
        #'mambavision_proposal': MambaVisionForImageClassification_v2(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='tiny'),
        'efficientnetv2_pspga': EfficientNetV2ForImageClassification_v3(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='s'),
        #'mambavision': MambaVisionForImageClassification(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='tiny'),
        #'nextvit': NextViTForImageClassification(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='small'),
        'efficientnetv2': EfficientNetV2ForImageClassification(num_labels=2,img_size=224,patch_size=16,hidden_dim=512,model_variant='s'),
        #'Resnet50': DeepModel('ResNet50'),
        #'DenseNet121': DeepModel('DenseNet121'),
    }
    train_full_loop(train_loader1, val_loader1, test_loader1, model_dicts, class_name, i, lr=1e-4)

print('finished')
