import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import argparse
import numpy as np
from sklearn.metrics import roc_curve, auc 
import pandas as pd  
import matplotlib.pyplot as plt  
##from keras.utils import to_categorical # type: ignore[import] 
from sklearn.preprocessing import label_binarize
import seaborn as sns
from sklearn.metrics import confusion_matrix

def get_args():
    parser = argparse.ArgumentParser(description='binary class runner')
    parser.add_argument('--class_name', type=str, default='abnormal', help='target class folder name')
    parser.add_argument('--data_root', type=str, default='./data', help='root containing class/normal and class/abnormal')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader worker count')
    args = parser.parse_args()
    return args

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}


class BinaryImagePathDataset(Dataset):
    def __init__(self, paths, labels, img_size=(256, 256), domain_blur=False, blur_kernel=5, blur_sigma=0.8):
        self.paths = [str(path) for path in paths]
        self.labels = [int(label) for label in labels]
        self.img_size = img_size
        self.domain_blur = domain_blur
        self.blur_kernel = (blur_kernel, blur_kernel)
        self.blur_sigma = blur_sigma

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image_path = self.paths[idx]
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f'Failed to read image: {image_path}')
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.img_size)
        if self.domain_blur:
            ycrcb = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
            ycrcb[:, :, 0] = cv2.GaussianBlur(ycrcb[:, :, 0], self.blur_kernel, self.blur_sigma)
            img = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
        img = np.asarray(img, dtype=np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1)).copy()
        return torch.from_numpy(img), torch.tensor(self.labels[idx], dtype=torch.long)


def load_image_paths(class_name, data_root='./data'):
    paths, labels = [], []
    data_path = os.path.join(data_root, class_name)

    for cls_name in ['normal', 'abnormal']:
        cls_path = os.path.join(data_path, cls_name)
        label = 0 if cls_name == 'normal' else 1
        if not os.path.isdir(cls_path):
            raise FileNotFoundError(f'Missing class folder: {cls_path}')

        for file_name in sorted(os.listdir(cls_path)):
            image_path = os.path.join(cls_path, file_name)
            if os.path.isfile(image_path) and os.path.splitext(file_name)[1].lower() in IMAGE_EXTENSIONS:
                paths.append(image_path)
                labels.append(label)

    return np.array(paths, dtype=object), np.array(labels, dtype=np.int64)


def make_image_loader(paths, labels, indices, batch_size=32, shuffle=False, img_size=(256, 256), num_workers=2, domain_blur=False, blur_kernel=5, blur_sigma=0.8):
    dataset = BinaryImagePathDataset(
        paths[indices], labels[indices], img_size=img_size,
        domain_blur=domain_blur, blur_kernel=blur_kernel, blur_sigma=blur_sigma
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def load_data(class_name, img_size=(256, 256), device='cpu', data_root='./data'):
    X, y = [], []
    data_path = os.path.join(data_root, class_name)

    for cls_name in ['normal', 'abnormal']:
        cls_path = os.path.join(data_path, cls_name)
        label = 0 if cls_name == 'normal' else 1

        for file_name in os.listdir(cls_path):
            image_path = os.path.join(cls_path, file_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, img_size)
            X.append(img)
            y.append(label)
 
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int64)
 
    X = np.transpose(X, (0, 3, 1, 2))
 
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    return X_tensor, y_tensor

def save_csv(model_name, acc, loss, recall, prec, f1, auc, class_name, time) : 
    acc = f'{acc[0]:.3f}({acc[1]:.3f}-{acc[2]:.3f})'
    loss = f'{loss[0]:.3f}({loss[1]:.3f}-{loss[2]:.3f})'
    recall = f'{recall[0]:.3f}({recall[1]:.3f}-{recall[2]:.3f})'
    prec = f'{prec[0]:.3f}({prec[1]:.3f}-{prec[2]:.3f})'
    f1 = f'{f1[0]:.3f}({f1[1]:.3f}-{f1[2]:.3f})'
    if auc is not None: 
        auc = f'{auc[0]:.3f}({auc[1]:.3f}-{auc[2]:.3f})'
    else:
        auc = 'Nan(Nan-Nan)'
    time = time.strftime("%Y-%m-%d %H:%M:%S")

    if not os.path.exists(f'./results/{class_name}/metrics.csv'):
        os.makedirs(f'./results/{class_name}', exist_ok=True)

    header = ['model name', 'Accuracy', 'Loss', 'Recall', 'Precision', 'F1 score', 'AUROC', 'timestamp']
    new_row = [model_name, acc, loss, recall, prec, f1, auc, time]

    csv_filename = f'./results/{class_name}/metrics.csv'

    if not os.path.exists(csv_filename):
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)
        df = pd.DataFrame([new_row], columns=header)
        df.to_csv(csv_filename, index=False, header=True)
    else:
        df = pd.read_csv(csv_filename)
    if model_name in df['model name'].values:
        df.loc[df['model name'] == model_name, ['Accuracy', 'Loss', 'Recall', 'Precision', 'F1 score', 'AUROC', 'timestamp']] = new_row[1:]
    else:
        df = pd.concat([df, pd.DataFrame([new_row], columns=header)], ignore_index=True)
    df.to_csv(csv_filename, index=False, header=True) 

def plot_confusion_matrix(y_test, y_pred, model_name, class_name, fold_num) :
    if y_pred.ndim > 1 and y_pred.shape[1] > 1:
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_pred_labels = y_pred 
    class_names = ['Normal', class_name] 

    confusion = confusion_matrix(y_test, y_pred_labels, labels=[0, 1])

    cm_df = pd.DataFrame(confusion, index=class_names, columns=class_names)

    plt.figure(figsize=(10,8))
    sns.heatmap(cm_df, annot=True, cmap='Purples', fmt='d', cbar=False, annot_kws={"size": 60})
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    os.makedirs(f'./results/{class_name}/cm', exist_ok=True)
    plt.savefig(f"./results/{class_name}/cm/{model_name}_{fold_num}.png")
    plt.close()
def roc_plot(y_test, y_prob, model_name, class_name, fold_num, n_classes=None):
    if y_prob.ndim == 1:
        y_prob = np.stack([1 - y_prob, y_prob], axis=1)  # shape: [N, 2]

    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red']
    class_labels = ['Normal', class_name]

    roc_defined = len(np.unique(y_test)) >= 2
    if not roc_defined:
        plt.text(0.5, 0.5, 'ROC undefined: only one class in y_true', ha='center', va='center')
    else:
        #y_test_oh = to_categorical(y_test, num_classes=2)
        y_test_oh = label_binarize(y_test, classes=[0, 1])
        y_test_oh = np.concatenate([1 - y_test_oh, y_test_oh], axis=1)

        for i in range(2):
            fpr, tpr, _ = roc_curve(y_test_oh[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2, label=f'{class_labels[i]} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    if roc_defined:
        plt.legend(loc='lower right')

    os.makedirs(f'./results/{class_name}/roc', exist_ok=True)
    plt.savefig(f"./results/{class_name}/roc/{model_name}_{fold_num}.png")
    plt.close()
