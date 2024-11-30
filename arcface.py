from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
import numpy as np
import os
from insightface.model_zoo import get_model

from sklearn.datasets import fetch_lfw_people

# 默认获取经过对齐的图像数据

data_dir = '~/scikit_learn_data/lfw_mtcnn/lfw_funneled'

batch_size = 32
epochs = 8
workers = 0 if os.name == 'nt' else 8

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('在该设备上运行: {}'.format(device))

mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
dataset = datasets.ImageFolder(data_dir, transform=transforms.Resize((512, 512)))
dataset.samples = [
    (p, p.replace(data_dir, data_dir + '_cropped'))
        for p, _ in dataset.samples
]
        
loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    collate_fn=training.collate_pil
)
"""
for i, (x, y) in enumerate(loader):
    mtcnn(x, save_path=y)
    print('\r第 {} 批，共 {} 批'.format(i + 1, len(loader)), end='')
""" 
# Remove mtcnn to reduce GPU memory usage
del mtcnn

resnet = get_model('arcface_r100_v1')  # 使用 ResNet-100 的预训练 ArcFace 模型
resnet.load_state_dict(torch.load('model.onnx'))  # 替换为你的模型路径
resnet.to(device)
resnet.head = torch.nn.Linear(resnet.embedding_size, len(dataset.class_to_idx))
resnet.eval()

class ArcFaceLoss(torch.nn.Module):
    def __init__(self, s=30.0, m=0.50, num_classes=1000, embedding_size=512):
        super().__init__()
        self.s = s  # 放缩因子
        self.m = m  # 角度间隔
        self.weight = torch.nn.Parameter(torch.FloatTensor(num_classes, embedding_size))
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        cosine = torch.nn.functional.linear(torch.nn.functional.normalize(embeddings), 
                                            torch.nn.functional.normalize(self.weight))
        phi = cosine - self.m
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output

optimizer = optim.Adam(resnet.parameters(), lr=0.001)
scheduler = MultiStepLR(optimizer, [5, 10])

trans = transforms.Compose([
    np.float32,
    transforms.ToTensor(),
    fixed_image_standardization
])
dataset = datasets.ImageFolder(data_dir, transform=trans)
img_inds = np.arange(len(dataset))
np.random.shuffle(img_inds)
train_inds = img_inds[:int(0.8 * len(img_inds))]
val_inds = img_inds[int(0.8 * len(img_inds)):]

train_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
)
val_loader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(val_inds)
)

loss_fn = ArcFaceLoss(s=30.0, m=0.50, num_classes=len(dataset.class_to_idx))
metrics = {
    'fps': training.BatchTimer(),
    'acc': training.accuracy
}

writer = SummaryWriter()
writer.iteration, writer.interval = 0, 10

print('\n\n初始化')
print('-' * 10)
resnet.eval()
training.pass_epoch(
    resnet, loss_fn, val_loader,
    batch_metrics=metrics, show_running=True, device=device,
    writer=writer
)

for epoch in range(epochs):
    print('\n循环 {}/{}'.format(epoch + 1, epochs))
    print('-' * 10)

    resnet.train()
    training.pass_epoch(
        resnet, loss_fn, train_loader, optimizer, scheduler,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, val_loader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

writer.close()