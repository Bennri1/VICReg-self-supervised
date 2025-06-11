#utils for the questions
import torch
from augmentations import *
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader
import HP
from tqdm import tqdm
import torch.nn.functional as F

def load_train_test(shuffle= True, transform=test_transform): #using only test_transform. train transform is used only for training the encoder
    train_dataset = CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root='./data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=HP.B_SIZE, shuffle=shuffle)
    test_loader = DataLoader(test_dataset, batch_size=HP.B_SIZE, shuffle=False)
    return train_loader, test_loader

def load_model(model_path = "VicReg_30.pth" ):
    model = torch.load(model_path, weights_only=False, map_location=torch.device(HP.DEVICE))
    return model


def extract_representations(encoder, dataloader, return_original=False):
    encoder.eval()
    all_representations, all_labels, all_images = [], [], []
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x = x.to(HP.DEVICE)
            features = encoder(x)
            all_representations.append(features)
            all_labels.append(y)
            if return_original:
                for img in x:
                    all_images.append(inv_test_transform(img))
    if return_original:
        return torch.cat(all_representations), torch.cat(all_labels), torch.stack(all_images)
    return torch.cat(all_representations), torch.cat(all_labels)

def compute_topk_neigbors_indices(representations, k=3, largest=False): #largest = True if we want the farthest representations
    N = representations.size(0)
    neighbors_indices = []

    for i in range(N):
        dists = ((representations[i] - representations) ** 2).sum(dim=1)
        if not largest:
            dists[i] = float('inf')  # exclude self
        topk = torch.topk(dists, k, largest=largest).indices
        neighbors_indices.append(topk)

    return torch.stack(neighbors_indices)

def get_topk_neighbors(representations, k, largest=False):
    indices = compute_topk_neigbors_indices(representations, k=k, largest=largest)
    neighbors = [representations[i] for i in indices]
    return torch.stack(neighbors)

def invariance(Z, Z_tag):
    return F.mse_loss(Z, Z_tag)

def variance(Z):
    std = torch.sqrt(Z.var(dim=0) + HP.EPSILON)
    return torch.mean(F.relu(HP.GAMMA - std))

def covariance(Z):
    B, D = Z.size()
    Z = Z - Z.mean(dim=0)
    cov = (Z.T @ Z) / (B - 1)
    # off_diag = cov - torch.diag(torch.diag(cov))
    # return (off_diag ** 2).sum() / D
    return cov.fill_diagonal_(0.0).pow(2).sum() / D


def vicreg_loss(Z, Z_tag):
    """returns the 3 components of vicreg loss"""
    return HP.LAMBD * invariance(Z, Z_tag), HP.MU *(variance(Z) + variance(Z_tag)), HP.NU * (covariance(Z)+covariance(Z_tag))

class CIFAR10Pair(CIFAR10):
    def __init__(self, root, train=True, download=False, transform=None):
      super().__init__(root=root, train=train, transform=None, download=download)
      self._transform = transform
#
    def __getitem__(self, index):
        img, _ = super().__getitem__(index)
        return self._transform(img), self._transform(img)


cifar10_labels = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck"
}