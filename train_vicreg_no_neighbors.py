from models import VICReg
from utils import *
import numpy as np
import matplotlib.pyplot as plt


class NeighborDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, topk_indices):
        self.dataset = dataset
        self.topk = topk_indices

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1, _ = self.dataset[idx]
        neighbor_indices = self.topk[idx]
        rand_idx = torch.randint(0, neighbor_indices.shape[0], (1,))
        neighbor_idx = neighbor_indices[rand_idx].item()
        img2, _ = self.dataset[neighbor_idx]
        return img1, img2


def train_vicreg_no_neighbors():
    cifar_dataset = CIFAR10(root='./data', train=True, download=True, transform=test_transform)
    cifar_loader = DataLoader(cifar_dataset, batch_size=HP.B_SIZE, shuffle=False)

    vicreg = load_model("/content/drive/My Drive/Ex3/VicReg_30.pth")
    encoder = vicreg.f.to(HP.DEVICE)
    for p in encoder.parameters():
        p.requires_grad = False

    representations, _ = extract_representations(encoder, cifar_loader)
    topk_neighbors_indices = compute_topk_neigbors_indices(representations, k=3)

    neighbor_dataset = NeighborDataset(cifar_dataset, topk_neighbors_indices)
    neighbor_loader = DataLoader(neighbor_dataset, batch_size=HP.B_SIZE, shuffle=True)

    del vicreg
    if HP.DEVICE == 'cuda':
        torch.cuda.empty_cache()

    model = VICReg().to(HP.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=HP.LR, betas=(0.9, 0.999), weight_decay=1e-6)


    model.train()
    batch_losses = []
    for epoch in range(10):
        for X, X_tag in tqdm(neighbor_loader):
            X, X_tag = X.to(HP.DEVICE), X_tag.to(HP.DEVICE)
            optimizer.zero_grad()
            Z, Z_tag = model(X), model(X_tag)
            inv, var, cov = vicreg_loss(Z, Z_tag)
            loss = inv + var + cov
            batch_losses.append(loss.item())
            loss.backward()
            optimizer.step()
            torch.save(model, f"/content/drive/My Drive/Ex3/VicReg_noNeighbors_{epoch}.pth")

        print(f"train loss avg until now: {sum(batch_losses)/len(batch_losses)}")
    plt.plot(batch_losses)
    return

def main():
    seed = 1
    np.random.seed(seed)
    torch.manual_seed(seed)
    train_vicreg_no_neighbors()