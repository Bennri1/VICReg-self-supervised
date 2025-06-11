
from models import *
import matplotlib.pyplot as plt
from utils import *



def load_data():
    train_dataset = CIFAR10Pair(root='./data', train=True, download=True, transform=train_transform)
    test_dataset = CIFAR10Pair(root='./data', train=False, download=True, transform=test_transform)
    return train_dataset, test_dataset


def train_vicreg(train_dataset, test_dataset,device = HP.DEVICE, epochs= HP.N_EPOCHS, ablate_var = False):
    train_loader = DataLoader(train_dataset, batch_size=HP.B_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=HP.B_SIZE, shuffle=False)
    model = VICReg()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=HP.LR, betas = (0.9, 0.999), weight_decay = 1e-6)
    batch_train_losses = {"Invariance Loss" : [], "Variance Loss" : [], "Covariance Loss" : [], "Total Loss":[]}
    epoch_test_losses = {"Invariance Test Loss": [], "Variance Test Loss": [], "Covariance Test Loss": []}
    for epoch in range(epochs):
        model.train()
        for X, X_tag in tqdm(train_loader): #SSL - no need for labels
            X, X_tag = X.to(device), X_tag.to(device)

            optimizer.zero_grad()
            Z, Z_tag = model(X), model(X_tag)
            if ablate_var:
                inv, var , cov = HP.LAMBD * invariance(Z, Z_tag), torch.tensor(0), HP.NU * (covariance(Z)+covariance(Z_tag))
                loss = inv+cov
            else:
                inv, var, cov = vicreg_loss(Z, Z_tag)
                loss = inv + var + cov
            batch_train_losses["Invariance Loss"].append(inv.item())
            batch_train_losses["Variance Loss"].append(var.item())
            batch_train_losses["Covariance Loss"].append(cov.item())
            batch_train_losses["Total Loss"].append(loss.item())

            loss.backward()
            optimizer.step()

        if (epoch+1)%3 == 0:
            if ablate_var:
              torch.save(model, f"/content/drive/My Drive/Ex3/VicReg_ablVar{epoch+1}.pth")
            else:
              torch.save(model, f"/content/drive/My Drive/Ex3/VicReg_{epoch+1}.pth")
            for name, losses in batch_train_losses.items():
                plt.figure()
                plt.plot(losses, label=name)
                plt.legend()
            plt.show()


        
        model.eval()
        running_test_losses = [0,0,0]
        with torch.no_grad():
              for X, X_tag in tqdm(test_loader):
                  X = X.to(device)
                  X_tag = X_tag.to(device)
                  Z, Z_tag = model(X), model(X_tag)
                  inv, var, cov = vicreg_loss(Z, Z_tag)
                  running_test_losses[0] += inv.item()
                  running_test_losses[1] += var.item()
                  running_test_losses[2] += cov.item()
              epoch_test_losses["Invariance Test Loss"].append(running_test_losses[0]/len(test_loader))
              epoch_test_losses["Variance Test Loss"].append(running_test_losses[1]/len(test_loader))
              epoch_test_losses["Covariance Test Loss"].append(running_test_losses[2]/len(test_loader))
    #plotting test losses
    for name, losses in epoch_test_losses.items():
        plt.figure()
        plt.plot(losses, label=name)
        plt.legend()
        plt.show()

    return

def main_vicreg():
    # torch.set_default_dtype(torch.float32)
    seed = 1
    torch.manual_seed(seed)
    train_dataset, test_dataset = load_data()
    train_vicreg(train_dataset, test_dataset, epochs=30)
    return

def main_vicreg_ablate():
    seed = 1
    torch.manual_seed(seed)
    train_dataset, test_dataset = load_data()
    train_vicreg(train_dataset, test_dataset, epochs=30, ablate_var = True)
    return

