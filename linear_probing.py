import torch

import HP
from utils import *
import torch.nn as nn
from tqdm import tqdm



def train_linear_probe(model):
    train_loader, test_loader = load_train_test()
    encoder = model.f

    train_reprs, train_labels = extract_representations(encoder, train_loader)
    test_reprs, test_labels = extract_representations(encoder, test_loader)

    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.to(HP.DEVICE)

    linear_probe = nn.Linear(HP.ENCODE_D, 10).to(HP.DEVICE)
    optimizer = torch.optim.SGD(linear_probe.parameters(), lr=0.05)
    criterion = nn.CrossEntropyLoss()
    epochs = 10

    for epoch in range(epochs):
        linear_probe.train()
        permutation = torch.randperm(train_reprs.size(0))
        correct, total = 0, 0
        for i in tqdm(range(0, train_reprs.size(0), HP.B_SIZE)):
            indices = permutation[i:i + HP.B_SIZE]
            batch_x = train_reprs[indices].to(HP.DEVICE)
            batch_y = train_labels[indices].to(HP.DEVICE)

            preds = linear_probe(batch_x)
            loss = criterion(preds, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            correct += (preds.argmax(1) == batch_y).sum().item()
            total += batch_y.size(0)
        acc = correct / total
        print(f"Epoch {epoch + 1}/{epochs}, Train Accuracy: {acc:.4f}")

        linear_probe.eval()
        correct_tst= 0
        with torch.no_grad():
            preds = linear_probe(test_reprs)
            correct_tst += (preds.argmax(1) == test_labels).sum().item()
            acc = correct / test_reprs.size(0)
            print(f"Epoch {epoch + 1}/{epochs}, Test Accuracy: {acc:.4f}")


def main():
    model = load_model()
    train_linear_probe(model)

def main_ablVar():
    model = load_model("VicReg_ablVar30.pth")
    train_linear_probe(model)

def main_no_neighbors():
    model = load_model("VicReg_noNeighbors_9.pth")
    train_linear_probe(model)