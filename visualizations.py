import HP
from utils import *
from tqdm import tqdm
from models import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def compute_embeddings(model):
    _ ,test_loader = load_train_test()
    model.eval()
    model.to(HP.DEVICE)

    embeddings = []
    labels = []

    with torch.no_grad():
        for images, b_labels in tqdm(test_loader):
            images = images.to(HP.DEVICE)
            Y = model.encode(images)
            embeddings.append(Y)
            labels.append(b_labels)

    return torch.cat(embeddings).numpy(), torch.cat(labels).numpy()

def pca_rep(embeddings):
    pca = PCA(n_components=2)
    reps_pca = pca.fit_transform(embeddings)
    return reps_pca

def tsne_rep(embeddings):
    tsne = TSNE(n_components=2)
    reps_tsne = tsne.fit_transform(embeddings)
    return reps_tsne

def plot_2d(reps_2d, labels, title):
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(reps_2d[:, 0], reps_2d[:, 1], c=labels, cmap='tab10', s=5, alpha=0.7)
    # plt.legend(*scatter.legend_elements(), title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    # Get unique classes from the labels
    unique_labels = np.unique(labels)

    # Create a legend with class names
    handles = []
    class_names = []
    for class_id in unique_labels:
        handles.append(plt.Line2D([], [], marker='o', color='w',
                                  markerfacecolor=plt.cm.tab10(class_id / 10),
                                  markersize=6))
        class_names.append(cifar10_labels[class_id])

    plt.legend(handles, class_names, title="Classes", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title(title)
    plt.tight_layout()
    plt.show()

def main():
    model = load_model("VicReg_30.pth")
    embeddings, labels = compute_embeddings(model)
    plot_2d(pca_rep(embeddings), labels, "PCA representation")
    plot_2d(tsne_rep(embeddings), labels, "TSNE representation")

def main_ablVar():
    model = load_model("VicReg_ablVar30.pth")
    embeddings, labels = compute_embeddings(model)
    plot_2d(pca_rep(embeddings), labels, "PCA representation - Ablate Var")
    plot_2d(tsne_rep(embeddings), labels, "TSNE representation - Ablate Var")

