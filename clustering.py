"""
This part is optional, so I allowed myself to have more disorganized code. Sorry for the reviewer.
"""

from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from utils import *


def main():
    vicreg_model = torch.load("VicReg_30.pth")

    train_loader, test_loader = load_train_test(shuffle=False)
    reps_q1, train_labels, train_images = extract_representations(vicreg_model.f, train_loader,return_original=True)

    reps_q1_np = reps_q1.numpy()
    np.random.seed(1)

    kmeans = KMeans(n_clusters=10, random_state=0)

    cluster_labels_q1 = kmeans.fit_predict(reps_q1_np)

    cluster_centers = kmeans.cluster_centers_

    reps_plus_centers = np.vstack([reps_q1_np, cluster_centers])
    reps_plus_centers_2d = TSNE(n_components=2, random_state=0).fit_transform(reps_plus_centers)
    reps_2d = reps_plus_centers_2d[:-10]
    centers_2d = reps_plus_centers_2d[-10:]


    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (i) Plot colored by cluster index
    scatter1 = axes[0].scatter(reps_2d[:, 0], reps_2d[:, 1], c=cluster_labels_q1, cmap='tab10', s=10)
    axes[0].scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', marker='x', s=100)
    axes[0].set_title("T-SNE Colored by Cluster Index")
    axes[0].axis('off')

    # (ii) Plot colored by actual class index
    scatter2 = axes[1].scatter(reps_2d[:, 0], reps_2d[:, 1], c=train_labels, cmap='tab10', s=10)
    axes[1].scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', marker='x', s=100)
    axes[1].set_title("T-SNE Colored by True Class")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    sil_score = silhouette_score(reps_q1_np, cluster_labels_q1)
    print(f"VICReg Silhouette Score: {sil_score:.4f}")

    no_neighbors_model = torch.load("VicReg_noNeighbors_9.pth")
    reps_q5, train_labels = extract_representations(no_neighbors_model.f, train_loader)
    reps_q5_np = reps_q5.numpy()
    kmeans = KMeans(n_clusters=10, random_state=0)

    cluster_labels_q5 = kmeans.fit_predict(reps_q5_np)

    cluster_centers = kmeans.cluster_centers_  # shape (10, 128)

    reps_plus_centers = np.vstack([reps_q5_np, cluster_centers])
    reps_plus_centers_2d = TSNE(n_components=2, random_state=0).fit_transform(reps_plus_centers)
    reps_2d = reps_plus_centers_2d[:-10]
    centers_2d = reps_plus_centers_2d[-10:]


    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # (i) Plot colored by cluster index
    scatter1 = axes[0].scatter(reps_2d[:, 0], reps_2d[:, 1], c=cluster_labels_q5, cmap='tab10', s=10)
    axes[0].scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', marker='x', s=100)
    axes[0].set_title("T-SNE Colored by Cluster Index")
    axes[0].axis('off')

    # (ii) Plot colored by actual class index
    scatter2 = axes[1].scatter(reps_2d[:, 0], reps_2d[:, 1], c=train_labels, cmap='tab10', s=10)
    axes[1].scatter(centers_2d[:, 0], centers_2d[:, 1], c='black', marker='x', s=100)
    axes[1].set_title("T-SNE Colored by True Class")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()
    sil_score = silhouette_score(reps_q5_np, cluster_labels_q5)
    print(f"VICREG no gen neighbors Silhouette Score: {sil_score:.4f}")
    return