from utils import *
import matplotlib.pyplot as plt

vicreg_model = load_model("VicReg_30.pth")
no_neighbors_model = load_model("VicReg_noNeighbors_9.pth")


def select_one_index_per_class(labels):
    selected_indices = []
    for class_id in range(10):
        indices = torch.where(labels == class_id)[0]
        # selected = indices[torch.randint(0, len(indices), (1,)).item()]
        selected = indices[0]
        selected_indices.append(selected)
    return selected_indices


def plot_neighbors(title, selected_indices, images, labels, neighbors_indices):
    fig, axs = plt.subplots(len(selected_indices), 6, figsize=(12, 15))
    fig.suptitle(title, fontsize=16)

    for row, idx in enumerate(selected_indices):
        anchor_img = images[idx]
        anchor_label = cifar10_labels[labels[idx].item()]
        axs[row, 0].imshow(anchor_img.permute(1, 2, 0))
        axs[row, 0].set_title(f"Original ({anchor_label})", fontsize=8)
        axs[row, 0].axis('off')

        neighbors_idxs = neighbors_indices[idx]
        for col, n_idx in enumerate(neighbors_idxs, start=1):
            axs[row, col].imshow(images[n_idx].permute(1, 2, 0))
            axs[row, col].set_title(cifar10_labels[labels[n_idx].item()], fontsize=8)
            axs[row, col].axis('off')
    plt.tight_layout()
    plt.show()


def main():
    train_loader, test_loader = load_train_test(shuffle=False)

    reps_q1, train_labels, train_images = extract_representations(vicreg_model.f, train_loader, return_original=True)
    reps_q5, _ = extract_representations(no_neighbors_model.f, train_loader)

    selected_idxs = select_one_index_per_class(train_labels)
    neighbors_indices_q1 = compute_topk_neigbors_indices(reps_q1, k=5, largest=False)
    farthest_indices_q1 = compute_topk_neigbors_indices(reps_q1, k=5, largest=True)

    neighbors_indices_q5 = compute_topk_neigbors_indices(reps_q5, k=5, largest=False)
    farthest_indices_q5 = compute_topk_neigbors_indices(reps_q5, k=5, largest=True)


    plot_neighbors("Q1 - VICReg Nearest Neighbors", selected_idxs, train_images, train_labels, neighbors_indices_q1)
    plot_neighbors("Q1 - VICReg Farthest Neighbors", selected_idxs, train_images, train_labels, farthest_indices_q1)

    plot_neighbors("Q5 – Nearest Neighbors", selected_idxs, train_images, train_labels, neighbors_indices_q5)
    plot_neighbors("Q5 - Farthest Neighbors", selected_idxs, train_images, train_labels, farthest_indices_q5)