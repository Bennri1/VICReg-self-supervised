from utils import *
from torchvision.datasets import MNIST
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import NearestNeighbors

def plot_roc_curve(y_true, score_dict):
    """
    score_dict: dictionary of {model_name : scores}
    """
    plt.figure(figsize=(8, 6))
    for model_name, scores in score_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.4f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.title('ROC Curve - Anomaly Detection using VICReg')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()



def compute_inverse_density(train_representations, test_representations, k=2):
    knn = NearestNeighbors(n_neighbors=k, metric='euclidean')
    knn.fit(train_representations)
    distances, _ = knn.kneighbors(test_representations)
    density = distances.mean(axis=1)
    return density

def plot_top_anomalies(scores, images, title, num_samples=7):
    top_indices = scores.argsort()[-num_samples:][::-1]
    top_images = [images[index] for index in top_indices]

    # Plot the top images
    fig, axes = plt.subplots(1, num_samples, figsize=(num_samples * 2, 2.5))
    for img, ax in zip(top_images, axes):
        # if isinstance(img, torch.Tensor):
        #     img = img.cpu()
        ax.imshow(img.permute(1, 2, 0).numpy())
        ax.axis("off")
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def main():
    vicreg_model = load_model("VicReg_30.pth")
    no_neighbors_model = load_model("VicReg_noNeighbors_9.pth")

    c10_train_loader, c10_test_loader = load_train_test(shuffle=False)
    cifar_test = CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    mnist_dataset = MNIST(root='./data', train=False, download=True, transform=mnist_transform)

    combined_test = ConcatDataset([cifar_test, mnist_dataset])
    combined_test_loader = DataLoader(combined_test, batch_size=HP.B_SIZE, shuffle=False)

    normal_labels = torch.zeros(len(cifar_test), dtype=torch.long)
    anomaly_labels = torch.ones(len(mnist_dataset), dtype=torch.long)
    combined_test_labels = torch.cat([normal_labels, anomaly_labels])

    reps_q1, train_labels = extract_representations(vicreg_model.f, c10_train_loader)
    test_reps_q1, _ = extract_representations(vicreg_model.f, combined_test_loader)

    inverse_density_scores_q1 = compute_inverse_density(reps_q1, test_reps_q1, k=2)

    reps_q5, train_labels_q5 = extract_representations(no_neighbors_model.f, c10_train_loader)
    test_reps_q5, _ = extract_representations(no_neighbors_model.f, combined_test_loader)

    inverse_density_scores_q5 = compute_inverse_density(reps_q5, test_reps_q5, k=2)

    plot_roc_curve(combined_test_labels, {"Vicreg": inverse_density_scores_q1, "No Neighbors": inverse_density_scores_q5})