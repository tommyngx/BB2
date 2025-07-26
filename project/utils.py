import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


def plot_metrics(train_losses, train_accs, test_losses, test_accs, save_path):
    """
    Plot training and test loss/accuracy with highlighted best epochs in fivethirtyeight style.

    Args:
        train_losses: List of training losses per epoch
        train_accs: List of training accuracies per epoch
        test_losses: List of test/validation losses per epoch
        test_accs: List of test/validation accuracies per epoch
        save_path: Path to save the plot (e.g., 'output/figures/model.png')
    """
    # Set epoch range
    epochs = list(range(1, len(train_losses) + 1))

    # Find best epochs
    index_loss = np.argmin(test_losses)  # Epoch with lowest test loss
    val_lowest = test_losses[index_loss]
    index_acc = np.argmax(test_accs)  # Epoch with highest test accuracy
    acc_highest = test_accs[index_acc]

    # Set plot style
    plt.style.use("fivethirtyeight")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    # Set lighter background color
    fig.patch.set_facecolor("#f7f7f7")
    for ax in axes:
        ax.set_facecolor("#f7f7f7")

    # Plot Loss
    axes[0].plot(epochs, train_losses, "r", label="Training Loss")
    axes[0].plot(epochs, test_losses, "g", label="Test Loss")
    axes[0].scatter(
        index_loss + 1,
        val_lowest,
        s=150,
        c="blue",
        label=f"Best Loss: {val_lowest:.4f} (Epoch {index_loss + 1:d})",
    )
    axes[0].set_title("Training and Test Loss")
    axes[0].set_xlabel("Epochs", color="#222831")
    axes[0].set_ylabel("Loss", color="#222831")
    axes[0].tick_params(axis="x", colors="#222831")
    axes[0].tick_params(axis="y", colors="#222831")
    axes[0].grid(True, linestyle="--", alpha=0.5, color="navy")
    legend = axes[0].legend()
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#222831")
    # Set legend text color
    for text in legend.get_texts():
        text.set_color("#222831")

    # Plot Accuracy
    axes[1].plot(epochs, train_accs, "r", label="Training Accuracy")
    axes[1].plot(epochs, test_accs, "g", label="Test Accuracy")
    axes[1].scatter(
        index_acc + 1,
        acc_highest,
        s=150,
        c="blue",
        label=f"Best Acc: {acc_highest:.4f} (Epoch {index_acc + 1:d})",
    )
    axes[1].set_title("Training and Test Accuracy")
    axes[1].set_xlabel("Epochs", color="#222831")
    axes[1].set_ylabel("Accuracy", color="#222831")
    axes[1].tick_params(axis="x", colors="#222831")
    axes[1].tick_params(axis="y", colors="#222831")
    axes[1].grid(True, linestyle="--", alpha=0.5, color="navy")
    legend = axes[1].legend()
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#222831")
    for text in legend.get_texts():
        text.set_color("#222831")

    # Thêm đường viền xung quanh khu vực plot
    for ax in axes:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(2)
            spine.set_color("#161A1F")

    # Điều chỉnh vị trí và kích thước để hai ảnh cover 95% chiều rộng, subplot phải không sát viền phải
    fig.subplots_adjust(
        left=0.07, right=0.84, wspace=0.15
    )  # giảm right xuống để subplot phải cách viền

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close()


def plot_metrics_orig(train_losses, train_accs, test_losses, test_accs, save_path):
    """
    Plot training and test loss/accuracy with highlighted best epochs in fivethirtyeight style.

    Args:
        train_losses: List of training losses per epoch
        train_accs: List of training accuracies per epoch
        test_losses: List of test/validation losses per epoch
        test_accs: List of test/validation accuracies per epoch
        save_path: Path to save the plot (e.g., 'output/figures/model.png')
    """
    # Set epoch range
    epochs = list(range(1, len(train_losses) + 1))

    # Find best epochs
    index_loss = np.argmin(test_losses)  # Epoch with lowest test loss
    val_lowest = test_losses[index_loss]
    index_acc = np.argmax(test_accs)  # Epoch with highest test accuracy
    acc_highest = test_accs[index_acc]

    # Set plot style
    plt.style.use("fivethirtyeight")
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 8))

    # Plot Loss
    axes[0].plot(epochs, train_losses, "r", label="Training Loss")
    axes[0].plot(epochs, test_losses, "g", label="Test Loss")
    axes[0].scatter(
        index_loss + 1,
        val_lowest,
        s=150,
        c="blue",
        label=f"Best Loss: {val_lowest:.4f} (Epoch {index_loss + 1:d})",
    )
    axes[0].set_title("Training and Test Loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, linestyle="--", alpha=0.5, color="white")  # grid màu trắng
    legend = axes[0].legend()
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    # Plot Accuracy
    axes[1].plot(epochs, train_accs, "r", label="Training Accuracy")
    axes[1].plot(epochs, test_accs, "g", label="Test Accuracy")
    axes[1].scatter(
        index_acc + 1,
        acc_highest,
        s=150,
        c="blue",
        label=f"Best Acc: {acc_highest:.4f} (Epoch {index_acc + 1:d})",
    )
    axes[1].set_title("Training and Test Accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, linestyle="--", alpha=0.5, color="white")  # grid màu trắng
    legend = axes[1].legend()
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("black")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    plt.close()
    # print(f"Saved training plot to {save_path}")


def plot_metrics2(train_losses, train_accs, test_losses, test_accs, save_path):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(12, 5))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, "-o", label="Train Loss")
    plt.plot(epochs, test_losses, "-o", label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("Loss per Epoch")

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, "-o", label="Train Acc")
    plt.plot(epochs, test_accs, "-o", label="Test Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.title("Accuracy per Epoch")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None, title=None):
    cm = confusion_matrix(y_true, y_pred)
    # Normalize the confusion matrix
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_normalized = np.divide(
        cm.astype("float"),
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )

    # Create an annotation matrix with both counts and percentages
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            count = cm[i, j]
            percent = cm_normalized[i, j] * 100
            annot[i, j] = f"{percent:.1f}%\n({count})"

    # Plot the heatmap
    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(
        cm_normalized,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar=True,
        vmin=0,
        vmax=1,
        annot_kws={"fontsize": 14},
    )
    # Customize the color bar
    ticks = np.linspace(0, 1, 5)
    cbar = ax.collections[0].colorbar
    cbar.set_ticks(ticks)
    cbar.ax.set_yticklabels([f"{int(t * 100)}%" for t in ticks])

    plt.xlabel("Predicted", fontsize=18)
    plt.ylabel("Actual", fontsize=18)

    plt.title(title, fontsize=20, pad=20)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix_orig(
    y_true, y_pred, class_names, save_path=None, title="Confusion Matrix", cmap="Blues"
):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.grid(False)  # Tắt grid nếu có
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm.max() / 2.0
    for i, j in np.ndindex(cm.shape):
        plt.text(
            j,
            i,
            format(cm[i, j], "d"),
            ha="center",
            va="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()
