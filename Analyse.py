import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import run_

def plot_learning_curves(loss_history, acc_history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss_history, label='Training Loss')
    plt.title('Loss Curve (Konvergenz)')
    plt.xlabel('Steps (x50)')
    plt.ylabel('Cross Entropy Loss')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    plt.plot(acc_history, color='orange', label='Training Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Steps (x50)')
    plt.ylabel('Accuracy (0-1)')
    plt.axhline(y=0.5, color='r', linestyle='--', label='Zufall (50%)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    # plt.savefig('learning_curves.png', dpi=500)
    plt.show()

def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Vertikal', 'Horizontal'],
                yticklabels=['Vertikal', 'Horizontal'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    # plt.savefig('confusion_matrix.png', dpi=500)
    plt.show()

def plot_kernels(conv_layer):
    fig, axes = plt.subplots(1, 8, figsize=(15, 3))
    fig.suptitle('Gelernte Convolutional Kernels (3x3)', fontsize=16)

    for i, ax in enumerate(axes):
        kernel = conv_layer.filters[i]
        ax.imshow(kernel, cmap='gray')
        ax.axis('off')
        ax.set_title(f'Filter {i + 1}')
    
    # plt.savefig('learned_kernels.png', dpi=500)
    plt.show()

def plot_sample_images(X, y):
    """
    Visualisiert Beispiele für die Input-Daten (Vertikale und Horizontale Linien).
    """
    fig, axes = plt.subplots(1, 6, figsize=(12, 3))
    fig.suptitle('Beispielhafte Input-Bilder', fontsize=16)
    
    # Wir suchen uns ein paar Beispiele
    # Indizes für Vertikal (0) und Horizontal (1)
    indices_v = [i for i, label in enumerate(y) if label == 0][:3]
    indices_h = [i for i, label in enumerate(y) if label == 1][:3]
    
    plot_indices = indices_v + indices_h
    
    for i, idx in enumerate(plot_indices):
        ax = axes[i]
        ax.imshow(X[idx], cmap='gray')
        ax.axis('off')
        label_name = "Vertikal" if y[idx] == 0 else "Horizontal"
        ax.set_title(label_name)
    
    # plt.savefig('sample_images.png', dpi=500)    
    plt.show()


if __name__ == "__main__":
    print("Führe Training aus (via Tests.py)...")
    # Training ausführen und Ergebnisse holen
    loss_history, acc_history, X_test, y_test, y_pred, conv_layer = run_.run_experiment()
    
    print("\nStarte Visualisierung...")
    plot_learning_curves(loss_history, acc_history)
    plot_confusion_matrix(y_test, y_pred)
    plot_sample_images(X_test, y_test)
    plot_kernels(conv_layer)
