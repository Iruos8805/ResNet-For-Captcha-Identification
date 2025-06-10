import matplotlib.pyplot as plt
import torch
import string
import random
from dataset import *
from PIL import Image

# Character set
characters = string.ascii_lowercase + string.digits

def plot_train_test_results(train_losses, val_losses, train_char_accs, val_char_accs, test_char_acc):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 6))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, marker='o', label='Train Loss')
    plt.plot(epochs, val_losses, marker='x', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss over Epochs')
    plt.grid(True)
    plt.legend()

    # Plot Character-wise Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_char_accs, marker='o', color='green', label='Train Char Accuracy')
    plt.plot(epochs, val_char_accs, marker='x', color='blue', label='Validation Char Accuracy')
    plt.axhline(y=test_char_acc, color='red', linestyle='--', label=f'Test Char Accuracy: {test_char_acc:.4f}')
    plt.xlabel('Epoch')
    plt.ylabel('Character-wise Accuracy')
    plt.title('Character-wise Accuracy over Epochs')
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()


def predict_random_sample(model, device):
    """
    Randomly selects a sample from the dataset, predicts using the model,
    decodes the output indices to characters, and prints prediction vs ground truth.
    """
    model.eval()
    dataset = CustomDataset(images, labels)

    idx = random.randint(0, len(dataset) - 1)
    image_tensor, label_tensor = dataset[idx]
    image_tensor = image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)  # shape: [1, captcha_length, num_classes]
        predicted_indices = torch.argmax(outputs, dim=2).squeeze(0).cpu().tolist()

    predicted_str = ''.join([characters[i] for i in predicted_indices])
    true_str = ''.join([characters[i] for i in label_tensor.tolist()])

    print('---------------------------------------------------')
    print(f"Predicted     : {predicted_str}")
    print(f"Ground Truth  : {true_str}")
    print('---------------------------------------------------')
