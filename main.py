import torch
from dataset import CustomDataset, images, labels
from train import train_model
from test import test
from utils import *

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('Starting execution')

    # Train model and collect metrics
    model, train_losses, val_losses, train_char_acc, val_char_acc = train_model(device=device)

    # Evaluate on test set
    test_char_acc = test(model, device=device)

    print('-----------------------------------')
    print(f"Test Character-wise Accuracy: {test_char_acc:.4f}")
    print('-----------------------------------')

    # Plot training curves and test accuracy
    plot_train_test_results(train_losses, val_losses, train_char_acc, val_char_acc, test_char_acc)

    # a random prediction
    predict_random_sample(model, device)

if __name__ == "__main__":
    main()
