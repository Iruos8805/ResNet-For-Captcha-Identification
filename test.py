from torch.utils.data import DataLoader
from dataset import *

#test

def test(model, device):
    test_dataset = CustomDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, pin_memory=True)

    model.eval()
    total_chars = 0
    correct_chars = 0

    with torch.no_grad():
        for batch_features, batch_labels in test_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)

            outputs = model(batch_features)  # [batch_size, captcha_length, num_classes]
            predictions = torch.argmax(outputs, dim=2)  # [batch_size, captcha_length]

            correct_chars += (predictions == batch_labels).sum().item()
            total_chars += batch_labels.numel()

    char_accuracy = correct_chars / total_chars
    return char_accuracy
