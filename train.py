import torch
import torch.nn as nn
import torch.optim as optim
from dataset import *
from Resnet_model_complete import *
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

#train

def train_model(device):
    epochs = 12
    batch_size = 32
    lr = 1e-4
    layers = [3, 4, 6, 3]

    X_trainval, X_test, y_trainval, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.2, random_state=42)

    train_dataset = CustomDataset(X_train, y_train)
    val_dataset = CustomDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

    model = ResNetCaptcha(layers=layers).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    train_losses, val_losses = [], []
    train_char_accs, val_char_accs = [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        char_correct = 0
        char_total = 0

        for batch_features, batch_labels in train_loader:
            batch_features, batch_labels = batch_features.to(device), batch_labels.to(device)
            outputs = model(batch_features)

            loss = criterion(outputs.view(-1, outputs.shape[-1]), batch_labels.view(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

            with torch.no_grad():
                predicted = torch.argmax(outputs, dim=2)
                char_correct += (predicted == batch_labels).sum().item()
                char_total += batch_labels.numel()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        train_char_accs.append(char_correct / char_total)

        model.eval()
        total_val_loss = 0
        val_char_correct = 0
        val_char_total = 0

        with torch.no_grad():
            for val_features, val_labels in val_loader:
                val_features, val_labels = val_features.to(device), val_labels.to(device)
                val_outputs = model(val_features)

                val_loss = criterion(val_outputs.view(-1, val_outputs.shape[-1]), val_labels.view(-1))
                total_val_loss += val_loss.item()

                val_predicted = torch.argmax(val_outputs, dim=2)
                val_char_correct += (val_predicted == val_labels).sum().item()
                val_char_total += val_labels.numel()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        val_char_accs.append(val_char_correct / val_char_total)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    print('---------------------------------------------------')
    print(f"Final Train Character Accuracy: {train_char_accs[-1]:.4f}")
    print(f"Final Validation Character Accuracy: {val_char_accs[-1]:.4f}")
    print('---------------------------------------------------')

    return model, train_losses, val_losses, train_char_accs, val_char_accs

