import os
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

from utils.early_stopping import EarlyStopping


def train_model(model, criterion, optimizer, train_loader, val_loader,
                num_epochs=100, base_results_dir=None, patience=10):

    model_name = model.__class__.__name__
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    results_dir = os.path.join(base_results_dir, f"{model_name}_{timestamp}")
    os.makedirs(results_dir, exist_ok=True)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    best_model_path = os.path.join(results_dir, f'{model_name}_best.pth')

    early_stopping = EarlyStopping(patience=patience, min_delta=0.001)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        # Training
        for inputs, labels in tqdm(train_loader,
                                   desc=f'Epoch {epoch+1}/{num_epochs} [Training]',
                                   unit='batch', leave=False):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in tqdm(val_loader,
                                       desc=f'Epoch {epoch+1}/{num_epochs} [Validation]',
                                       unit='batch', leave=False):
                outputs = model(inputs)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item() * inputs.size(0)

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)

        # Early stopping
        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break

    # Plot loss
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss')
    plt.title('Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.savefig(os.path.join(results_dir,
                f'loss_plot_{model_name}_epochs_{epoch+1}.png'))
    plt.close()

    print(f"Best model saved to {best_model_path}")

    return results_dir