import os
import torch
import pandas as pd
from tqdm import tqdm

def evaluate_model_and_save_results(model, criterion, test_loader, results_dir,
                                    metrics_filename='test_evaluation_results.txt',
                                    predictions_filename='continuous_predictions.csv'):

    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    y_pred_continuous = []

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluating Test Set', unit='batch'):

            if inputs.shape[1] != 3:
                inputs = inputs.permute(0, 3, 1, 2)

            outputs = model(inputs).squeeze()
            y_pred_continuous.append(outputs.cpu().item())

            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            test_mae += torch.sum(torch.abs(outputs - labels)).item()

    test_loss /= len(test_loader.dataset)
    test_mae /= len(test_loader.dataset)

    print(f'Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}')

    metrics_path = os.path.join(results_dir, metrics_filename)
    with open(metrics_path, 'w') as file:
        file.write(f'Test Loss: {test_loss:.4f}\n')
        file.write(f'Test MAE: {test_mae:.4f}\n')

    predictions_path = os.path.join(results_dir, predictions_filename)
    pd.DataFrame(y_pred_continuous, columns=['ContinuousPredictions']).to_csv(predictions_path, index=False)

    print(f"Continuous predictions saved to {predictions_path}")