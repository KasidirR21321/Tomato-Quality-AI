import numpy as np
import os

def adjust_and_convert_to_classes(y_pred_continuous, test_mae, results_dir):
    y_pred_classes = np.zeros(len(y_pred_continuous))
    upper_decimals = np.zeros(len(y_pred_continuous))
    lower_decimals = np.zeros(len(y_pred_continuous))
    
    for i, val in enumerate(y_pred_continuous):
        y_pred_adjusted_upper = val + test_mae
        y_pred_adjusted_lower = val - test_mae
        
        upper_decimals[i] = y_pred_adjusted_upper % 1
        lower_decimals[i] = y_pred_adjusted_lower % 1
        
        upper_rounded = np.floor(y_pred_adjusted_upper) if upper_decimals[i] <= 0.5 else np.ceil(y_pred_adjusted_upper)
        lower_rounded = np.floor(y_pred_adjusted_lower) if lower_decimals[i] <= 0.5 else np.ceil(y_pred_adjusted_lower)
        
        if abs(upper_rounded - y_pred_adjusted_upper) < abs(lower_rounded - y_pred_adjusted_lower):
            y_pred_classes[i] = upper_rounded
        else:
            y_pred_classes[i] = lower_rounded

    data_to_save = np.column_stack((y_pred_continuous, upper_decimals, lower_decimals, y_pred_classes))
    
    output_filepath = os.path.join(results_dir, 'adjustment_details.txt')
    np.savetxt(output_filepath, data_to_save, fmt='%.4f',
               header='y_pred_continuous, upper_decimal, lower_decimal, y_pred_classes',
               comments='')
    
    return y_pred_classes