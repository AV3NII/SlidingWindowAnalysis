import os

def log(metrics, exp_setup):
    """
    Log the evaluation metrics to a file.

    Parameters:
    - metrics: Dictionary containing evaluation metrics
    """
    # Create the directory structure if it doesn't exist
    model_name = metrics['model_name']
    window_size = metrics['window_size']

    # Define the output path
    output_dir = 'results'
    output_path = os.path.join('..', output_dir, exp_setup, model_name, "metrics.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Check if the file already exists
    file_exists = os.path.isfile(output_path)

    # Write the metrics to the file (append mode)
    with open(output_path, 'a') as f:
        # Write the header only if the file does not already exist
        if not file_exists:
            f.write(f"model_name,window_size,rmse,mae,smape,r2,forecast_bias,training_time\n")
        f.write(f"{model_name},{window_size},{metrics['rmse']},{metrics['mae']},{metrics['smape']},{metrics['r2']},{metrics['forecast_bias']},{metrics['training_time']}\n")

    print(f"Metrics have been logged to: {output_path}")