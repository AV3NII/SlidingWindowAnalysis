import os

def log(metrics):
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
    output_path = os.path.join('..',output_dir, model_name, f"{window_size}-DayWindow-metrics.md")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Write the metrics to the markdown file
    with open(output_path, 'w') as f:
        f.write(f"# Model Performance Metrics\n\n")
        f.write(f"**Model Name:** {model_name}\n\n")
        f.write(f"**Window Size:** {window_size}\n\n")
        f.write(f"**RMSE:** {metrics['rmse']:.4f}\n\n")
        f.write(f"**MAE:** {metrics['mae']:.4f}\n\n")
        f.write(f"**SMAPE:** {metrics['smape']:.4f}\n\n")
        f.write(f"**R²:** {metrics['r2']:.4f}\n\n")
        f.write(f"**Forecast Bias:** {metrics['forecast_bias']:.4f}\n\n")
        f.write(f"**Training Time (seconds):** {metrics['training_time']:.4f}\n\n")

    print(f"Metrics have been logged to: {output_path}")