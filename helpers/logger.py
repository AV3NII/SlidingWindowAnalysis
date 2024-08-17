import os

def log_metrics_to_markdown(metrics):
    # Extract model name and window size
    model_name = metrics.pop('model_name')
    window_size = metrics.pop('window_size')

    # Create directory path
    dir_path = os.path.join('..', 'results', str(window_size)+'-dayWindow', model_name)
    os.makedirs(dir_path, exist_ok=True)

    # Create file path
    file_path = os.path.join(dir_path, 'metrics.md')

    # Write metrics to markdown file
    with open(file_path, 'w') as f:
        f.write(f'# Model Evaluation {model_name}\n\n')
        for key, value in metrics.items():
            f.write(f'**{key.capitalize()}**: {value}\n\n')
