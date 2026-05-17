
from pathlib import Path
from datetime import datetime

def make_checkpoint_dir(model_name: str):
    """Creates directory in which model checkpoints are saved. Creates local directory if model training is run locally,
    creates folder on Google Drive if training is run with Google Colab kernel.

    Args:
        model_name: Name of architecture of model trained

    Returns:
        save_checkpoint_dir: Directory where models are to be saved
    """
    date = Path(datetime.today().isoformat().split('T')[0])

    try:
        # save to google drive if using colab kernel
        from google.colab import drive
        drive.mount('/content/drive')

        save_checkpoint_dir = Path(f'/content/drive/MyDrive/colab_notebooks/projects/forecast-electricity-markets/models/{model_name}') / date
    except ImportError:
            # save locally if not using colab kernel
        save_checkpoint_dir = Path(f'results/models/{model_name}') / date
    if save_checkpoint_dir.exists():
        num_runs = len(list(save_checkpoint_dir.glob("*/")))
        save_checkpoint_dir = save_checkpoint_dir / Path(f"Run{num_runs}")
    else:
        save_checkpoint_dir = save_checkpoint_dir / Path(f"Run0")
    save_checkpoint_dir.mkdir(exist_ok=True, parents=True)

    return save_checkpoint_dir