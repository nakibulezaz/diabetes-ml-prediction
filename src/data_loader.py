from pathlib import Path
import pandas as pd
from .logging_utils import get_logger

logger = get_logger(__name__)

def load_raw_data(path: Path) -> pd.DataFrame:
    logger.info(f"Loading raw data from {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded dataset with shape {df.shape}")
    return df