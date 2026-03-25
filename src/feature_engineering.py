import pandas as pd
from .logging_utils import get_logger

logger = get_logger(__name__)

def basic_feature_stats(df: pd.DataFrame) -> pd.DataFrame:
    logger.info("Computing basic feature statistics")
    stats = df.describe().T
    logger.info("Feature statistics computed")
    return stats