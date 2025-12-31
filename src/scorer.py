import json
from pathlib import Path

import pandas as pd
import logging
import matplotlib.pyplot as plt
from catboost import CatBoostClassifier

logger = logging.getLogger(__name__)

logger.info('Importing pretrained model...')

# Import model (CPU)
model = CatBoostClassifier()
model.load_model('./models/my_catboost.cbm')

# Define optimal threshold
model_th = 0.98
logger.info('Pretrained model imported successfully...')


def _export_top5_feature_importances(dt: pd.DataFrame, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fi = model.get_feature_importance()

    # Use dataframe column names if available and consistent
    if hasattr(dt, "columns") and len(dt.columns) == len(fi):
        feature_names = list(dt.columns)
    else:
        try:
            feature_names = list(model.feature_names_)
        except Exception:
            feature_names = [f"f{i}" for i in range(len(fi))]

    pairs = sorted(zip(feature_names, fi), key=lambda x: x[1], reverse=True)[:5]
    data = {name: float(val) for name, val in pairs}

    out_path = out_dir / "top5_feature_importances.json"
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Saved top-5 feature importances to: %s", str(out_path))


def _export_score_density(dt: pd.DataFrame, out_dir: str):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scores = model.predict_proba(dt)[:, 1]

    out_path = out_dir / "predicted_score_density.png"
    plt.figure()
    plt.hist(scores, bins=60, density=True)
    plt.title("Predicted score density")
    plt.xlabel("score (p(fraud))")
    plt.ylabel("density")
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()

    logger.info("Saved score density plot to: %s", str(out_path))


def make_pred(dt, path_to_file):
    # Make submission dataframe
    submission = pd.DataFrame({
        'index': pd.read_csv(path_to_file).index,
        'prediction': (model.predict_proba(dt)[:, 1] > model_th).astype(int),
    })
    logger.info('Prediction complete for file: %s', path_to_file)

    # Export artifacts for 8-10 points
    try:
        _export_top5_feature_importances(dt, "./output")
        _export_score_density(dt, "./output")
    except Exception:
        logger.exception("Failed to export artifacts (json/png).")

    return submission
