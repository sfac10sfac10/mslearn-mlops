import json
import joblib
import numpy as np
import os

# AML mounts model under this env var
MODEL_DIR = os.environ.get("AZUREML_MODEL_DIR", "/var/azureml-app/azureml-models")
# When deploying a single model, AML also exposes AZUREML_MODEL_DIR directly to the model version folder.
# We handle both cases defensively:


def _find_model_path():
    # If AZUREML_MODEL_DIR points directly to the model folder, try model.joblib inside it
    direct = os.path.join(MODEL_DIR, "model.joblib")
    if os.path.exists(direct):
        return direct
    # Otherwise, search recursively for model.joblib (robust across layouts)
    for root, _, files in os.walk(MODEL_DIR):
        if "model.joblib" in files:
            return os.path.join(root, "model.joblib")
    raise FileNotFoundError("model.joblib not found under AZUREML_MODEL_DIR")


def init():
    global model
    model_path = _find_model_path()
    model = joblib.load(model_path)


def run(raw_data):
    # Expect the same format you used in testing (columns + data)
    # e.g., {"input_data": {"columns": [...], "data": [[...], ...]}}
    payload = json.loads(raw_data)
    data = payload.get("input_data", {}).get("data", [])
    arr = np.array(data, dtype=float)
    preds = model.predict_proba(arr)[:, 1].tolist()
    return {"predictions": preds}
