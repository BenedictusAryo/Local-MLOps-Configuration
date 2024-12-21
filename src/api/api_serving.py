"""
API Serving Module
This script loads the trained model and serves it as an API using LiteServe.
this script assumes that the model has been trained and saved in the saved_model folder.
and the preprocessing steps have been saved in the preprocessing folder.
"""

import litserve as ls
from src.logging.console_log import setup_logging
from src.api.api_model import PredictionRequest, PredictionResponse
from src.data.preprocess_data import preprocess_transform
from settings import settings
import pandas as pd
import numpy as np
import pickle


# setup logging
logger = setup_logging()


class ModelAPIServing(ls.LitAPI):
    def setup(self, device):
        """Setup the model for serving"""
        # Load the model
        logger.info(f"Loading model from {settings.MODEL_PATH}")
        with open(settings.MODEL_PATH, "rb") as file:
            self.model = pickle.load(file)
        logger.info(f"Model loaded successfully from {settings.MODEL_PATH}")

        # Load the scaler
        logger.info(f"Loading scaler from {settings.SCALER_PATH}")
        with open(settings.SCALER_PATH, "rb") as file:
            self.scaler = pickle.load(file)
        logger.info(f"Scaler loaded successfully from {settings.SCALER_PATH}")

        # Get class labels
        self.class_labels = self.model.classes_

    def decode_request(self, request: PredictionRequest, context) -> list:
        """Decode the incoming request"""
        return pd.DataFrame(request.model_dump(), index=[0])

    def predict(self, features, context) -> PredictionResponse:
        """Make a prediction"""
        # Preprocess the data
        transformed_features = preprocess_transform(
            features,
            self.scaler,
            settings.NUMERICAL_FEATURE_COLUMNS,
            settings.CATEGORICAL_COLUMNS,
        )

        # Make a prediction
        prediction = self.model.predict_proba(transformed_features)
        return prediction

    def encode_response(self, response, context) -> PredictionResponse:
        """Encode the response"""
        return PredictionResponse(
            prediction=self.class_labels[np.argmax(response, axis=1)].tolist()[0],
            probability=np.max(response).tolist(),
        )


if __name__ == "__main__":
    # Serve the model
    api = ModelAPIServing()
    server = ls.LitServer(
        api,
        api_path="/predict",
    )

    server.run(port=settings.API_PORT)
