"""
Model Serving API using LiteServe
"""

from src.api.api_serving import ModelAPIServing
from settings import settings
import litserve as ls


if __name__ == "__main__":
    # Serve the model
    api = ModelAPIServing()
    server = ls.LitServer(
        api,
        api_path="/predict",
        track_requests=True,
    )

    server.run(port=settings.API_PORT)
