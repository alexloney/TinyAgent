import urllib
import json

import dotenv
dotenv.load_dotenv()

class EmbeddingModel:
    """Generate embeddings."""

    def __init__(self,
                 model: str = dotenv.get_key(dotenv.find_dotenv(), "OPENAPI_EMBEDDING_MODEL"),
                 base_url: str = dotenv.get_key(dotenv.find_dotenv(), "OPENAPI_HOST"),
                 api_key: str = dotenv.get_key(dotenv.find_dotenv(), "OPENAPI_API_KEY")) -> None:
        """Initialize the embedding model with the given model."""
        self.model = model
        self.base_url = base_url
        self.api_key = api_key

    def embed(self, text: str) -> list[float]:
        """Convert text into a numerical vector."""
        # POST to the OpenAI-compatible /embeddings endpoint
        request = urllib.request.Request(
            f"{self.base_url}/embeddings",
            data=json.dumps({
                "model": self.model,
                "input": text
            }).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        )
        with urllib.request.urlopen(request) as resp:
            response = json.loads(resp.read())

        # Extract and return the embedding
        return response["data"][0]["embedding"]
