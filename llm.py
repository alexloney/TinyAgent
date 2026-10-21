import json
import urllib.request

import dotenv
dotenv.load_dotenv()

from response import Response

class LLM:
    def __init__(
            self,
            model: str = dotenv.get_key(dotenv.find_dotenv(), "OPENAPI_MODEL"),
            base_url: str = dotenv.get_key(dotenv.find_dotenv(), "OPENAPI_HOST"),
            api_key: str = dotenv.get_key(dotenv.find_dotenv(), "OPENAPI_API_KEY"),
            think: bool = False
    ):
        """Initialize the LLM with the given model."""
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.think = think
        self.reasoning_required = self._is_reasoning_required()

    def generate(
            self,
            messages: list[dict],
            tools: list | None = None
    ) -> Response:
        """Generate a response from the LLM given a list of messages."""
        # Build the request body
        body = {
            "model": self.model,
            "messages": messages
        }

        # Tools and Reasoning
        if tools:
            body["tools"] = tools

        # I ran into an issue where reasoning is required for some models, which causes a
        # 400 error from OpenRouter if we set it to none.
        if not self.reasoning_required:
            if not self.think:
                body["reasoning_effort"] = "none"

        # POST to the OpenAI-compatible /chat/completions endpoint
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )

        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read().decode("utf-8"))

        # Extract message, tool_call, and metadata
        message = data["choices"][0]["message"]
        tool_calls = message.get("tool_calls", [])
        tool_call = tool_calls[0] if tool_calls else None
        metadata = {
            "model": data["model"],
            "prompt_tokens": data["usage"]["prompt_tokens"],
            "completion_tokens": data["usage"]["completion_tokens"]
        }

        # Format as Response dataclass
        return Response(
            content=message.get("content"),
            reasoning=message.get("reasoning"),
            tool_call=tool_call,
            metadata=metadata
        )

    def _is_reasoning_required(self) -> bool:
        """Determine if reasoning is required based on the think flag."""

        request = urllib.request.Request(
            f"{self.base_url}/models",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
        )
        with urllib.request.urlopen(request) as response:
            data = json.loads(response.read().decode("utf-8"))

        for model in data["data"]:
            if model["id"].endswith(self.model):
                return model["reasoning"]["mandatory"]

        # Fallback if the model is not found
        return False