import base64
from pathlib import Path
from typing import Any

from agents.prompts import AgentPrompts


class VisionaryAgent:
    """
    Vision Controller that identifies food in an image and estimates
    calories and macros for each visible item.

    Prompt templates are sourced entirely from prompts.AgentPrompts.
    To change the Visionary's rules or output format, edit prompts.py only.

    Usage
    -----
    agent  = VisionaryAgent(image_path)
    system = agent.system_prompt
    result = agent.analyze(openai_client)
    """

    SYSTEM_PROMPT: str = AgentPrompts.VISIONARY_SYSTEM_PROMPT

    def __init__(self, image_path: str, extra_context: str = "") -> None:
        """
        Parameters
        ----------
        image_path : str
            Absolute or relative path to the food image (jpg / png / webp).
        extra_context : str, optional
            Any extra hint to include in the user prompt, e.g. portion context
            ("this is a restaurant-size plate") or meal type ("breakfast").
        """
        self._image_path = Path(image_path)
        self._extra_context = (extra_context.strip() + "\n") if extra_context else ""

        if not self._image_path.exists():
            raise FileNotFoundError(f"Image not found: {self._image_path}")

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    @property
    def system_prompt(self) -> str:
        return AgentPrompts.VISIONARY_SYSTEM_PROMPT

    def build_user_prompt(self) -> str:
        """Render the VISIONARY_USER_PROMPT template with optional context."""
        return AgentPrompts.render(
            AgentPrompts.VISIONARY_USER_PROMPT,
            extra_context=self._extra_context,
        )

    def _encode_image(self) -> str:
        """Return the image as a base64-encoded UTF-8 string."""
        with open(self._image_path, "rb") as fh:
            return base64.b64encode(fh.read()).decode("utf-8")

    def _media_type(self) -> str:
        suffix = self._image_path.suffix.lower()
        return {
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".png":  "image/png",
            ".webp": "image/webp",
            ".gif":  "image/gif",
        }.get(suffix, "image/jpeg")

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def analyze(self, client: Any, model: str = "gpt-4o") -> str:
        """
        Send the image to the vision-capable *client* and return the
        structured calorie breakdown as a string.

        Parameters
        ----------
        client : openai.OpenAI
            An already-configured OpenAI (or compatible) client instance.
        model : str
            Must be a vision-capable model; defaults to "gpt-4o".
        """
        image_data = self._encode_image()
        media_type = self._media_type()

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{media_type};base64,{image_data}",
                                "detail": "high",
                            },
                        },
                        {"type": "text", "text": self.build_user_prompt()},
                    ],
                },
            ],
        )
        return response.choices[0].message.content
