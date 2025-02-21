"""
title: Perplexity Manifold Pipe 2025
author: richtong
author_url: https://github.com/richtong
source_url: https://github.com/tne-ai/open-webui-functions/blob/perplexity-2025-02.py
version: 0.1.2

original authors:
  author: justinh-rahb and moblangeois
  author_url: https://github.com/open-webui
  funding_url: https://github.com/open-webui
  version: 0.1.1
  license: MIT
"""

from pydantic import BaseModel, Field
from typing import Optional, Union, Generator, Iterator
from open_webui.utils.misc import get_last_user_message
from open_webui.utils.misc import pop_system_message

import os
import requests


class Pipe:
    class Valves(BaseModel):
        NAME_PREFIX: str = Field(
            default="Perplexity/",
            description="The prefix applied before the model names.",
        )
        PERPLEXITY_API_BASE_URL: str = Field(
            default="https://api.perplexity.ai",
            description="The base URL for Perplexity API endpoints.",
        )
        PERPLEXITY_API_KEY: str = Field(
            default="",
            description="Required API key to access Perplexity services.",
        )

    def __init__(self):
        self.type = "manifold"
        self.valves = self.Valves()

    def pipes(self):
        return [
            {
                "id": "sonar-reasoning-pro",
                "name": f"{self.valves.NAME_PREFIX}Sonar Reasoning Pro 128k 8k output",
            },
            {
                "id": "sonar-reasoning",
                "name": f"{self.valves.NAME_PREFIX}Sonar Reasoning 128k 8k output",
            },
            {
                "id": "sonar-pro",
                "name": f"{self.valves.NAME_PREFIX}Sonar Pro 200k",
            },
            {
                "id": "sonar",
                "name": f"{self.valves.NAME_PREFIX}Sonar 128k",
            },
            {
                "id": "r1-1776",
                "name": f"{self.valves.NAME_PREFIX}Deepseek-R1 128k",
            },
            {
                "id": "llama-3.1-sonar-small-128k-online",
                "name": f"{self.valves.NAME_PREFIX}Llama 3.1 Sonar Small 128k Online",
            },
            {
                "id": "llama-3.1-sonar-large-128k-online",
                "name": f"{self.valves.NAME_PREFIX}Llama 3.1 Sonar Large 128k Online",
            },
            {
                "id": "llama-3.1-sonar-huge-128k-online",
                "name": f"{self.valves.NAME_PREFIX}Llama 3.1 Sonar Huge 128k Online",
            },
            {
                "id": "llama-3.1-sonar-small-128k-chat",
                "name": f"{self.valves.NAME_PREFIX}Llama 3.1 Sonar Small 128k Chat",
            },
            {
                "id": "llama-3.1-sonar-large-128k-chat",
                "name": f"{self.valves.NAME_PREFIX}Llama 3.1 Sonar Large 128k Chat",
            },
            {
                "id": "llama-3.1-8b-instruct",
                "name": f"{self.valves.NAME_PREFIX}Llama 3.1 8B Instruct",
            },
            {
                "id": "llama-3.1-70b-instruct",
                "name": f"{self.valves.NAME_PREFIX}Llama 3.1 70B Instruct",
            },
        ]

    def pipe(self, body: dict, __user__: dict) -> Union[str, Generator, Iterator]:
        print(f"pipe:{__name__}")

        if not self.valves.PERPLEXITY_API_KEY:
            raise Exception("PERPLEXITY_API_KEY not provided in the valves.")

        headers = {
            "Authorization": f"Bearer {self.valves.PERPLEXITY_API_KEY}",
            "Content-Type": "application/json",
            "accept": "application/json",
        }

        system_message, messages = pop_system_message(body.get("messages", []))
        system_prompt = "You are a helpful assistant."
        if system_message is not None:
            system_prompt = system_message["content"]

        model_id = body["model"]
        if model_id.startswith(self.valves.NAME_PREFIX):
            model_id = model_id[len(self.valves.NAME_PREFIX) :]
        if model_id.startswith("perplexity."):
            model_id = model_id[len("perplexity.") :]

        payload = {
            "model": model_id,
            "messages": [{"role": "system", "content": system_prompt}, *messages],
            "stream": body.get("stream", True),
            "return_citations": True,
            "return_images": True,
        }

        print(payload)

        try:
            r = requests.post(
                url=f"{self.valves.PERPLEXITY_API_BASE_URL}/chat/completions",
                json=payload,
                headers=headers,
                stream=True,
            )

            r.raise_for_status()

            if body.get("stream", False):
                return r.iter_lines()
            else:
                response = r.json()
                formatted_response = {
                    "id": response["id"],
                    "model": response["model"],
                    "created": response["created"],
                    "usage": response["usage"],
                    "object": response["object"],
                    "choices": [
                        {
                            "index": choice["index"],
                            "finish_reason": choice["finish_reason"],
                            "message": {
                                "role": choice["message"]["role"],
                                "content": choice["message"]["content"],
                            },
                            "delta": {"role": "assistant", "content": ""},
                        }
                        for choice in response["choices"]
                    ],
                }
                return formatted_response
        except Exception as e:
            return f"Error: {e}"
