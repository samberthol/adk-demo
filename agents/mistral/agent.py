# agents/mistral/agent.py
import os
import logging
import json
import asyncio
from typing import AsyncGenerator, List, Optional, Dict, Any

import httpx
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def _get_gcloud_access_token() -> str:
    # ... (function remains the same) ...
    try:
        process = await asyncio.create_subprocess_shell(
            "gcloud auth print-access-token",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()

        if process.returncode != 0:
            error_message = stderr.decode().strip() if stderr else "Unknown error"
            logger.error(f"Failed to get gcloud access token. Return code: {process.returncode}. Error: {error_message}")
            raise RuntimeError(f"gcloud auth print-access-token failed: {error_message}")

        access_token = stdout.decode().strip()
        if not access_token:
             raise RuntimeError("gcloud auth print-access-token returned empty token.")
        return access_token
    except FileNotFoundError:
        logger.error("Failed to get gcloud access token: 'gcloud' command not found.")
        raise RuntimeError("'gcloud' command not found. Is Google Cloud SDK installed and in PATH?")
    except Exception as e:
        logger.error(f"Unexpected error getting gcloud access token: {e}", exc_info=True)
        raise RuntimeError(f"Unexpected error getting gcloud access token: {e}")


class MistralVertexAgent(BaseAgent):
    """
    Custom ADK agent interacting with Mistral models on Vertex AI via rawPredict endpoint.
    Reads configuration from environment variables.
    """
    # --- Add Class Attribute Declarations ---
    # Declare the fields that the agent will use, allowing assignment in __init__
    model_id: str
    project_id: str
    location: str
    instruction: Optional[str] = None
    # --- End of Declarations ---

    # Keep name, description, etc. in __init__ signature if passed during instantiation
    def __init__(
        self,
        name: str = "MistralVertexAgent",
        description: Optional[str] = None,
        instruction: Optional[str] = None, # Keep instruction if it was being passed
        **kwargs,
    ):
        # Initialize BaseAgent first (adjust if name/description/instruction are passed differently)
        super().__init__(name=name, description=description, **kwargs)

        # Read configuration from environment variables
        model_id_val = os.environ.get('MISTRAL_MODEL_ID')
        project_id_val = os.environ.get('GCP_PROJECT_ID')
        location_val = os.environ.get('REGION')

        # Validate required config
        missing_vars = []
        if not model_id_val: missing_vars.append('MISTRAL_MODEL_ID')
        if not project_id_val: missing_vars.append('GCP_PROJECT_ID')
        if not location_val: missing_vars.append('REGION')

        if missing_vars:
            # This error will now be raised directly if env vars are missing (due to removed try/except in meta-agent)
            raise ValueError(f"MistralVertexAgent requires environment variables: {', '.join(missing_vars)}")

        # --- Assign validated values to the declared fields ---
        self.model_id = model_id_val
        self.project_id = project_id_val
        self.location = location_val
        self.instruction = instruction # Assign instruction from parameter

        # Construct the rawPredict URL using the assigned self attributes
        self.api_endpoint = f"https://{self.location}-aiplatform.googleapis.com"
        self.predict_url = (
            f"{self.api_endpoint}/v1/projects/{self.project_id}"
            f"/locations/{self.location}/publishers/mistralai/models/{self.model_id}:rawPredict"
        )

        # Model parameters
        self.model_parameters = {
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 1024,
        }
        # Log success *after* all assignments and URL construction
        logger.info(f"[{self.name}] Initialized using environment config for model '{self.model_id}' at '{self.predict_url}'")

    async def run_async(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event | Content, None]:
        # ... (run_async method remains the same) ...
        current_event = context.current_event
        if not current_event or not current_event.is_request() or not current_event.content:
             logger.warning(f"[{self.name}] Received invalid/non-request event type: {type(current_event)}")
             raise ValueError("Invalid input: Expected a request event with content.")

        messages_payload = []
        if self.instruction:
            messages_payload.append({"role": "system", "content": self.instruction})

        history = context.history or []
        for event in history[-10:]:
             try:
                 role = None
                 text = None
                 if event.is_request() and event.content and event.content.parts:
                      role = "user"
                      text = event.content.parts[0].text
                 elif event.is_final_response() and event.content and event.content.parts:
                      role = "assistant"
                      text = event.content.parts[0].text

                 if role and text:
                      messages_payload.append({"role": role, "content": text})

             except Exception as e:
                  logger.warning(f"[{self.name}] Error processing history event {type(event)}: {e}")

        try:
            current_text = current_event.content.parts[0].text
            messages_payload.append({"role": "user", "content": current_text})
        except (AttributeError, IndexError) as e:
            logger.error(f"[{self.name}] Could not extract text from current event: {e}", exc_info=True)
            raise ValueError("Invalid request content.")

        try:
            access_token = await _get_gcloud_access_token()
        except Exception as e:
             raise RuntimeError(f"Authentication failed: {e}") from e

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
        }

        payload = {
            "messages": messages_payload,
            **self.model_parameters
        }

        logger.info(f"[{self.name}] Sending rawPredict request to {self.predict_url}")

        async with httpx.AsyncClient(timeout=120.0) as client:
            try:
                response = await client.post(
                    url=self.predict_url,
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()

            except httpx.HTTPStatusError as e:
                logger.error(f"[{self.name}] HTTP error during rawPredict call: {e.response.status_code} - {e.response.text}", exc_info=True)
                raise RuntimeError(f"API call failed: {e.response.status_code} - Check logs for details.") from e
            except httpx.RequestError as e:
                 logger.error(f"[{self.name}] Request error during rawPredict call: {e}", exc_info=True)
                 raise RuntimeError(f"API request failed: {e}") from e
            except Exception as e:
                logger.error(f"[{self.name}] Unexpected error during rawPredict HTTP call: {e}", exc_info=True)
                raise RuntimeError(f"Unexpected error during API call: {e}") from e

        try:
            response_dict = response.json()
            content_text = response_dict.get("choices", [{}])[0].get("message", {}).get("content", "[No content found in response]")
            if content_text == "[No content found in response]":
                 logger.warning(f"[{self.name}] Could not extract content from response structure: {response_dict}")

        except json.JSONDecodeError:
            logger.error(f"[{self.name}] Failed to decode JSON response. Status: {response.status_code}. Response Text: {response.text}")
            raise RuntimeError(f"Invalid JSON response from API (Status: {response.status_code}).")
        except (IndexError, KeyError, TypeError) as e:
             logger.error(f"[{self.name}] Failed to parse expected fields from response JSON: {e}. Response: {response_dict}", exc_info=True)
             raise RuntimeError(f"Failed to parse response structure: {e}")

        logger.info(f"[{self.name}] Received response from Vertex AI rawPredict.")
        final_content = Content(parts=[Part(text=content_text)])
        yield final_content