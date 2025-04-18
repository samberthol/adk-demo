# agents/mistral/agent.py
import os
import logging
import json
import asyncio
from typing import AsyncGenerator, List, Optional, Dict, Any

# --- HTTP Client ---
import httpx

# --- ADK Core Components ---
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event # Use base Event

# --- Content Representation (Using ADK's expected structure) ---
# Assuming Content/Part are implicitly handled or available via ADK context
from google.genai.types import Content, Part

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def _get_gcloud_access_token() -> str:
    """Asynchronously gets the gcloud access token."""
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
        # logger.info("Successfully retrieved gcloud access token.") # Keep comments minimal
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
    """
    def __init__(
        self,
        model_id: str, # Expecting just the model ID, e.g., 'mistral-small-2503'
        project_id: str,
        location: str, # e.g., 'us-central1'
        name: str = "MistralVertexAgent",
        description: Optional[str] = None,
        instruction: Optional[str] = None, # System instruction for Mistral
        **kwargs,
    ):
        super().__init__(name=name, description=description, **kwargs)
        # Ensure required config is present
        if not all([model_id, project_id, location]):
             raise ValueError("MistralVertexAgent requires model_id, project_id, and location.")

        self.model_id = model_id
        self.project_id = project_id
        self.location = location
        self.instruction = instruction # Store system instruction if provided

        # Construct the rawPredict URL
        self.api_endpoint = f"https://{self.location}-aiplatform.googleapis.com"
        self.predict_url = (
            f"{self.api_endpoint}/v1/projects/{self.project_id}"
            f"/locations/{self.location}/publishers/mistralai/models/{self.model_id}:rawPredict"
        )

        # Model parameters (can be customized)
        # Note: rawPredict might use different parameter names than SDK predict
        # Refer to Mistral/Vertex documentation for rawPredict specific parameters
        self.model_parameters = {
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 1024, # Corresponds to max_output_tokens in SDK
        }
        logger.info(f"[{self.name}] Initialized for model '{self.model_id}' at '{self.predict_url}'")

    async def run_async(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event | Content, None]:
        """Runs the agent for one turn using the rawPredict endpoint."""

        current_event = context.current_event
        if not current_event or not current_event.is_request() or not current_event.content:
             logger.warning(f"[{self.name}] Received invalid/non-request event type: {type(current_event)}")
             raise ValueError("Invalid input: Expected a request event with content.")

        # --- Construct the messages payload for Mistral ---
        messages_payload = []

        # Add system instruction if provided
        if self.instruction:
            messages_payload.append({"role": "system", "content": self.instruction})

        # Process history (limit to ~last 5 turns / 10 messages for context)
        history = context.history or []
        for event in history[-10:]:
             try:
                 # Map ADK roles (user/model) to Mistral roles (user/assistant)
                 role = None
                 text = None
                 if event.is_request() and event.content and event.content.parts:
                      role = "user"
                      text = event.content.parts[0].text
                 elif event.is_final_response() and event.content and event.content.parts:
                      role = "assistant" # Mistral uses 'assistant'
                      text = event.content.parts[0].text

                 if role and text:
                      # Mistral rawPredict expects content as a simple string for text
                      messages_payload.append({"role": role, "content": text})

             except Exception as e:
                  logger.warning(f"[{self.name}] Error processing history event {type(event)}: {e}")

        # Add current user message
        try:
            current_text = current_event.content.parts[0].text
            # Mistral rawPredict expects content as a simple string for text
            messages_payload.append({"role": "user", "content": current_text})
        except (AttributeError, IndexError) as e:
            logger.error(f"[{self.name}] Could not extract text from current event: {e}", exc_info=True)
            raise ValueError("Invalid request content.")

        # --- Prepare Request ---
        try:
            access_token = await _get_gcloud_access_token()
        except Exception as e:
             # Re-raise auth errors directly
             raise RuntimeError(f"Authentication failed: {e}") from e

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            # "Accept": "application/json" # Often optional, Content-Type is key
        }

        # Combine messages with model parameters into the final payload
        # Structure might vary slightly based on exact rawPredict requirements
        payload = {
            "messages": messages_payload,
            **self.model_parameters # Add temperature, max_tokens etc.
            # "model": self.model_id # Model ID might be needed in payload too, check docs
        }

        logger.info(f"[{self.name}] Sending rawPredict request to {self.predict_url}")
        # logger.debug(f"[{self.name}] Payload: {json.dumps(payload)}") # Optional: Log payload for debugging

        # --- Make Asynchronous HTTP Call ---
        async with httpx.AsyncClient(timeout=120.0) as client: # Increased timeout
            try:
                response = await client.post(
                    url=self.predict_url,
                    headers=headers,
                    json=payload # Send data as JSON
                )
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

            except httpx.HTTPStatusError as e:
                logger.error(f"[{self.name}] HTTP error during rawPredict call: {e.response.status_code} - {e.response.text}", exc_info=True)
                raise RuntimeError(f"API call failed: {e.response.status_code} - Check logs for details.") from e
            except httpx.RequestError as e:
                 logger.error(f"[{self.name}] Request error during rawPredict call: {e}", exc_info=True)
                 raise RuntimeError(f"API request failed: {e}") from e
            except Exception as e:
                logger.error(f"[{self.name}] Unexpected error during rawPredict HTTP call: {e}", exc_info=True)
                raise RuntimeError(f"Unexpected error during API call: {e}") from e # Re-raise

        # --- Process Response ---
        try:
            response_dict = response.json()
            # Extract content based on expected Mistral rawPredict response structure
            # This assumes the structure is like the SDK: response['choices'][0]['message']['content']
            # Verify this structure against actual rawPredict API documentation/output
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
        # --- Yield Final Response ---
        final_content = Content(parts=[Part(text=content_text)])
        yield final_content