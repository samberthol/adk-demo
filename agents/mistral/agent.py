# agents/mistral/agent.py
import os
import logging
import asyncio
from typing import AsyncGenerator, List, Optional, Dict, Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

# Import the Mistral GCP client library
try:
    from mistralai_gcp import MistralGoogleCloud
except ImportError:
    raise ImportError("mistralai-gcp is required for MistralVertexAgent. Please install it.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MistralVertexAgent(BaseAgent):
    """
    Custom ADK agent interacting with Mistral models on Vertex AI via the mistralai-gcp library.
    Reads configuration from environment variables and uses ADC for authentication.
    Uses the first user event in the history as input.
    """
    instruction: Optional[str] = None
    model_name_version: Optional[str] = None
    client: Optional[MistralGoogleCloud] = None
    model_parameters: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        name: str = "MistralVertexAgent",
        description: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs,
    ):
        # Initialize BaseAgent
        super().__init__(name=name, description=description, instruction=instruction, **kwargs)

        # Assign model_name_version here
        self.model_name_version = os.environ.get('MISTRAL_MODEL_ID')
        project_id_val = os.environ.get('GCP_PROJECT_ID')
        location_val = os.environ.get('REGION')

        # Validate required config
        missing_vars = []
        if not self.model_name_version: missing_vars.append('MISTRAL_MODEL_ID')
        if not project_id_val: missing_vars.append('GCP_PROJECT_ID')
        if not location_val: missing_vars.append('REGION')

        if missing_vars:
            raise ValueError(f"MistralVertexAgent requires environment variables: {', '.join(missing_vars)}")

        try:
            # Initialize client implicitly
            self.client = MistralGoogleCloud()
            logger.info(f"[{self.name}] Initialized MistralGoogleCloud client implicitly using environment.")

        except Exception as e:
            logger.error(f"[{self.name}] Failed to initialize MistralGoogleCloud client: {e}", exc_info=True)
            raise RuntimeError(f"MistralGoogleCloud client initialization failed: {e}") from e

        # Assign model parameters
        self.model_parameters = {
            "temperature": 0.7,
            "top_p": 1.0,
            "max_tokens": 1024,
        }
        logger.info(f"[{self.name}] Configured to use model '{self.model_name_version}'")

    # Implement _run_async_impl
    async def _run_async_impl(
        self, ctx: InvocationContext
    ) -> AsyncGenerator[Event | Content, None]:

        # --- Find the FIRST user event in history --- ## MODIFIED HERE ##
        history = ctx.session.events or []
        first_user_event = None
        for event in history:
            # Look for the first event authored by 'user'
            if event and event.author == 'user':
                first_user_event = event
                break # Stop after finding the first one

        if not first_user_event or not first_user_event.content:
            logger.warning(f"[{self.name}] No initial user event found in history.")
            yield Event(author=self.name, content=Content(parts=[Part(text="[Agent Error: Could not find initial user request in history.]")]))
            return

        try:
            # Extract text from the first user event's content parts
            if not first_user_event.content.parts or not hasattr(first_user_event.content.parts[0], 'text'):
                 raise ValueError("First user event content part is missing text.")
            current_text = first_user_event.content.parts[0].text
        except (AttributeError, IndexError, ValueError) as e:
            logger.error(f"[{self.name}] Could not extract text from first user event: {e}", exc_info=True)
            yield Event(author=self.name, content=Content(parts=[Part(text="[Agent Error: Could not read initial user request content.]")]))
            return
        # --- End Input Handling Modification ---

        # --- Construct the messages payload (Still No History Processing) ---
        # This agent acts only on the single input found above
        messages_payload = []
        if self.instruction:
             messages_payload.append({"role": "system", "content": self.instruction})
        messages_payload.append({"role": "user", "content": current_text})
        # --- End Payload Construction ---

        # --- Make Asynchronous Call via mistralai-gcp Client ---
        content_text = "[Agent encountered an error]"
        try:
            if not self.client:
                raise RuntimeError("Mistral client was not initialized.")
            if not self.model_parameters:
                 raise RuntimeError("Model parameters were not initialized.")
            if not self.model_name_version:
                 raise RuntimeError("Model name was not initialized.")

            logger.info(f"[{self.name}] Sending request to Mistral model '{self.model_name_version}' based on first user input...")

            call_params = {
                "model": self.model_name_version,
                "messages": messages_payload,
                **self.model_parameters
            }

            def sync_predict():
                 response = self.client.chat.complete(**call_params)
                 return response

            response = await asyncio.to_thread(sync_predict)

            logger.info(f"[{self.name}] Received response from MistralGoogleCloud client.")

            if response.choices and response.choices[0].message:
                 content_text = response.choices[0].message.content
            else:
                 logger.warning(f"[{self.name}] Received no choices or message content in response from client. Response: {response}")
                 content_text = "[Agent received no response choices]"

        except Exception as e:
            logger.error(f"[{self.name}] Error during MistralGoogleCloud API call: {e}", exc_info=True)
            content_text = f"[Agent encountered API error: {type(e).__name__}]"

        # --- Yield Final Response as an Event ---
        final_event = Event(
            author=self.name,
            content=Content(parts=[Part(text=content_text)])
        )
        yield final_event