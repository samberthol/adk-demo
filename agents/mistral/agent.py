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
    # You might need specific exception types from this library if available
except ImportError:
    raise ImportError("mistralai-gcp is required for MistralVertexAgent. Please install it.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MistralVertexAgent(BaseAgent):
    """
    Custom ADK agent interacting with Mistral models on Vertex AI via the mistralai-gcp library.
    Reads configuration from environment variables and uses ADC for authentication.
    """
    # === REMOVE THE CLASS-LEVEL TYPE HINT HERE ===
    # client: MistralGoogleCloud # Store the client
    # =============================================

    instruction: Optional[str] = None # Keep this one from BaseAgent

    def __init__(
        self,
        name: str = "MistralVertexAgent",
        description: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs,
    ):
        # Initialize BaseAgent
        super().__init__(name=name, description=description, instruction=instruction, **kwargs)

        # Read configuration from environment variables
        self.model_name_version = os.environ.get('MISTRAL_MODEL_ID') # e.g., mistral-small-2503
        project_id_val = os.environ.get('GCP_PROJECT_ID')
        location_val = os.environ.get('REGION')
        # self.instruction is handled by super init now if passed

        # Validate required config
        missing_vars = []
        if not self.model_name_version: missing_vars.append('MISTRAL_MODEL_ID')
        if not project_id_val: missing_vars.append('GCP_PROJECT_ID')
        if not location_val: missing_vars.append('REGION')

        if missing_vars:
            # Raise the error here so it's caught by the meta-agent's try/except
            raise ValueError(f"MistralVertexAgent requires environment variables: {', '.join(missing_vars)}")

        try:
            # Initialize the MistralGoogleCloud client
            # It should use Application Default Credentials (ADC) automatically
            # ASSIGN SELF.CLIENT HERE - Python handles dynamic typing
            self.client = MistralGoogleCloud(
                project_id=project_id_val,
                region=location_val
            )
            logger.info(f"[{self.name}] Initialized MistralGoogleCloud client for project '{project_id_val}' in region '{location_val}'")

        except Exception as e:
            logger.error(f"[{self.name}] Failed to initialize MistralGoogleCloud client: {e}", exc_info=True)
            # Re-raise as RuntimeError to be caught by meta-agent's specific handling
            raise RuntimeError(f"MistralGoogleCloud client initialization failed: {e}") from e

        # Model parameters (use names expected by mistralai-gcp client)
        self.model_parameters = {
            "temperature": 0.7,
            "top_p": 1.0, # Check docs if different name needed
            "max_tokens": 1024, # Check docs if different name needed
        }
        logger.info(f"[{self.name}] Configured to use model '{self.model_name_version}'")

    # run_async method remains the same...
    async def run_async(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event | Content, None]:

        current_event = context.current_event
        if not current_event or not current_event.is_request() or not current_event.content:
             logger.warning(f"[{self.name}] Received invalid/non-request event type: {type(current_event)}")
             raise ValueError("Invalid input: Expected a request event with content.")

        # --- Construct the messages payload ---
        # (The format [{role: 'user', content: '...'}, ...] is standard)
        messages_payload = []
        # Use self.instruction from BaseAgent if set
        if self.instruction:
             # System prompt might need specific handling depending on the client
             # Let's assume it's just another message for now
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

        # --- Make Asynchronous Call via mistralai-gcp Client ---
        content_text = "[Agent encountered an error]"
        try:
            logger.info(f"[{self.name}] Sending request via MistralGoogleCloud client...")

            # Prepare parameters, filtering out None values if necessary
            call_params = {
                "model": self.model_name_version,
                "messages": messages_payload,
                **self.model_parameters # Spread the parameters dict
            }

            # Define the synchronous prediction function
            def sync_predict():
                 # Use the client.chat.complete method (or just client.complete if appropriate)
                 # Check mistralai-gcp docs for exact method structure
                 response = self.client.chat.complete(**call_params)
                 return response

            # Run the synchronous SDK call in a separate thread
            # Alternatively, check if mistralai-gcp has an async client or methods like complete_async
            response = await asyncio.to_thread(sync_predict)

            logger.info(f"[{self.name}] Received response from MistralGoogleCloud client.")

            # --- Process mistralai-gcp Response ---
            # Structure is likely similar to native Mistral API: response.choices[0].message.content
            if response.choices:
                 content_text = response.choices[0].message.content
            else:
                 logger.warning(f"[{self.name}] Received no choices in response from client.")
                 content_text = "[Agent received no response choices]"

        except Exception as e:
            # Catch potential errors from the mistralai-gcp client or ADC
            logger.error(f"[{self.name}] Error during MistralGoogleCloud API call: {e}", exc_info=True)
            content_text = f"[Agent encountered API error: {type(e).__name__}]"
            # Optionally re-raise: raise RuntimeError(f"Mistral API call failed: {e}") from e

        # --- Yield Final Response ---
        final_content = Content(parts=[Part(text=content_text)])
        yield final_content