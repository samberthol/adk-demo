# agents/mistral/agent.py
import os
import logging
import asyncio
from typing import AsyncGenerator, List, Optional, Dict, Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part

try:
    import google.cloud.aiplatform as aiplatform
    from google.api_core import exceptions as api_core_exceptions
except ImportError:
    raise ImportError("google-cloud-aiplatform is required for MistralVertexAgent. Please install it.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MistralVertexAgent(BaseAgent):
    """
    Custom ADK agent interacting with Mistral models on Vertex AI via the Python SDK.
    Reads configuration from environment variables and uses ADC for authentication.
    Passes validated config to superclass during initialization.
    """
    # Declare the fields expected by the class/Pydantic/BaseAgent
    model_id: str
    project_id: str
    location: str
    endpoint: aiplatform.Endpoint # Now explicitly required by validation
    instruction: Optional[str] = None

    def __init__(
        self,
        name: str = "MistralVertexAgent",
        description: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs,
    ):
        # --- Read, validate, and prepare values BEFORE calling super().__init__ ---

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
            # This error will cause the instantiation to fail if try/except is removed
            raise ValueError(f"MistralVertexAgent requires environment variables: {', '.join(missing_vars)}")

        endpoint_instance = None
        try:
            # Initialize the Vertex AI SDK. This uses ADC automatically.
            aiplatform.init(project=project_id_val, location=location_val)

            # Construct the endpoint name for the publisher model
            endpoint_name = f"projects/{project_id_val}/locations/{location_val}/publishers/mistralai/models/{model_id_val}"

            # Get an SDK client for the endpoint
            endpoint_instance = aiplatform.Endpoint(endpoint_name=endpoint_name)
            # Maybe add a check here? e.g., accessing endpoint_instance.display_name could force an API call to verify it exists
            logger.debug(f"Successfully created Endpoint object for {endpoint_name}")

        except Exception as e:
            logger.error(f"[{name}] Failed to initialize Vertex AI SDK or Endpoint: {e}", exc_info=True)
            # Re-raise to ensure instantiation fails if SDK/Endpoint setup fails
            raise RuntimeError(f"Vertex AI SDK/Endpoint initialization failed: {e}") from e

        # --- Call super().__init__ passing the prepared values ---
        super().__init__(
            model_id=model_id_val,
            project_id=project_id_val,
            location=location_val,
            endpoint=endpoint_instance, # Pass the created endpoint object
            instruction=instruction,     # Pass instruction
            name=name,
            description=description,
            **kwargs,
        )
        # --- End of super().__init__ call ---

        # The fields (self.model_id, self.project_id, etc.) are now set by the superclass init

        # Model parameters (adjust names if SDK expects different ones)
        self.model_parameters = {
            "temperature": 0.7,
            "topP": 1.0,
            "maxOutputTokens": 1024,
        }
        logger.info(f"[{self.name}] Initialized successfully using SDK for model '{self.model_id}' in '{self.location}'")

    async def run_async(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event | Content, None]:
        # --- (run_async method remains the same as the previous SDK version) ---
        # ... (code for preparing messages, calling endpoint.predict via asyncio.to_thread, parsing response) ...

        current_event = context.current_event
        if not current_event or not current_event.is_request() or not current_event.content:
             logger.warning(f"[{self.name}] Received invalid/non-request event type: {type(current_event)}")
             raise ValueError("Invalid input: Expected a request event with content.")

        messages_payload = []
        # Use self.instruction which was set during __init__
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

        instances = [{"messages": messages_payload}]
        parameters = self.model_parameters

        content_text = "[Agent encountered an error]"
        try:
            logger.info(f"[{self.name}] Sending request to Vertex AI Endpoint via SDK...")

            def sync_predict():
                # Use self.endpoint which was set during __init__
                prediction_response = self.endpoint.predict(
                    instances=instances,
                    parameters=parameters,
                )
                return prediction_response

            prediction_response = await asyncio.to_thread(sync_predict)

            logger.info(f"[{self.name}] Received response from Vertex AI SDK.")

            if prediction_response.predictions:
                first_prediction = prediction_response.predictions[0]
                if isinstance(first_prediction, dict):
                     if 'content' in first_prediction:
                          content_text = first_prediction['content']
                     elif 'choices' in first_prediction and \
                          isinstance(first_prediction['choices'], list) and \
                          len(first_prediction['choices']) > 0 and \
                          isinstance(first_prediction['choices'][0], dict) and \
                          'message' in first_prediction['choices'][0] and \
                          isinstance(first_prediction['choices'][0]['message'], dict) and \
                          'content' in first_prediction['choices'][0]['message']:
                          content_text = first_prediction['choices'][0]['message']['content']
                     else:
                          logger.warning(f"[{self.name}] Could not find expected content structure in prediction: {first_prediction}")
                          content_text = f"[Agent received unexpected response structure: {str(first_prediction)[:200]}]"
                else:
                     logger.warning(f"[{self.name}] Prediction format not a dict: {type(first_prediction)}")
                     content_text = f"[Agent received unexpected response type: {type(first_prediction)}]"
            else:
                logger.warning(f"[{self.name}] Received empty predictions list from SDK.")
                content_text = "[Agent received no predictions]"

        except api_core_exceptions.GoogleAPIError as e:
            logger.error(f"[{self.name}] Vertex AI SDK API error: {e}", exc_info=True)
            # Decide how to surface this error. Raising it will stop the agent turn.
            # You could also yield an error message Content object.
            # Let's yield an error message for now.
            content_text = f"[Agent encountered API error: {e.message}]"
            # raise RuntimeError(f"Vertex AI API call failed: {e}") from e
        except Exception as e:
            logger.error(f"[{self.name}] Unexpected error during SDK prediction: {e}", exc_info=True)
            content_text = f"[Agent encountered unexpected error: {type(e).__name__}]"
            # raise RuntimeError(f"Unexpected error during prediction: {e}") from e

        final_content = Content(parts=[Part(text=content_text)])
        yield final_content