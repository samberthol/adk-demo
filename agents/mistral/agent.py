# agents/mistral/agent.py
import os
import logging
import asyncio
from typing import AsyncGenerator, List, Optional, Dict, Any

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event
from google.genai.types import Content, Part # Assuming Content/Part structure is still desired output

# Import the Vertex AI SDK
try:
    import google.cloud.aiplatform as aiplatform
    from google.api_core import exceptions as api_core_exceptions
except ImportError:
    raise ImportError("google-cloud-aiplatform is required for MistralVertexAgent. Please install it.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# No longer need _get_gcloud_access_token

class MistralVertexAgent(BaseAgent):
    """
    Custom ADK agent interacting with Mistral models on Vertex AI via the Python SDK.
    Reads configuration from environment variables and uses ADC for authentication.
    """
    # Declare fields expected by the class
    model_id: str
    project_id: str
    location: str
    instruction: Optional[str] = None
    endpoint: aiplatform.Endpoint # Added endpoint attribute

    def __init__(
        self,
        name: str = "MistralVertexAgent",
        description: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs,
    ):
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
            raise ValueError(f"MistralVertexAgent requires environment variables: {', '.join(missing_vars)}")

        # Assign validated values to the declared fields
        self.model_id = model_id_val
        self.project_id = project_id_val
        self.location = location_val
        self.instruction = instruction

        try:
            # Initialize the Vertex AI SDK. This uses ADC automatically.
            aiplatform.init(project=self.project_id, location=self.location)

            # Construct the endpoint name for the publisher model
            endpoint_name = f"projects/{self.project_id}/locations/{self.location}/publishers/mistralai/models/{self.model_id}"

            # Get an SDK client for the endpoint
            self.endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)

        except Exception as e:
            logger.error(f"[{self.name}] Failed to initialize Vertex AI SDK or Endpoint: {e}", exc_info=True)
            raise RuntimeError(f"Vertex AI SDK initialization failed: {e}") from e

        # Model parameters (adjust names if SDK expects different ones)
        self.model_parameters = {
            "temperature": 0.7,
            "topP": 1.0, # Check SDK docs, might be topP or top_p
            "maxOutputTokens": 1024, # Check SDK docs, might be maxOutputTokens or max_tokens
        }
        logger.info(f"[{self.name}] Initialized using SDK for model '{self.model_id}' in '{self.location}'")

    async def run_async(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event | Content, None]:

        current_event = context.current_event
        if not current_event or not current_event.is_request() or not current_event.content:
             logger.warning(f"[{self.name}] Received invalid/non-request event type: {type(current_event)}")
             raise ValueError("Invalid input: Expected a request event with content.")

        # --- Construct the messages payload for Mistral API ---
        # (Structure might need slight adjustment based on SDK predict method requirements)
        messages_payload = []
        if self.instruction:
            # Check how the SDK expects system instructions (might be part of messages or separate param)
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
                      role = "assistant" # Mistral uses 'assistant'
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

        # --- Prepare SDK Prediction Request ---
        # The standard Vertex AI prediction format requires an 'instances' list.
        # Wrap the messages payload inside the expected instance structure.
        # The exact structure might vary, consult SDK docs or experiment.
        # Assuming the container expects a "messages" key within the instance:
        instances = [{"messages": messages_payload}]
        parameters = self.model_parameters

        # --- Make Asynchronous SDK Call ---
        content_text = "[Agent encountered an error]"
        try:
            logger.info(f"[{self.name}] Sending request to Vertex AI Endpoint via SDK...")

            # Define the synchronous prediction function to run in a thread
            def sync_predict():
                # Use the endpoint.predict method
                prediction_response = self.endpoint.predict(
                    instances=instances,
                    parameters=parameters,
                )
                return prediction_response

            # Run the synchronous SDK call in a separate thread
            prediction_response = await asyncio.to_thread(sync_predict)

            logger.info(f"[{self.name}] Received response from Vertex AI SDK.")

            # --- Process SDK Response ---
            # The response object structure depends on the model/endpoint.
            # Inspect prediction_response.predictions structure.
            # Assuming it returns a list of predictions, and each prediction is a dict
            # with a 'content' key or similar based on Mistral's output format.
            if prediction_response.predictions:
                first_prediction = prediction_response.predictions[0]
                if isinstance(first_prediction, dict):
                     # Common patterns: 'content', 'choices'[0]['message']['content']
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
            raise RuntimeError(f"Vertex AI API call failed: {e}") from e
        except Exception as e:
            logger.error(f"[{self.name}] Unexpected error during SDK prediction: {e}", exc_info=True)
            raise RuntimeError(f"Unexpected error during prediction: {e}") from e

        # --- Yield Final Response ---
        final_content = Content(parts=[Part(text=content_text)])
        yield final_content