# agents/custom/vertex_mistral_agent.py
import os
import logging
from typing import AsyncGenerator, List, Optional, Dict, Any

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
# --- MODIFIED IMPORT ---
# Remove RequestEvent, rely on event methods later
from google.adk.events import Event, FinalResponseEvent, ErrorEvent
from google.genai.types import Content, Part


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VertexMistralAgent(BaseAgent):
    """
    A custom ADK agent that interacts with a Mistral model hosted on Vertex AI
    using the google-cloud-aiplatform SDK.
    """
    def __init__(
        self,
        model_name: str, # Expecting full path: projects/.../models/...
        project: str,
        location: str,
        name: str = "VertexMistralAgent",
        description: Optional[str] = None,
        instruction: Optional[str] = None,
        # Add other params like tools if needed by the agent's logic
        **kwargs,
    ):
        super().__init__(name=name, description=description, **kwargs)
        self.model_name = model_name
        self.project = project
        self.location = location
        self.instruction = instruction
        # Initialize the Vertex AI client (uses ADC in Cloud Run)
        try:
            aiplatform.init(project=self.project, location=self.location)
            # Get a reference to the model
            # Note: This assumes the model_name is the full resource path
            # It doesn't guarantee the model exists until predict is called
            self.model = aiplatform.Model(model_name=self.model_name)
            logger.info(f"[{self.name}] Initialized Vertex AI client for model: {self.model_name}")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to initialize Vertex AI client: {e}", exc_info=True)
            self.model = None # Ensure model is None if init fails

    async def run_async(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event, None]:
        """Runs the agent for one turn."""
        if not self.model:
            yield ErrorEvent(message="Vertex AI client not initialized.")
            return

        current_event = context.current_event
        # --- MODIFIED CHECK ---
        # Check if it's a request event using its method, not type name
        if not current_event or not current_event.is_request() or not current_event.content:
             logger.warning(f"[{self.name}] Received invalid/non-request event: {type(current_event)}")
             yield ErrorEvent(message="Invalid request event for VertexMistralAgent.")
             return

        # --- Construct the payload for Vertex AI ---
        # (Payload construction logic remains the same)
        messages_payload = []
        # Add system instruction handling here if needed based on model requirements

        history = context.history or []
        for event in history:
             if event.is_request() and event.content and event.content.parts:
                  messages_payload.append(
                       {"role": "user", "content": event.content.parts[0].text}
                  )
             elif event.is_final_response() and event.content and event.content.parts:
                  messages_payload.append(
                       {"role": "assistant", "content": event.content.parts[0].text}
                  )

        current_text = current_event.content.parts[0].text
        messages_payload.append({"role": "user", "content": current_text})

        instances = [{"messages": messages_payload}]
        parameters = {
            "temperature": 0.7,
            "max_output_tokens": 1024,
        }
        parameters_dict = {}
        json_format.ParseDict(parameters, parameters_dict)

        logger.info(f"[{self.name}] Sending prediction request to Vertex AI model {self.model_name}...")

        try:
            response = self.model.predict(instances=instances, parameters=parameters_dict)

            # --- Process the response ---
            # (Response processing logic remains the same - MUST be verified)
            if not response.predictions:
                 logger.warning(f"[{self.name}] Vertex AI response contained no predictions.")
                 yield ErrorEvent(message="Model returned no predictions.")
                 return

            prediction_dict = json_format.MessageToDict(response.predictions[0]._pb)

            if 'candidates' in prediction_dict and prediction_dict['candidates']:
                # This path might be for Gemini models on Vertex, check Mistral response structure
                content_text = prediction_dict['candidates'][0]['content']['parts'][0]['text']
            elif 'choices' in prediction_dict and prediction_dict['choices']: # More likely for Mistral
                 content_text = prediction_dict['choices'][0]['message']['content']
            else:
                 logger.warning(f"[{self.name}] Could not extract content from prediction. Response structure: {prediction_dict}")
                 content_text = "[Could not parse model response]"

            logger.info(f"[{self.name}] Received response from Vertex AI.")
            yield FinalResponseEvent(
                 content=Content(parts=[Part(text=content_text)])
            )

        except Exception as e:
            logger.error(f"[{self.name}] Error during Vertex AI prediction: {e}", exc_info=True)
            error_message = f"Error calling model: {str(e)}"
            yield ErrorEvent(message=error_message)