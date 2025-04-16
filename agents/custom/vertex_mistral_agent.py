# agents/custom/vertex_mistral_agent.py
import os
import logging
from typing import AsyncGenerator, List, Optional, Dict, Any

# --- Vertex AI SDK ---
from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

# --- ADK Core Components ---
from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
# --- MODIFIED IMPORT ---
# Only import base Event, assuming specific subclasses aren't available here
from google.adk.events import Event

# --- Content Representation (re-using from genai or fallback) ---
try:
    from google.genai.types import Content, Part
except ImportError:
    logging.warning("google.genai.types not found, using fallback structures for Content/Part.")
    from dataclasses import dataclass
    @dataclass
    class Part:
        text: str
    @dataclass
    class Content:
        parts: List[Part]
        role: Optional[str] = None

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VertexMistralAgent(BaseAgent):
    """
    A custom ADK agent that interacts with a Mistral model hosted on Vertex AI
    using the google-cloud-aiplatform SDK. (Attempt 3)
    """
    def __init__(
        self,
        model_name: str, # Expecting full Vertex path: projects/.../models/...
        project: str,
        location: str,
        name: str = "VertexMistralAgent",
        description: Optional[str] = None,
        instruction: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(name=name, description=description, **kwargs)
        self.model_name = model_name
        self.project = project
        self.location = location
        self.instruction = instruction
        self.model_parameters = {
            "temperature": 0.7,
            "max_output_tokens": 1024,
        }
        self.model = None
        try:
            aiplatform.init(project=self.project, location=self.location)
            self.model = aiplatform.Model(model_name=self.model_name)
            logger.info(f"[{self.name}] Initialized Vertex AI client for model path: {self.model_name}")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to initialize Vertex AI client: {e}", exc_info=True)

    async def run_async(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event | Content, None]:
        """Runs the agent for one turn."""
        if not self.model:
            logger.error(f"[{self.name}] Aborting run: Vertex AI model reference is not initialized.")
            # --- MODIFIED ERROR HANDLING ---
            # Raise an exception instead of yielding ErrorEvent
            raise RuntimeError("Vertex AI client not initialized.")

        current_event = context.current_event
        if not current_event or not hasattr(current_event, 'is_request') or not current_event.is_request() or not current_event.content:
             logger.warning(f"[{self.name}] Received invalid/non-request event type: {type(current_event)}")
             # --- MODIFIED ERROR HANDLING ---
             raise ValueError("Invalid input: Expected a request event with content.")

        # --- Construct the payload (Same as before) ---
        messages_payload = []
        history = context.history or []
        for event in history[-10:]:
             try:
                  if event.is_request() and event.content and event.content.parts:
                       messages_payload.append({"role": "user", "content": event.content.parts[0].text})
                  elif hasattr(event, 'is_final_response') and event.is_final_response() and event.content and event.content.parts:
                       messages_payload.append({"role": "assistant", "content": event.content.parts[0].text})
             except Exception as e:
                  logger.warning(f"[{self.name}] Error processing history event {type(event)}: {e}")

        try:
            current_text = current_event.content.parts[0].text
            messages_payload.append({"role": "user", "content": current_text})
        except (AttributeError, IndexError) as e:
            logger.error(f"[{self.name}] Could not extract text from current event: {e}", exc_info=True)
            # --- MODIFIED ERROR HANDLING ---
            raise ValueError("Invalid request content.")

        instances = [{"messages": messages_payload}]
        parameters_proto = json_format.ParseDict(self.model_parameters, Value())

        logger.info(f"[{self.name}] Sending prediction request to Vertex AI model {self.model_name}...")

        try:
            response = self.model.predict(instances=instances, parameters=parameters_proto)

            # --- Process the response (Same as before, needs verification) ---
            if not response.predictions:
                 logger.warning(f"[{self.name}] Vertex AI response contained no predictions property.")
                 # --- MODIFIED ERROR HANDLING ---
                 raise ValueError("Model returned no predictions.")

            try:
                 prediction_dict = json_format.MessageToDict(response.predictions[0]._pb)
            except Exception as e:
                 logger.error(f"[{self.name}] Failed to access or parse prediction protobuf: {e}. Response: {response}", exc_info=True)
                 # --- MODIFIED ERROR HANDLING ---
                 raise RuntimeError("Failed to parse model prediction structure.")

            content_text = "[Could not parse model response]"
            if isinstance(prediction_dict, dict):
                if 'choices' in prediction_dict and prediction_dict['choices']:
                     try: content_text = prediction_dict['choices'][0]['message']['content']
                     except Exception: pass # Ignore parsing errors for now, use default
                elif 'candidates' in prediction_dict and prediction_dict['candidates']:
                     try: content_text = prediction_dict['candidates'][0]['content']['parts'][0]['text']
                     except Exception: pass # Ignore parsing errors for now, use default
                else: logger.warning(f"[{self.name}] Unexpected prediction structure.")
            else: logger.warning(f"[{self.name}] Prediction was not a dictionary.")

            logger.info(f"[{self.name}] Received response from Vertex AI.")
            # --- Yield the final response (Same as before) ---
            final_content = Content(parts=[Part(text=content_text)])
            yield final_content

        except Exception as e:
            # --- MODIFIED ERROR HANDLING ---
            # Catch potential errors during the API call and re-raise them.
            # The ADK runner should catch and handle these exceptions.
            logger.error(f"[{self.name}] Error during Vertex AI prediction call: {e}", exc_info=True)
            raise e # Re-raise the original exception