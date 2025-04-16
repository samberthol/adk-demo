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
# Use base Event and specific ErrorEvent (assuming ErrorEvent is correctly exposed)
from google.adk.events import Event, ErrorEvent

# --- Content Representation (re-using from genai for convenience) ---
# Ensure google-genai is installed or replace with your own content structure
# If google-genai was removed, you might need to define simple dicts or dataclasses
try:
    from google.genai.types import Content, Part
except ImportError:
    # Define fallback structures if google-genai is not available
    logging.warning("google.genai.types not found, using fallback structures for Content/Part.")
    from dataclasses import dataclass
    @dataclass
    class Part:
        text: str
    @dataclass
    class Content:
        parts: List[Part]
        role: Optional[str] = None # Optional role attribute


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VertexMistralAgent(BaseAgent):
    """
    A custom ADK agent that interacts with a Mistral model hosted on Vertex AI
    using the google-cloud-aiplatform SDK.

    Assumes authentication is handled via Application Default Credentials (ADC),
    suitable for environments like Cloud Run.
    """
    def __init__(
        self,
        model_name: str, # Expecting full Vertex path: projects/.../models/...
        project: str,
        location: str,
        name: str = "VertexMistralAgent",
        description: Optional[str] = None,
        instruction: Optional[str] = None,
        # Add other params like 'tools' if you plan to implement tool handling
        **kwargs,
    ):
        """
        Initializes the agent and the Vertex AI client.

        Args:
            model_name: The full Vertex AI model resource name.
            project: Google Cloud project ID.
            location: Google Cloud location (region).
            name: The name of the agent.
            description: A description for the agent.
            instruction: System instructions for the underlying LLM.
            **kwargs: Additional keyword arguments for BaseAgent.
        """
        super().__init__(name=name, description=description, **kwargs)
        self.model_name = model_name
        self.project = project
        self.location = location
        self.instruction = instruction
        self.model_parameters = { # Default parameters, can be overridden
            "temperature": 0.7,
            "max_output_tokens": 1024,
            # Add other relevant parameters like top_p, top_k if needed
        }
        self.model = None # Initialize model reference to None

        # Initialize the Vertex AI client (uses ADC in Cloud Run/Vertex AI)
        try:
            aiplatform.init(project=self.project, location=self.location)
            # Get a reference to the model object.
            # This doesn't guarantee the model exists or is accessible until predict is called.
            self.model = aiplatform.Model(model_name=self.model_name)
            logger.info(f"[{self.name}] Initialized Vertex AI client for model path: {self.model_name}")
        except Exception as e:
            logger.error(f"[{self.name}] Failed to initialize Vertex AI client: {e}", exc_info=True)
            # self.model remains None

    async def run_async(
        self, context: InvocationContext
    ) -> AsyncGenerator[Event | Content, None]:
        """
        Runs the agent for one turn, processing the incoming event.

        Args:
            context: The invocation context containing the current event and history.

        Yields:
            An Event (e.g., ErrorEvent) or a Content object representing the final response.
        """
        if not self.model:
            logger.error(f"[{self.name}] Aborting run: Vertex AI model reference is not initialized.")
            yield ErrorEvent(message="Vertex AI client not initialized.")
            return

        current_event = context.current_event
        # Check if the current event is a valid request
        if not current_event or not hasattr(current_event, 'is_request') or not current_event.is_request() or not current_event.content:
             logger.warning(f"[{self.name}] Received invalid or non-request event type: {type(current_event)}")
             yield ErrorEvent(message="Invalid input: Expected a request event with content.")
             return

        # --- Construct the message payload for the Vertex AI API ---
        messages_payload = []

        # TODO: Implement proper handling of system instructions based on Mistral API on Vertex.
        # Mistral typically expects instructions within the message list, often as the first user
        # message or potentially using a 'system' role if supported by the specific endpoint.
        # if self.instruction:
        #    messages_payload.append({"role": "system", "content": self.instruction}) # Check if 'system' role works

        # Add conversation history (limit context window appropriately)
        history = context.history or []
        for event in history[-10:]: # Example: Limit history to last 10 turns
             try:
                  if event.is_request() and event.content and event.content.parts:
                       messages_payload.append(
                            {"role": "user", "content": event.content.parts[0].text}
                       )
                  # Check if the event is a final response (method might vary)
                  elif hasattr(event, 'is_final_response') and event.is_final_response() and event.content and event.content.parts:
                       # Assuming the model's role is 'assistant'
                       messages_payload.append(
                            {"role": "assistant", "content": event.content.parts[0].text}
                       )
                  # TODO: Add handling for tool request/response events if implementing tool use
             except Exception as e:
                  logger.warning(f"[{self.name}] Error processing history event {type(event)}: {e}")


        # Add the current user message
        try:
            current_text = current_event.content.parts[0].text
            messages_payload.append({"role": "user", "content": current_text})
        except (AttributeError, IndexError) as e:
            logger.error(f"[{self.name}] Could not extract text from current event: {e}", exc_info=True)
            yield ErrorEvent(message="Invalid request content.")
            return

        # --- Prepare the prediction request ---
        # The instance format depends on the specific Vertex AI prediction endpoint behavior
        # For many language models, providing a 'messages' list is common.
        instances = [{"messages": messages_payload}]

        # Convert parameters dictionary to protobuf Struct format if needed by SDK method
        parameters_proto = json_format.ParseDict(self.model_parameters, Value())

        logger.info(f"[{self.name}] Sending prediction request to Vertex AI model {self.model_name}...")

        try:
            # Make the prediction call using the aiplatform SDK Model object
            response = self.model.predict(instances=instances, parameters=parameters_proto)

            # --- Process the prediction response ---
            if not response.predictions:
                 logger.warning(f"[{self.name}] Vertex AI response contained no predictions property.")
                 yield ErrorEvent(message="Model returned no predictions.")
                 return

            # The response structure can vary. Inspect carefully.
            # Convert protobuf Struct prediction to a Python dict for easier access
            try:
                 # Assuming the first prediction contains the relevant data
                 prediction_dict = json_format.MessageToDict(response.predictions[0]._pb)
            except (AttributeError, IndexError, TypeError) as e:
                 logger.error(f"[{self.name}] Failed to access or parse prediction protobuf: {e}. Response: {response}", exc_info=True)
                 yield ErrorEvent(message="Failed to parse model prediction structure.")
                 return


            # Extract the actual text content - **THIS PATH NEEDS VERIFICATION**
            # Based on Mistral API structure, it's likely under 'choices' or 'candidates'
            content_text = "[Could not parse model response]" # Default
            if isinstance(prediction_dict, dict):
                if 'choices' in prediction_dict and prediction_dict['choices']:
                     try:
                          content_text = prediction_dict['choices'][0]['message']['content']
                     except (IndexError, KeyError, TypeError) as e:
                          logger.warning(f"[{self.name}] Could not extract content via ['choices'][0]['message']['content']: {e}")
                elif 'candidates' in prediction_dict and prediction_dict['candidates']:
                     try:
                          # Gemini structure often looks like this
                          content_text = prediction_dict['candidates'][0]['content']['parts'][0]['text']
                     except (IndexError, KeyError, TypeError) as e:
                          logger.warning(f"[{self.name}] Could not extract content via ['candidates'][0]['content']['parts'][0]['text']: {e}")
                else:
                     logger.warning(f"[{self.name}] Unexpected prediction structure. Keys: {prediction_dict.keys()}")
            else:
                 logger.warning(f"[{self.name}] Prediction was not a dictionary: {type(prediction_dict)}")


            logger.info(f"[{self.name}] Received response from Vertex AI.")

            # --- Yield the final response ---
            # Yield the Content object directly. ADK's runner should handle this.
            final_content = Content(parts=[Part(text=content_text)])
            yield final_content

        except Exception as e:
            # Catch potential errors during the API call (e.g., network, permissions, invalid args)
            logger.error(f"[{self.name}] Error during Vertex AI prediction call: {e}", exc_info=True)
            # Provide a more informative error message if possible
            error_message = f"Error calling model: {str(e)}"
            yield ErrorEvent(message=error_message)