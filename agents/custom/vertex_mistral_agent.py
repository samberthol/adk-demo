# agents/custom/vertex_mistral_agent.py
import os
import logging
from typing import AsyncGenerator, List, Optional, Dict, Any

from google.cloud import aiplatform
from google.protobuf import json_format
from google.protobuf.struct_pb2 import Value

from google.adk.agents import BaseAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event, RequestEvent, FinalResponseEvent, ErrorEvent
from google.genai.types import Content, Part # Re-use Content/Part for structure


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

        request_event = context.current_event
        if not isinstance(request_event, RequestEvent) or not request_event.content:
            yield ErrorEvent(message="Invalid request event for VertexMistralAgent.")
            return

        # --- Construct the payload for Vertex AI ---
        # Adapt this based on how you want to handle history and instructions
        # For now, just using the latest user message and the system instruction

        messages_payload = []

        # Add system instruction if provided
        if self.instruction:
             # Vertex AI often uses a different format for system prompts or
             # includes them within the first user message context, depending on the model.
             # For Mistral via rawPredict/SDK, it's usually part of the messages list.
             # Let's prepend it conceptually, though the API might require specific structuring.
             # We might need a dedicated 'system' role or merge it.
             # Sticking to user/assistant roles for now as per Mistral API examples.
             # A simple approach: add instruction to the first user turn? Or use system role if supported.
             # Let's assume we just build the history for now. Mistral API typically uses user/assistant roles.
             pass # Revisit instruction handling if needed based on API behavior


        # Add conversation history (potentially limit size)
        # ADK's context.history should provide past turns
        history = context.history or []
        for event in history:
             # Convert ADK events back to user/assistant message format
             # This needs careful mapping based on your Event types
             if event.is_request() and event.content and event.content.parts:
                  messages_payload.append(
                       {"role": "user", "content": event.content.parts[0].text}
                  )
             elif event.is_final_response() and event.content and event.content.parts:
                  # Assuming model's role is 'assistant'
                  messages_payload.append(
                       {"role": "assistant", "content": event.content.parts[0].text}
                  )
             # Handle ToolRequest/ResponseEvents if tools are involved

        # Add the current user message
        current_text = request_event.content.parts[0].text
        messages_payload.append({"role": "user", "content": current_text})

        # Define instance and parameters for the prediction call
        # The structure depends heavily on the specific model's requirements on Vertex AI
        instances = [{"messages": messages_payload}]
        parameters = {
            "temperature": 0.7, # Example parameter
            "max_output_tokens": 1024, # Example parameter
            # Add other parameters like top_p, top_k as needed
        }
        parameters_dict = {}
        json_format.ParseDict(parameters, parameters_dict)

        logger.info(f"[{self.name}] Sending prediction request to Vertex AI model {self.model_name}...")
        # logger.debug(f"[{self.name}] Payload (Instances): {instances}") # Be careful logging PII
        # logger.debug(f"[{self.name}] Parameters: {parameters}")

        try:
            # Use the predict method of the Model object
            response = self.model.predict(instances=instances, parameters=parameters_dict)

            # --- Process the response ---
            if not response.predictions:
                 logger.warning(f"[{self.name}] Vertex AI response contained no predictions.")
                 yield ErrorEvent(message="Model returned no predictions.")
                 return

            # The response structure for publisher models can vary. Inspect the response object.
            # It often mirrors the structure seen in REST API calls.
            # Assuming a structure similar to the REST examples:
            # prediction = response.predictions[0] # Usually a dict or protobuf Struct

            # Convert protobuf Struct to dict if necessary
            prediction_dict = json_format.MessageToDict(response.predictions[0]._pb)

            # Extract the content - PATH MAY VARY GREATLY depending on model!
            # Check the actual response structure carefully.
            # Example path based on common chat completion structures:
            if 'candidates' in prediction_dict and prediction_dict['candidates']:
                content_text = prediction_dict['candidates'][0]['content']['parts'][0]['text']
            elif 'choices' in prediction_dict and prediction_dict['choices']: # More like Mistral API
                 content_text = prediction_dict['choices'][0]['message']['content']
            else:
                 # Fallback or further inspection needed
                 logger.warning(f"[{self.name}] Could not extract content from prediction. Response structure: {prediction_dict}")
                 content_text = "[Could not parse model response]"


            logger.info(f"[{self.name}] Received response from Vertex AI.")
            yield FinalResponseEvent(
                 content=Content(parts=[Part(text=content_text)])
            )

        except Exception as e:
            logger.error(f"[{self.name}] Error during Vertex AI prediction: {e}", exc_info=True)
            error_message = f"Error calling model: {str(e)}"
            # Check for specific error types if needed (e.g., permissions)
            yield ErrorEvent(message=error_message)