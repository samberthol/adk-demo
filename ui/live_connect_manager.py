# ui/live_connect_manager.py
import asyncio
import os
from google import genai
from google.genai import types
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration from the sample (adjust as needed)
CHANNELS = 1
SEND_SAMPLE_RATE = 16000 # Sample rate expected by LiveConnect input
RECEIVE_SAMPLE_RATE = 24000 # Sample rate provided by LiveConnect output
MODEL = "models/gemini-2.0-flash-live-001" # Use the appropriate live model

# Configure LiveConnect (audio only response)
CONFIG = types.LiveConnectConfig(
    response_modalities=["audio", "text"], # Request both audio and text transcript
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck") # Choose voice
        )
    ),
    audio_processing_config=types.AudioProcessingConfig(
        input_processing_config=types.InputAudioProcessingConfig(
            # Adjust input processing if needed, e.g., noise reduction
        ),
        output_processing_config=types.OutputAudioProcessingConfig(
            output_sample_rate_hertz=RECEIVE_SAMPLE_RATE,
            # Adjust output processing if needed, e.g., volume normalization
        )
    )
)

class LiveConnectManager:
    def __init__(self):
        self._client = None
        self._session = None
        self._audio_in_queue = asyncio.Queue() # Queue for audio from LiveConnect
        self._text_out_queue = asyncio.Queue() # Queue for text from LiveConnect
        self._audio_out_queue = asyncio.Queue() # Queue for audio TO LiveConnect
        self._tasks = []
        self._is_running = False
        self._api_key = os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        # Ensure client is configured for the right API version if needed
        genai.configure(api_key=self._api_key)
        self._client = genai.GenerativeModel(MODEL) # Or use genai.Client as in sample

    async def _receive_audio_loop(self):
        """Receives audio and text from the LiveConnect session."""
        try:
            while self._is_running and self._session:
                turn = self._session.receive()
                async for response in turn:
                    if not self._is_running: break # Check flag again
                    if data := response.data:
                        await self._audio_in_queue.put(data) # Put raw audio bytes
                    if text := response.text:
                        await self._text_out_queue.put(text) # Put transcribed text
                # Handle turn completion if needed (e.g., clear queues on interruption)
        except Exception as e:
            logger.error(f"Error in receive loop: {e}")
        finally:
            logger.info("Receive loop finished.")

    async def _send_audio_loop(self):
        """Sends audio from the output queue to the LiveConnect session."""
        try:
            while self._is_running and self._session:
                audio_chunk = await self._audio_out_queue.get()
                if audio_chunk is None: # Use None as a signal to stop
                    break
                if self._is_running: # Check again before sending
                     await self._session.send(input={"data": audio_chunk, "mime_type": "audio/pcm"})
                self._audio_out_queue.task_done()
        except Exception as e:
            logger.error(f"Error in send loop: {e}")
        finally:
            logger.info("Send loop finished.")

    async def start_session(self):
        """Starts the LiveConnect session and background tasks."""
        if self._is_running:
            logger.warning("Session already running.")
            return

        self._is_running = True
        try:
            # Use a context manager for the session
            # Note: Direct async context manager might be tricky with external start/stop
            # Adapt based on google-genai library specifics for managing session lifecycle
            # This part might need adjustment based on how the async context manager
            # behaves when not used directly in an `async with`.
            # Falling back to manual connect/close might be needed.

            # Placeholder: Assuming direct connect method exists or adapting from sample
            self._session = await self._client.connect_live(config=CONFIG) # Simplified, check library docs
            logger.info("LiveConnect session started.")

            # Start background tasks
            self._tasks.append(asyncio.create_task(self._receive_audio_loop()))
            self._tasks.append(asyncio.create_task(self._send_audio_loop()))

        except Exception as e:
            logger.error(f"Failed to start LiveConnect session: {e}")
            self._is_running = False
            self._session = None
            raise

    async def stop_session(self):
        """Stops the LiveConnect session and background tasks."""
        if not self._is_running:
            return

        self._is_running = False
        logger.info("Stopping LiveConnect session...")

        # Signal send loop to stop
        await self._audio_out_queue.put(None)

        # Cancel and await tasks
        for task in self._tasks:
            task.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []

        # Clear queues (optional)
        while not self._audio_in_queue.empty(): self._audio_in_queue.get_nowait()
        while not self._text_out_queue.empty(): self._text_out_queue.get_nowait()
        while not self._audio_out_queue.empty(): self._audio_out_queue.get_nowait()


        # Close the session if needed (check library documentation for explicit close)
        if self._session and hasattr(self._session, 'close'):
             await self._session.close() # Hypothetical close method
             logger.info("LiveConnect session explicitly closed.")
        elif self._session:
             # If no explicit close, rely on context manager semantics or GC
             logger.info("LiveConnect session implicitly closed or left to GC.")

        self._session = None
        logger.info("LiveConnectManager stopped.")

    async def send_audio_chunk(self, chunk: bytes):
        """Adds an audio chunk to the queue to be sent."""
        if self._is_running:
            await self._audio_out_queue.put(chunk)

    async def get_received_audio_chunk(self):
        """Gets a received audio chunk (non-blocking)."""
        try:
            return self._audio_in_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def get_transcription(self):
         """Gets transcribed text (non-blocking)."""
         try:
             return self._text_out_queue.get_nowait()
         except asyncio.QueueEmpty:
             return None