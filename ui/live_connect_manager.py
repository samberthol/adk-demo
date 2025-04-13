# ui/live_connect_manager.py
import asyncio
import os
from google import genai
from google.genai import types
import logging
import os # Make sure os is imported if not already

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration (Sample rates might be needed elsewhere, e.g., in AudioProcessor)
CHANNELS = 1
SEND_SAMPLE_RATE = 16000 # Sample rate expected by LiveConnect input (informational)
RECEIVE_SAMPLE_RATE = 24000 # Sample rate provided by LiveConnect output (informational)
MODEL = "models/gemini-2.0-flash-live-001" # Use the appropriate live model

# Configure LiveConnect
# Removed audio_processing_config to avoid AttributeError with current library version
CONFIG = types.LiveConnectConfig(
    response_modalities=["audio", "text"], # Request both audio and text transcript
    speech_config=types.SpeechConfig(
        voice_config=types.VoiceConfig(
            prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck") # Choose voice
        )
    ),
)

class LiveConnectManager:
    def __init__(self):
        self._client = None
        self._session = None
        self._audio_in_queue = asyncio.Queue() # Queue for audio bytes from LiveConnect
        self._text_out_queue = asyncio.Queue() # Queue for text transcript from LiveConnect
        self._audio_out_queue = asyncio.Queue() # Queue for audio bytes TO LiveConnect
        self._tasks = []
        self._is_running = False

        # Check for API key env var (still good practice)
        self._api_key = os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        # --- CORRECTED INITIALIZATION ---
        # Use genai.Client() as shown in the LiveAPI sample.
        # Authentication should be handled automatically via the environment variable.
        try:
             # Explicitly pass the API key read from the environment variable
             self._client = genai.Client(api_key=self._api_key)
             logger.info("genai.Client initialized with provided API key.")
        except Exception as e:
             logger.error(f"Failed to initialize genai.Client: {e}", exc_info=True)
             raise # Re-raise critical initialization error
        # --- END OF CORRECTION ---

    async def _receive_audio_loop(self):
        """Receives audio and text from the LiveConnect session."""
        # ... (no changes needed in this method) ...
        try:
            while self._is_running and self._session:
                turn = self._session.receive()
                async for response in turn:
                    if not self._is_running: break
                    if data := response.data:
                        await self._audio_in_queue.put(data)
                    if text := response.text:
                        await self._text_out_queue.put(text)
        except asyncio.CancelledError:
            logger.info("Receive loop cancelled.")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}", exc_info=True)
        finally:
            logger.info("Receive loop finished.")


    async def _send_audio_loop(self):
        """Sends audio byte chunks from the outgoing queue to the LiveConnect session."""
        # ... (no changes needed in this method) ...
        try:
            while self._is_running and self._session:
                audio_chunk = await self._audio_out_queue.get()
                if audio_chunk is None:
                    break
                if self._is_running:
                    await self._session.send(input={"data": audio_chunk, "mime_type": "audio/pcm"})
                self._audio_out_queue.task_done()
        except asyncio.CancelledError:
             logger.info("Send loop cancelled.")
        except Exception as e:
            logger.error(f"Error in send loop: {e}", exc_info=True)
        finally:
            logger.info("Send loop finished.")


    async def start_session(self):
        """Starts the LiveConnect session and background tasks."""
        if self._is_running:
            logger.warning("Session already running.")
            return
        if not self._client:
             logger.error("Cannot start session: genai.Client was not initialized.")
             raise RuntimeError("genai.Client failed to initialize.")

        self._is_running = True
        try:
            # --- CORRECTED SESSION START ---
            # Use the client.aio.live.connect pattern from the sample
            # Note: The sample used 'async with', we need the explicit call to get the session object
            # The exact path might vary slightly based on library structure, but this mirrors the sample.
            # Check if 'aio' attribute exists if issues persist
            if not hasattr(self._client, 'aio') or not hasattr(self._client.aio, 'live') or not hasattr(self._client.aio.live, 'connect'):
                 logger.error("Client object does not have expected 'aio.live.connect' structure.")
                 # Fallback or alternative attempt if needed:
                 # model_client = self._client.get_model(MODEL) # Try getting model specific client first?
                 # self._session = await model_client.connect_live(config=CONFIG)
                 raise AttributeError("Correct method to initiate live connection not found on client object.")

            self._session = await self._client.aio.live.connect(model=MODEL, config=CONFIG)
            # --- END OF CORRECTION ---

            logger.info("LiveConnect session started successfully.")

            self._tasks = []
            self._tasks.append(asyncio.create_task(self._receive_audio_loop()))
            self._tasks.append(asyncio.create_task(self._send_audio_loop()))
            logger.info("Send/Receive loops created.")

        except Exception as e:
            logger.error(f"Failed to start LiveConnect session: {e}", exc_info=True)
            self._is_running = False
            self._session = None
            raise

    async def stop_session(self):
        """Stops the LiveConnect session and background tasks gracefully."""
        # ... (no changes needed in this method, relies on self._session) ...
        if not self._is_running:
            logger.info("Stop session called but not running.")
            return

        self._is_running = False # Signal loops to stop checking
        logger.info("Stopping LiveConnect session...")

        try:
            self._audio_out_queue.put_nowait(None)
        except asyncio.QueueFull:
             logger.warning("Audio out queue full during stop signal, sender might not stop cleanly.")
        except Exception as e:
            logger.error(f"Error putting None sentinel in audio_out_queue: {e}")

        for task in self._tasks:
            if not task.done():
                task.cancel()

        try:
             results = await asyncio.wait_for(asyncio.gather(*self._tasks, return_exceptions=True), timeout=5.0)
             logger.info(f"Gather results on stop: {results}")
        except asyncio.TimeoutError:
             logger.warning("Timeout waiting for tasks to finish during stop_session.")
        except Exception as e:
             logger.error(f"Error during task gathering in stop_session: {e}")
        finally:
             self._tasks = []

        while not self._audio_in_queue.empty(): self._audio_in_queue.get_nowait()
        while not self._text_out_queue.empty(): self._text_out_queue.get_nowait()
        while not self._audio_out_queue.empty(): self._audio_out_queue.get_nowait()

        if self._session and hasattr(self._session, 'close') and callable(self._session.close):
            try:
                if asyncio.iscoroutinefunction(self._session.close):
                     await self._session.close()
                else:
                     self._session.close() # Synchronous close
                logger.info("LiveConnect session explicitly closed.")
            except Exception as e:
                 logger.error(f"Error closing LiveConnect session: {e}")
        elif self._session:
            logger.info("No explicit close method found on session object, relying on connection termination.")

        self._session = None
        logger.info("LiveConnectManager stopped.")


    async def send_audio_chunk(self, chunk: bytes):
        """Adds an audio chunk (bytes) to the queue to be sent to Gemini."""
        # ... (no changes needed in this method) ...
        if self._is_running:
            try:
                await self._audio_out_queue.put(chunk)
            except Exception as e:
                 logger.error(f"Failed to put audio chunk in queue: {e}")


    async def get_received_audio_chunk(self) -> bytes | None:
        """Gets a received audio chunk (bytes) from Gemini (non-blocking)."""
        # ... (no changes needed in this method) ...
        try:
            return self._audio_in_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def get_transcription(self) -> str | None:
        """Gets transcribed text from Gemini (non-blocking)."""
        # ... (no changes needed in this method) ...
        try:
            return self._text_out_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None