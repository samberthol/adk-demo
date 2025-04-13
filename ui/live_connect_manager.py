# ui/live_connect_manager.py
import asyncio
import os
from google import genai
from google.genai import types
import logging

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
    # audio_processing_config parameter removed here
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
        self._api_key = os.getenv("GEMINI_API_KEY")
        if not self._api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")

        # Ensure client is configured for the right API version if needed
        genai.configure(api_key=self._api_key)
        self._client = genai.GenerativeModel(MODEL) # Or use genai.Client if needed for specific API version

    async def _receive_audio_loop(self):
        """Receives audio and text from the LiveConnect session."""
        try:
            while self._is_running and self._session:
                turn = self._session.receive()
                async for response in turn:
                    if not self._is_running: break # Check flag again
                    if data := response.data:
                        # Put raw audio bytes received from Gemini into the queue
                        await self._audio_in_queue.put(data)
                    if text := response.text:
                        # Put transcribed text received from Gemini into the queue
                        await self._text_out_queue.put(text)
                # Handle turn completion if needed (e.g., clear queues on interruption)
        except asyncio.CancelledError:
            logger.info("Receive loop cancelled.")
        except Exception as e:
            logger.error(f"Error in receive loop: {e}", exc_info=True)
        finally:
            logger.info("Receive loop finished.")

    async def _send_audio_loop(self):
        """Sends audio byte chunks from the outgoing queue to the LiveConnect session."""
        try:
            while self._is_running and self._session:
                # Wait for an audio chunk from the calling code (e.g., AudioProcessor)
                audio_chunk = await self._audio_out_queue.get()
                if audio_chunk is None: # Use None as a signal to stop
                    break
                if self._is_running: # Check again before sending
                    # Send the raw audio bytes
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

        self._is_running = True
        try:
            # Connect to the LiveConnect service using the configured client and CONFIG
            # Ensure the connect_live method is appropriate for GenerativeModel or use genai.Client
            self._session = await self._client.connect_live(config=CONFIG)
            logger.info("LiveConnect session started successfully.")

            # Clear any stale tasks before starting new ones
            self._tasks = []
            # Start background tasks for sending and receiving
            self._tasks.append(asyncio.create_task(self._receive_audio_loop()))
            self._tasks.append(asyncio.create_task(self._send_audio_loop()))
            logger.info("Send/Receive loops created.")

        except Exception as e:
            logger.error(f"Failed to start LiveConnect session: {e}", exc_info=True)
            self._is_running = False
            self._session = None
            # Re-raise the exception so the calling code knows about the failure
            raise

    async def stop_session(self):
        """Stops the LiveConnect session and background tasks gracefully."""
        if not self._is_running:
            logger.info("Stop session called but not running.")
            return

        self._is_running = False # Signal loops to stop checking
        logger.info("Stopping LiveConnect session...")

        # Signal send loop to stop by putting None in the queue
        try:
            await self._audio_out_queue.put(None)
        except Exception as e:
            logger.error(f"Error putting None sentinel in audio_out_queue: {e}")


        # Cancel currently running async tasks
        for task in self._tasks:
            if not task.done():
                task.cancel()

        # Await task completion (or cancellation)
        # Use a timeout to prevent hanging indefinitely
        try:
             results = await asyncio.wait_for(asyncio.gather(*self._tasks, return_exceptions=True), timeout=5.0)
             logger.info(f"Gather results: {results}")
        except asyncio.TimeoutError:
             logger.warning("Timeout waiting for tasks to finish during stop_session.")
        except Exception as e:
             logger.error(f"Error during task gathering in stop_session: {e}")
        finally:
             self._tasks = [] # Clear the task list


        # Clear queues (optional, but good practice)
        while not self._audio_in_queue.empty(): self._audio_in_queue.get_nowait()
        while not self._text_out_queue.empty(): self._text_out_queue.get_nowait()
        while not self._audio_out_queue.empty(): self._audio_out_queue.get_nowait()


        # Close the session if the library provides an explicit close method
        # (This depends on the specific session object returned by connect_live)
        if self._session and hasattr(self._session, 'close') and callable(self._session.close):
            try:
                # Check if it's an async close
                if asyncio.iscoroutinefunction(self._session.close):
                     await self._session.close()
                else:
                     self._session.close() # Synchronous close
                logger.info("LiveConnect session explicitly closed.")
            except Exception as e:
                 logger.error(f"Error closing LiveConnect session: {e}")
        elif self._session:
            # If no explicit close, log that we rely on connection termination or GC
            logger.info("No explicit close method found on session object, relying on connection termination.")

        self._session = None # Release the session object reference
        logger.info("LiveConnectManager stopped.")

    async def send_audio_chunk(self, chunk: bytes):
        """Adds an audio chunk (bytes) to the queue to be sent to Gemini."""
        if self._is_running:
            try:
                # Consider adding a timeout to put_nowait or using put with timeout
                # if backpressure is a concern
                await self._audio_out_queue.put(chunk)
            except Exception as e:
                 logger.error(f"Failed to put audio chunk in queue: {e}")


    async def get_received_audio_chunk(self) -> bytes | None:
        """Gets a received audio chunk (bytes) from Gemini (non-blocking)."""
        try:
            return self._audio_in_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def get_transcription(self) -> str | None:
        """Gets transcribed text from Gemini (non-blocking)."""
        try:
            return self._text_out_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None