import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase, ClientSettings
import numpy as np
import asyncio
import threading
import queue # Use standard queue for thread-safe communication
import av # PyAV library used by streamlit-webrtc
import logging
import time

from live_connect_manager import LiveConnectManager, SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE # Import manager and constants

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Client settings for WebRTC (can be adjusted)
WEBRTC_CLIENT_SETTINGS = ClientSettings(
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"audio": True, "video": False},
)

# Thread-safe queues for communication between Streamlit thread and Asyncio thread
# Max sizes can be adjusted based on performance/memory
audio_to_send_queue = queue.Queue(maxsize=10)
audio_to_play_queue = queue.Queue(maxsize=10)
text_received_queue = queue.Queue(maxsize=10)

# --- Asyncio Event Loop Management ---
# We run the asyncio part in a separate thread
_event_loop = None
_loop_thread = None
_manager_instance = None

def get_asyncio_loop():
    global _event_loop
    if _event_loop is None:
        _event_loop = asyncio.new_event_loop()
    return _event_loop

def start_asyncio_thread():
    global _loop_thread, _manager_instance
    if _loop_thread is None or not _loop_thread.is_alive():
        loop = get_asyncio_loop()
        _manager_instance = LiveConnectManager() # Create manager instance here
        _loop_thread = threading.Thread(target=run_async_tasks, args=(loop, _manager_instance), daemon=True)
        _loop_thread.start()
        logger.info("Asyncio thread started.")
        # Give the loop a moment to start
        time.sleep(0.5)


def run_async_tasks(loop, manager):
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(async_task_manager(manager))
    finally:
        loop.close()
        logger.info("Asyncio loop closed.")

async def async_task_manager(manager: LiveConnectManager):
    """Manages starting the session and processing queues."""
    try:
        await manager.start_session()
        # Start queue processing tasks
        sender_task = asyncio.create_task(process_audio_to_send(manager))
        receiver_task = asyncio.create_task(process_received_data(manager))
        await asyncio.gather(sender_task, receiver_task)
    except Exception as e:
         logger.error(f"Error in async_task_manager: {e}", exc_info=True)
    finally:
        logger.info("Stopping session from async_task_manager...")
        # Ensure session stops even if tasks fail
        if manager._is_running:
             await manager.stop_session()


async def process_audio_to_send(manager: LiveConnectManager):
    """Task to get audio from thread-safe queue and send via manager."""
    while manager._is_running:
        try:
            # Use asyncio.to_thread for blocking queue.get
            chunk = await asyncio.to_thread(audio_to_send_queue.get, timeout=0.1)
            if chunk is None: # Stop signal
                break
            await manager.send_audio_chunk(chunk)
            audio_to_send_queue.task_done()
        except queue.Empty:
            await asyncio.sleep(0.01) # Prevent busy-waiting
        except Exception as e:
            logger.error(f"Error sending audio chunk: {e}")
            await asyncio.sleep(0.1)

async def process_received_data(manager: LiveConnectManager):
     """Task to get data from manager queues and put into thread-safe queues."""
     while manager._is_running:
        try:
             audio_chunk = await manager.get_received_audio_chunk()
             if audio_chunk:
                 try:
                     audio_to_play_queue.put_nowait(audio_chunk)
                 except queue.Full:
                     logger.warning("Audio playback queue full, dropping chunk.")

             text_chunk = await manager.get_transcription()
             if text_chunk:
                 try:
                      text_received_queue.put_nowait(text_chunk)
                 except queue.Full:
                      logger.warning("Text display queue full, dropping text.")

             await asyncio.sleep(0.01) # Small sleep to yield control
        except Exception as e:
             logger.error(f"Error processing received data: {e}")
             await asyncio.sleep(0.1)


def stop_asyncio_thread():
    global _event_loop, _loop_thread, _manager_instance
    if _loop_thread and _loop_thread.is_alive() and _event_loop and _event_loop.is_running() and _manager_instance:
        logger.info("Requesting asyncio thread stop...")
        # Signal the manager to stop via the event loop
        asyncio.run_coroutine_threadsafe(_manager_instance.stop_session(), _event_loop)
        # Signal the sender queue processing task to stop
        audio_to_send_queue.put(None)

        _loop_thread.join(timeout=5.0) # Wait for thread to finish
        if _loop_thread.is_alive():
             logger.warning("Asyncio thread did not stop gracefully.")
    else:
         logger.info("Asyncio thread already stopped or not running.")

    _loop_thread = None
    _event_loop = None
    _manager_instance = None
    # Clear queues
    while not audio_to_send_queue.empty(): audio_to_send_queue.get_nowait()
    while not audio_to_play_queue.empty(): audio_to_play_queue.get_nowait()
    while not text_received_queue.empty(): text_received_queue.get_nowait()


# --- Streamlit Audio Processor ---
class LiveConnectAudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self._buffer = bytearray()
        # Required samples per frame for LiveConnect input (16kHz mono)
        # Calculate bytes per frame: 10ms frame duration at 16kHz, 16-bit mono
        self._target_chunk_size = int(SEND_SAMPLE_RATE * 1 * 2 * 0.010) # ~320 bytes
        logger.info(f"Target chunk size for sending: {self._target_chunk_size} bytes")


    async def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Processes audio frames from WebRTC, sends to LiveConnect, returns audio for playback."""
        # Ensure the asyncio thread is running
        start_asyncio_thread()

        # 1. Process incoming audio frame (Resample to SEND_SAMPLE_RATE if needed)
        try:
            # Resample the frame to the sample rate LiveConnect expects (SEND_SAMPLE_RATE)
            # PyAV's resampler can be complex; direct conversion might be simpler if rates match or close
            # Assuming input frame is already PCM S16 mono for simplicity here.
            # Need to handle format/rate conversion robustly in production.

            # Convert frame data to bytes (assuming format='s16', layout='mono')
            # Check frame.format.name and frame.layout.name
            if frame.format.name != 's16':
                 logger.warning(f"Unexpected frame format: {frame.format.name}")
                 # Add conversion logic if necessary
            if frame.layout.name != 'mono':
                  logger.warning(f"Unexpected frame layout: {frame.layout.name}")
                  # Add conversion logic if necessary

            # Extract data as bytes
            in_data = frame.to_ndarray(format='s16').tobytes()
            self._buffer.extend(in_data)

            # Send chunks of the target size
            while len(self._buffer) >= self._target_chunk_size:
                chunk_to_send = self._buffer[:self._target_chunk_size]
                del self._buffer[:self._target_chunk_size]
                try:
                    # Put data into the thread-safe queue for the asyncio thread
                    audio_to_send_queue.put_nowait(chunk_to_send)
                except queue.Full:
                    logger.warning("Audio sending queue full, dropping chunk.")

        except Exception as e:
            logger.error(f"Error processing incoming audio frame: {e}", exc_info=True)


        # 2. Get audio chunk received from LiveConnect for playback
        out_data = None
        try:
            # Get data from the thread-safe queue populated by the asyncio thread
            out_data = audio_to_play_queue.get_nowait()
            audio_to_play_queue.task_done()
        except queue.Empty:
            # Send silence if no audio is ready from LiveConnect
            # Calculate silence buffer size based on expected output frame duration/rate
            # Example: 10ms frame at 24kHz, 16-bit mono = 24000 * 1 * 2 * 0.010 = 480 bytes
            silence_chunk_size = int(RECEIVE_SAMPLE_RATE * 1 * 2 * 0.010)
            out_data = b'\0' * silence_chunk_size


        # 3. Construct the output AudioFrame for playback
        # Ensure the data format matches what WebRTC expects (usually S16 PCM)
        # The sample rate MUST match RECEIVE_SAMPLE_RATE (24kHz)
        try:
            # Convert bytes back to numpy array (assuming s16 mono)
            num_samples = len(out_data) // 2 # 2 bytes per sample for s16
            if len(out_data) % 2 != 0:
                 logger.warning("Received odd number of bytes for s16 format, padding.")
                 out_data += b'\0' # Add null byte if odd length

            new_ndarray = np.frombuffer(out_data, dtype=np.int16)

            # Reshape for mono layout (samples, channels)
            if new_ndarray.size > 0:
                new_ndarray = new_ndarray.reshape(-1, 1)
            else:
                # Handle empty data case
                silence_samples = silence_chunk_size // 2
                new_ndarray = np.zeros((silence_samples, 1), dtype=np.int16)


            # Create the new frame
            new_frame = av.AudioFrame.from_ndarray(new_ndarray, format='s16', layout='mono')
            new_frame.sample_rate = RECEIVE_SAMPLE_RATE # Crucial: Set output sample rate
            new_frame.pts = frame.pts # Preserve presentation timestamp if possible

            return new_frame

        except Exception as e:
             logger.error(f"Error constructing output audio frame: {e}", exc_info=True)
             # Return an empty/silent frame on error
             silence_chunk_size = int(RECEIVE_SAMPLE_RATE * 1 * 2 * 0.010)
             silence_samples = silence_chunk_size // 2
             silent_ndarray = np.zeros((silence_samples, 1), dtype=np.int16)
             error_frame = av.AudioFrame.from_ndarray(silent_ndarray, format='s16', layout='mono')
             error_frame.sample_rate = RECEIVE_SAMPLE_RATE
             return error_frame

    def on_ended(self):
        """Callback when the WebRTC connection ends."""
        logger.info("WebRTC connection ended. Cleaning up...")
        stop_asyncio_thread()
        logger.info("Cleanup complete.")

# --- Streamlit UI ---
st.title("Live Voice Agent Interaction")
st.write("Click 'Start' to connect your microphone and talk to the agent.")

# Placeholder for displaying transcriptions
transcript_placeholder = st.empty()
full_transcript = ""


# Start the WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="live-voice",
    mode=WebRtcMode.SENDRECV,
    client_settings=WEBRTC_CLIENT_SETTINGS,
    audio_processor_factory=LiveConnectAudioProcessor,
    async_processing=True, # Use async processing
    # rtc_configuration={"iceTransportPolicy": "relay"} # Uncomment if behind strict NAT/Firewall
)

if webrtc_ctx.state.playing:
    st.write("Status: Connected and listening...")
    # Continuously check the text queue and update the display
    while True:
         try:
              text = text_received_queue.get_nowait()
              full_transcript += text + " "
              transcript_placeholder.text_area("Transcript:", full_transcript, height=200)
              text_received_queue.task_done()
         except queue.Empty:
              # Break the loop if the streamer stops playing
              if not webrtc_ctx.state.playing:
                   break
              time.sleep(0.1) # Prevent busy-waiting
else:
    st.write("Status: Disconnected. Click Start.")
    # Ensure cleanup if stopped externally
    stop_asyncio_thread()
    full_transcript = "" # Clear transcript when stopped
    transcript_placeholder.text_area("Transcript:", full_transcript, height=200)

st.write("---")
st.write("Remember to stop the connection when finished.")

# Ensure cleanup on app exit/rerun
# Note: Streamlit's execution model can make perfect cleanup tricky.
# Using daemon threads helps, but explicit stop on session end is best.