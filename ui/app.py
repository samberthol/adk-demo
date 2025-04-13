# ui/app.py
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import asyncio
import threading
# queue module not strictly needed with current async/queue usage in manager
# import queue
import av # PyAV library used by streamlit-webrtc
import logging
import time
# Import SciPy for resampling - MAKE SURE TO ADD 'scipy' to requirements.txt
try:
    from scipy.signal import resample_poly
except ImportError:
    st.error("SciPy not found. Please add 'scipy' to requirements.txt and rebuild the container.")
    st.stop()


# Assuming live_connect_manager.py is correct and doesn't import pyaudio
try:
    # Import constants including CHANNELS
    from live_connect_manager import LiveConnectManager, SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE, CHANNELS
except ImportError:
    try:
        from .live_connect_manager import LiveConnectManager, SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE, CHANNELS
    except ImportError as e:
         st.error(f"Failed to import LiveConnectManager or constants. Check path and file existence. Error: {e}")
         st.stop()


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(threadName)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Constants ---
# WebRTC Configuration
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
MEDIA_STREAM_CONSTRAINTS = {"audio": True, "video": False}
# Audio Processing Configuration
AUDIO_CHUNK_DURATION_MS = 20 # Process audio in X ms chunks
BYTES_PER_SAMPLE = 2 # Corresponds to s16 format

# Calculate chunk size based on duration and sample rate for outgoing audio (to Gemini)
SEND_CHUNK_SIZE = int(SEND_SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE * (AUDIO_CHUNK_DURATION_MS / 1000.0))
# Calculate chunk size for silence generation for incoming audio (from Gemini)
RECEIVE_CHUNK_SIZE = int(RECEIVE_SAMPLE_RATE * CHANNELS * BYTES_PER_SAMPLE * (AUDIO_CHUNK_DURATION_MS / 1000.0))
RECEIVE_SILENCE_SAMPLES = RECEIVE_CHUNK_SIZE // (CHANNELS * BYTES_PER_SAMPLE)

logger.info(f"Send Chunk Size: {SEND_CHUNK_SIZE} bytes ({AUDIO_CHUNK_DURATION_MS}ms at {SEND_SAMPLE_RATE}Hz)")
logger.info(f"Receive Chunk Size (Silence): {RECEIVE_CHUNK_SIZE} bytes ({AUDIO_CHUNK_DURATION_MS}ms at {RECEIVE_SAMPLE_RATE}Hz)")


# --- Refactored Asyncio Event Loop Management using Session State ---

def ensure_manager():
    """Gets or creates the LiveConnectManager instance in session state."""
    if 'live_connect_manager' not in st.session_state:
        logger.info("Creating LiveConnectManager instance in session state.")
        try:
            st.session_state.live_connect_manager = LiveConnectManager()
        except ValueError as e:
            st.error(f"Failed to initialize LiveConnectManager: {e}. Is GEMINI_API_KEY env var set?")
            logger.error(f"Failed to initialize LiveConnectManager: {e}", exc_info=True)
            st.stop() # Stop execution if manager fails init
        except Exception as e:
            st.error(f"Unexpected error initializing LiveConnectManager: {e}")
            logger.error(f"Unexpected error initializing LiveConnectManager: {e}", exc_info=True)
            st.stop() # Stop execution
    return st.session_state.live_connect_manager

def run_manager_in_thread(loop, manager):
    """Target function for the background asyncio thread."""
    thread_name = threading.current_thread().name
    logger.info(f"Asyncio thread '{thread_name}' started. Setting event loop.")
    asyncio.set_event_loop(loop)
    st.session_state.manager_task_running = True
    try:
        logger.info(f"Asyncio thread '{thread_name}': Starting manager session...")
        loop.run_until_complete(manager.start_session())
        logger.info(f"Asyncio thread '{thread_name}': Manager session started (or attempted).")

        st.session_state.stop_future = loop.create_future()
        logger.info(f"Asyncio thread '{thread_name}': Created stop_future, awaiting signal...")
        loop.run_until_complete(st.session_state.stop_future)
        logger.info(f"Asyncio thread '{thread_name}': stop_future completed.")

    except Exception as e:
        logger.error(f"Exception in asyncio thread '{thread_name}': {e}", exc_info=True)
    finally:
        logger.info(f"Asyncio thread '{thread_name}': Loop finishing.")
        st.session_state.manager_task_running = False
        logger.info(f"Asyncio thread '{thread_name}': Exiting.")


def start_asyncio_thread():
    """Starts the asyncio thread if not already running for this session."""
    if not st.session_state.get("asyncio_thread_started", False):
        logger.info("Request received to start asyncio background thread...")
        manager = ensure_manager()
        if not manager:
             logger.error("Cannot start thread, manager instance is missing.")
             return

        st.session_state.event_loop = asyncio.new_event_loop()
        st.session_state.manager_thread = threading.Thread(
            target=run_manager_in_thread,
            args=(st.session_state.event_loop, manager),
            daemon=True,
            name="AsyncioMgrThread" # Give thread a name for logging
        )
        st.session_state.manager_thread.start()
        st.session_state.asyncio_thread_started = True # Set flag immediately
        logger.info("Asyncio thread starting...")
        time.sleep(1) # Give thread time to start loop and manager session
        if not st.session_state.get('manager_task_running', False):
             logger.warning("Asyncio thread started but manager task start confirmation flag not set yet.")
    else:
        logger.debug("Asyncio thread start requested, but flag indicates it's already started.")


def stop_asyncio_thread():
    """Signals the asyncio thread and associated tasks to stop."""
    if st.session_state.get("asyncio_thread_started", False):
        logger.info("Request received to stop asyncio background thread...")
        loop = st.session_state.get('event_loop')
        manager = st.session_state.get('live_connect_manager')
        manager_thread = st.session_state.get('manager_thread')

        # Ensure loop/manager exists before trying to interact
        if loop and manager:
             if loop.is_running():
                 # Signal the main loop runner (stop_future) to exit first
                 stop_future = st.session_state.get('stop_future')
                 if stop_future and not stop_future.done():
                     logger.info("Signaling stop_future...")
                     loop.call_soon_threadsafe(stop_future.set_result, True)
                 else:
                     logger.warning("Could not signal stop_future (future missing or already done).")

                 # Then, stop the manager's internal session and tasks
                 if manager._is_running:
                     logger.info("Calling manager.stop_session()...")
                     future = asyncio.run_coroutine_threadsafe(manager.stop_session(), loop)
                     try:
                         future.result(timeout=5) # Wait briefly for graceful shutdown
                         logger.info("Manager stop_session() completed or timed out.")
                     except Exception as e:
                         logger.error(f"Error or timeout waiting for manager.stop_session: {e}")
                 else:
                      logger.info("Manager was not running when stop was called.")
             else:
                  logger.warning("Stop requested, but asyncio loop was not running.")
        else:
            logger.warning("Stop requested, but loop or manager not found in session state.")

        # Finally, wait for the thread to finish
        if manager_thread and manager_thread.is_alive():
            logger.info("Waiting for asyncio thread to join...")
            manager_thread.join(timeout=7.0)
            if manager_thread.is_alive():
                logger.warning("Asyncio thread did not stop gracefully after join.")
            else:
                 logger.info("Asyncio thread joined successfully.")
        else:
             logger.info("Asyncio thread was not alive or found for joining.")

        # Clean up state variables
        st.session_state.pop('manager_thread', None)
        st.session_state.pop('event_loop', None)
        st.session_state.pop('stop_future', None)
        st.session_state.pop('manager_task_running', None)
        st.session_state.asyncio_thread_started = False # Reset flag
        logger.info("Asyncio thread state cleared.")
    else:
         logger.debug("Stop asyncio thread requested, but flag indicates it wasn't started.")

# --- Streamlit Audio Processor ---
class LiveConnectAudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self._buffer = bytearray()
        # Use constant for target chunk size calculation
        self._target_chunk_size = SEND_CHUNK_SIZE
        # Store references from session state during init
        self.loop = st.session_state.get('event_loop')
        self.manager = st.session_state.get('live_connect_manager')
        logger.info(f"AudioProcessor initialized. Loop is set: {self.loop is not None}. Manager is set: {self.manager is not None}")
        if self.loop:
             logger.info(f"AudioProcessor.__init__: Loop running: {self.loop.is_running()}")
        if self.manager:
             logger.info(f"AudioProcessor.__init__: Manager running: {self.manager._is_running}")

    def _generate_silence(self) -> np.ndarray:
        """Generates a silent numpy array for the output frame size."""
        return np.zeros((RECEIVE_SILENCE_SAMPLES, CHANNELS), dtype=np.int16)

    def _create_error_frame(self) -> av.AudioFrame:
         """Creates a silent AudioFrame for returning on errors."""
         silent_ndarray = self._generate_silence()
         error_frame = av.AudioFrame.from_ndarray(silent_ndarray, format='s16', layout='mono' if CHANNELS==1 else 'stereo')
         error_frame.sample_rate = RECEIVE_SAMPLE_RATE
         return error_frame

    async def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Processes audio frames from WebRTC, sends to LiveConnect, returns audio for playback."""
        # Check if loop and manager are ready before processing
        if not self.loop or not self.manager or not self.manager._is_running or not self.loop.is_running():
             # Log periodically if issues persist
             if time.time() % 5 < 0.1:
                 logger.warning(f"Asyncio loop/manager not ready in recv. Returning silence.")
             return self._create_error_frame()

        # 1. Process incoming audio frame
        in_data = None # Initialize in_data
        try:
            # --- Input Validation and Resampling ---
            if frame.format.name != 's16':
                 logger.error(f"Unsupported incoming audio format: {frame.format.name}. Expecting s16.")
            elif frame.layout.name != ('mono' if CHANNELS == 1 else 'stereo'):
                 logger.error(f"Unsupported incoming audio layout: {frame.layout.name}. Expecting {'mono' if CHANNELS == 1 else 'stereo'}.")
            elif frame.sample_rate != SEND_SAMPLE_RATE:
                 # --- Resampling Implementation ---
                 logger.debug(f"Resampling needed: Input rate {frame.sample_rate}Hz != Target {SEND_SAMPLE_RATE}Hz")
                 try:
                     ndarray_s16 = frame.to_ndarray(format='s16')
                     if CHANNELS != 1:
                          logger.error("Resampling code currently assumes mono input. Skipping frame.")
                     else:
                          num_original_samples = ndarray_s16.shape[0]
                          up = SEND_SAMPLE_RATE
                          down = frame.sample_rate
                          resampled_data = resample_poly(ndarray_s16[:, 0], up, down)
                          in_data = resampled_data.astype(np.int16).tobytes()
                          logger.debug(f"Resampled audio. Original samples: {num_original_samples}, New bytes: {len(in_data)}")
                 except Exception as e:
                     logger.error(f"Error during resampling: {e}", exc_info=True)
                     # in_data remains None if resampling fails
            else:
                 # Format, layout, rate match - convert to bytes
                 in_data = frame.to_ndarray(format='s16').tobytes()

            # --- Send Processed Data ---
            if in_data:
                self._buffer.extend(in_data)
                # Send chunks of the target size
                while len(self._buffer) >= self._target_chunk_size:
                    chunk_to_send = self._buffer[:self._target_chunk_size]
                    del self._buffer[:self._target_chunk_size]
                    # Safely call the async method from this sync context (fire-and-forget)
                    asyncio.run_coroutine_threadsafe(
                         self.manager.send_audio_chunk(chunk_to_send), self.loop
                    ) # Removed .result()
            else:
                 logger.warning("No valid input data (in_data is None) after processing/resampling.")

        except Exception as e:
            logger.error(f"Error processing incoming audio frame: {e}", exc_info=True)

        # 2. Get audio chunk received from LiveConnect for playback
        out_data = None
        try:
            # Get data using run_coroutine_threadsafe as get_received_audio_chunk is async
            future = asyncio.run_coroutine_threadsafe(self.manager.get_received_audio_chunk(), self.loop)
            out_data = future.result(timeout=0.005) # Short timeout ok for non-blocking getter
        except asyncio.TimeoutError:
             pass # No data ready is normal
        except Exception as e:
             logger.error(f"Error getting received audio chunk: {e}")

        # Use silence if no data received
        if not out_data:
            out_data = b'\0' * RECEIVE_CHUNK_SIZE

        # 3. Construct the output AudioFrame
        try:
            bytes_per_sample = 2 * CHANNELS
            if len(out_data) % bytes_per_sample != 0:
                 logger.warning(f"Output data length {len(out_data)} not multiple of bytes per sample {bytes_per_sample}. Truncating.")
                 valid_len = (len(out_data) // bytes_per_sample) * bytes_per_sample
                 out_data = out_data[:valid_len]

            if not out_data: # Handle case where truncation resulted in empty data
                out_data = b'\0' * RECEIVE_CHUNK_SIZE

            new_ndarray = np.frombuffer(out_data, dtype=np.int16)
            expected_samples = len(out_data) // bytes_per_sample
            # Reshape based on CHANNELS constant
            new_ndarray = new_ndarray.reshape(expected_samples, CHANNELS)

            new_frame = av.AudioFrame.from_ndarray(new_ndarray, format='s16', layout='mono' if CHANNELS==1 else 'stereo')
            new_frame.sample_rate = RECEIVE_SAMPLE_RATE
            new_frame.pts = frame.pts # Attempt to preserve PTS
            return new_frame

        except Exception as e:
            logger.error(f"Error constructing output audio frame: {e}", exc_info=True)
            return self._create_error_frame() # Return silence on error

    def on_ended(self):
        """Callback when the WebRTC connection ends."""
        logger.info("AudioProcessor.on_ended called.")
        # Main UI loop handles stopping the background thread now

# --- Streamlit UI ---
st.set_page_config(layout="wide")

st.title("Live Voice Agent Interaction")
st.markdown("Connect your microphone and speak to the agent.")

col1, col2 = st.columns([0.7, 0.3])

with col1:
    status_placeholder = st.empty()
    transcript_placeholder = st.empty()

with col2:
    if st.button("Clear Session State (Debug)"):
        logger.info("Clearing Streamlit session state.")
        stop_asyncio_thread() # Stop thread first
        for key in list(st.session_state.keys()):
            if key != 'query_params': # Preserve query params
                 del st.session_state[key]
        st.rerun()

    st.write("---")
    st.write("**Instructions:**")
    st.write("1. Allow microphone access.")
    st.write("2. Use 'Start/Stop' below.")
    st.write("3. Speak clearly.")
    st.write("4. Stop when finished.")


# --- Main Component Logic ---
# Ensure manager is created/retrieved at the start of each rerun
manager = ensure_manager()
if not manager:
     st.error("LiveConnectManager could not be initialized. Cannot proceed.")
     st.stop()

webrtc_ctx = webrtc_streamer(
    key="live-voice-interaction",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
    audio_processor_factory=LiveConnectAudioProcessor,
    async_processing=True,
)

# --- Manage background thread based on WebRTC component state ---
thread_started = st.session_state.get("asyncio_thread_started", False)

if webrtc_ctx.state.playing:
    status_placeholder.write("Status: ✅ Connected and Listening...")
    if not thread_started:
        start_asyncio_thread() # Start thread only if not already marked as started

    # Display transcriptions dynamically
    full_transcript = st.session_state.get("full_transcript", "")
    while True: # Loop as long as we are playing to update transcript
         if not webrtc_ctx.state.playing:
             logger.info("WebRTC component stopped playing, breaking transcript loop.")
             break # Exit loop if component stops

         new_text = None
         if st.session_state.get("asyncio_thread_started", False): # Check flag again
             loop = st.session_state.get('event_loop')
             current_manager = st.session_state.get('live_connect_manager') # Use current manager instance
             if loop and loop.is_running() and current_manager:
                  try:
                      # Use run_coroutine_threadsafe as get_transcription is async
                      future = asyncio.run_coroutine_threadsafe(current_manager.get_transcription(), loop)
                      new_text = future.result(timeout=0.01)
                  except (asyncio.TimeoutError):
                      pass # No new text is okay
                  except Exception as e:
                      logger.error(f"Error getting transcription in UI loop: {e}")
             else:
                  if time.time() % 10 < 0.1: # Log periodically
                       logger.warning("Transcript check: Component playing but loop/manager not running/ready.")

         if new_text:
             full_transcript += new_text # Append text chunk
             st.session_state.full_transcript = full_transcript # Store updated transcript

         # Update placeholder
         transcript_placeholder.text_area("Transcript:", full_transcript, height=300, key="transcript_display")

         time.sleep(0.1) # Polling interval for transcript updates

else:
    # Component is not playing
    status_placeholder.write("Status: ⏸️ Disconnected / Stopped. (Allow mic & click 'Start' below)")
    if thread_started: # Stop thread only if it was marked as started
         stop_asyncio_thread()
    st.session_state.full_transcript = "" # Clear transcript when stopped
    transcript_placeholder.text_area("Transcript:", "", height=300, key="transcript_display_cleared")


logger.debug(f"Streamlit script execution finished. Playing: {webrtc_ctx.state.playing}, Thread Started Flag: {st.session_state.get('asyncio_thread_started', False)}")