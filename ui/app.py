# ui/app.py
import streamlit as st
# Ensure ClientSettings is NOT imported
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import asyncio
import threading
import queue # Standard queue is not used in this refactor, but kept import just in case
import av # PyAV library used by streamlit-webrtc
import logging
import time

# Assuming live_connect_manager.py is correct and doesn't import pyaudio
# Need to ensure this import path is correct relative to app.py
try:
    from live_connect_manager import LiveConnectManager, SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE
except ImportError:
    # Handle case where the file might be in the same directory or adjust path
    try:
        from .live_connect_manager import LiveConnectManager, SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE
    except ImportError as e:
         st.error(f"Failed to import LiveConnectManager. Check path and file existence. Error: {e}")
         st.stop()


# Configure logging
# Consider configuring format and level more formally if needed
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# WebRTC Configuration
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
MEDIA_STREAM_CONSTRAINTS = {"audio": True, "video": False}

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
    asyncio.set_event_loop(loop)
    st.session_state.manager_task_running = True
    logger.info("Asyncio thread started. Setting event loop.")
    try:
        logger.info("Asyncio thread: Starting manager session...")
        # Run start_session, which should initialize and start background tasks within manager
        loop.run_until_complete(manager.start_session())
        logger.info("Asyncio thread: Manager session started (or attempted).")

        # Create a future that will be awaited until stop is signaled
        st.session_state.stop_future = loop.create_future()
        logger.info("Asyncio thread: Created stop_future, awaiting signal...")
        # This keeps the loop running, processing manager's background tasks
        loop.run_until_complete(st.session_state.stop_future)
        logger.info("Asyncio thread: stop_future completed.")

    except Exception as e:
        # Log exceptions occurring within the thread's main execution
        logger.error(f"Exception in asyncio thread run_manager_in_thread: {e}", exc_info=True)
    finally:
        logger.info("Asyncio thread: Loop finishing.")
        # Avoid explicit loop.close() here, let thread exit clean up?
        st.session_state.manager_task_running = False
        logger.info("Asyncio thread: Exiting.")


def start_asyncio_thread():
    """Starts the asyncio thread if not already running for this session."""
    # Use a session state flag to track if the thread is supposed to be running
    if not st.session_state.get("asyncio_thread_started", False):
        logger.info("Request received to start asyncio background thread...")
        manager = ensure_manager() # Get or create manager
        if not manager:
             logger.error("Cannot start thread, manager instance is missing.")
             return

        st.session_state.event_loop = asyncio.new_event_loop() # Create new loop for the thread
        st.session_state.manager_thread = threading.Thread(
            target=run_manager_in_thread,
            args=(st.session_state.event_loop, manager),
            daemon=True # Daemon threads exit when main program exits
        )
        st.session_state.manager_thread.start()
        st.session_state.asyncio_thread_started = True # Set flag
        logger.info("Asyncio thread started via threading.Thread.")
        time.sleep(1) # Give thread time to start loop and manager session
        if not st.session_state.get('manager_task_running', False):
             logger.warning("Asyncio thread started but manager task may not be running yet.")
    else:
        logger.debug("Asyncio thread start requested, but flag indicates it's already started.")


def stop_asyncio_thread():
    """Signals the asyncio thread and associated tasks to stop."""
    if st.session_state.get("asyncio_thread_started", False):
        logger.info("Request received to stop asyncio background thread...")
        loop = st.session_state.get('event_loop')
        manager = st.session_state.get('live_connect_manager')
        manager_thread = st.session_state.get('manager_thread')

        # Signal the main loop runner (stop_future) to exit first
        stop_future = st.session_state.get('stop_future')
        if loop and loop.is_running() and stop_future and not stop_future.done():
             logger.info("Signaling stop_future...")
             loop.call_soon_threadsafe(stop_future.set_result, True)
        else:
             logger.warning("Could not signal stop_future (loop not running or future missing/done).")


        # Then, stop the manager's internal session and tasks
        if loop and loop.is_running() and manager and manager._is_running:
            logger.info("Calling manager.stop_session()...")
            future = asyncio.run_coroutine_threadsafe(manager.stop_session(), loop)
            try:
                future.result(timeout=5) # Wait briefly for graceful shutdown
                logger.info("Manager stop_session() completed or timed out.")
            except Exception as e:
                logger.error(f"Error or timeout waiting for manager.stop_session: {e}")
        else:
             logger.warning("Could not call manager.stop_session (loop/manager not ready or not running).")

        # Finally, wait for the thread to finish
        if manager_thread and manager_thread.is_alive():
            logger.info("Waiting for asyncio thread to join...")
            manager_thread.join(timeout=7.0)
            if manager_thread.is_alive():
                logger.warning("Asyncio thread did not stop gracefully after join.")
            else:
                 logger.info("Asyncio thread joined successfully.")
        else:
             logger.info("Asyncio thread was not alive for joining.")

        # Clean up state variables
        st.session_state.pop('manager_thread', None)
        st.session_state.pop('event_loop', None)
        st.session_state.pop('stop_future', None)
        st.session_state.pop('manager_task_running', None)
        # Keep manager instance? Let's keep it for potential reuse without reinit.
        # st.session_state.pop('live_connect_manager', None)
        st.session_state.asyncio_thread_started = False # Reset flag
        logger.info("Asyncio thread state cleared.")
    else:
         logger.debug("Stop asyncio thread requested, but flag indicates it wasn't started.")

# --- Streamlit Audio Processor ---
class LiveConnectAudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self._buffer = bytearray()
        # Calculate target chunk size based on SEND_SAMPLE_RATE (e.g., 16kHz)
        self._target_chunk_size = int(SEND_SAMPLE_RATE * CHANNELS * 2 * 0.020) # e.g., 20ms chunk, 16bit=2 bytes
        self.loop = st.session_state.get('event_loop')
        self.manager = st.session_state.get('live_connect_manager')
        logger.info(f"AudioProcessor initialized at {time.time()}. Loop is set: {self.loop is not None}. Manager is set: {self.manager is not None}")
        if self.loop:
             logger.info(f"AudioProcessor.__init__: Loop running: {self.loop.is_running()}")
        if self.manager:
             logger.info(f"AudioProcessor.__init__: Manager running: {self.manager._is_running}")

    async def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Processes audio frames from WebRTC, sends to LiveConnect, returns audio for playback."""
        current_time = time.time()
        # Log basic frame info
        # logger.debug(f"AudioProcessor.recv at {current_time}: Received frame pts={frame.pts}, samples={frame.samples}, rate={frame.sample_rate}, format={frame.format.name}, layout={frame.layout.name}")

        # Check if loop and manager are ready before processing
        if not self.loop or not self.manager or not self.manager._is_running or not self.loop.is_running():
             if current_time % 5 < 0.1: # Log warning periodically, not constantly
                 logger.warning(f"Asyncio loop/manager not ready in recv (Loop: {self.loop}, Manager: {self.manager}, M_Running: {self.manager._is_running if self.manager else 'N/A'}, L_Running: {self.loop.is_running() if self.loop else 'N/A'}). Returning silence.")
             # Return silence immediately if not ready
             silence_chunk_size = int(RECEIVE_SAMPLE_RATE * CHANNELS * 2 * 0.020) # Match chunk size duration
             silence_samples = silence_chunk_size // (CHANNELS * 2)
             silent_ndarray = np.zeros((silence_samples, CHANNELS), dtype=np.int16)
             error_frame = av.AudioFrame.from_ndarray(silent_ndarray, format='s16', layout='mono' if CHANNELS==1 else 'stereo')
             error_frame.sample_rate = RECEIVE_SAMPLE_RATE
             return error_frame

        # 1. Process incoming audio frame
        try:
            # Convert frame data to bytes (EXPECTS s16 mono based on previous assumptions)
            # Add robust format/layout/rate checking and conversion here!
            if frame.format.name != 's16':
                 logger.error(f"Unsupported incoming audio format: {frame.format.name}")
                 # Handle format conversion or skip frame
                 in_data = None # Placeholder
            elif frame.layout.name != ('mono' if CHANNELS == 1 else 'stereo'):
                 logger.error(f"Unsupported incoming audio layout: {frame.layout.name}")
                 # Handle channel conversion or skip frame
                 in_data = None # Placeholder
            elif frame.sample_rate != SEND_SAMPLE_RATE:
                 # !!! IMPORTANT: RESAMPLING NEEDED !!!
                 # logger.warning(f"Resampling needed: Input rate {frame.sample_rate} != Target {SEND_SAMPLE_RATE}")
                 # This requires libraries like numpy/scipy or librosa
                 # Placeholder: Just log for now, will likely cause API issues
                 in_data = frame.to_ndarray().tobytes() # Process anyway for now
            else:
                 # Format, layout, rate match assumptions
                 in_data = frame.to_ndarray().tobytes()

            if in_data:
                self._buffer.extend(in_data)
                # Send chunks of the target size
                while len(self._buffer) >= self._target_chunk_size:
                    chunk_to_send = self._buffer[:self._target_chunk_size]
                    del self._buffer[:self._target_chunk_size]
                    # Safely call the async method from this sync context
                    asyncio.run_coroutine_threadsafe(
                         self.manager.send_audio_chunk(chunk_to_send), self.loop
                    ).result(timeout=0.1) # Add short timeout to prevent potential deadlock if queue is full

        except Exception as e:
            logger.error(f"Error processing incoming audio frame: {e}", exc_info=True)

        # 2. Get audio chunk received from LiveConnect for playback
        out_data = None
        try:
            future = asyncio.run_coroutine_threadsafe(self.manager.get_received_audio_chunk(), self.loop)
            out_data = future.result(timeout=0.005) # Very short timeout, data should be waiting or None
        except asyncio.TimeoutError:
             pass # No data ready is normal
        except Exception as e:
             logger.error(f"Error getting received audio chunk: {e}")

        # Use silence if no data
        silence_chunk_size = int(RECEIVE_SAMPLE_RATE * CHANNELS * 2 * 0.020) # Match chunk size duration
        if not out_data:
            out_data = b'\0' * silence_chunk_size

        # 3. Construct the output AudioFrame
        try:
            # Ensure data length is suitable for s16 format
            bytes_per_sample = 2 * CHANNELS
            if len(out_data) % bytes_per_sample != 0:
                 # Truncate to the nearest valid frame size if length is wrong
                 logger.warning(f"Output data length {len(out_data)} not multiple of bytes per sample {bytes_per_sample}. Truncating.")
                 valid_len = (len(out_data) // bytes_per_sample) * bytes_per_sample
                 out_data = out_data[:valid_len]

            # Handle empty buffer case after potential truncation
            if not out_data:
                out_data = b'\0' * silence_chunk_size

            new_ndarray = np.frombuffer(out_data, dtype=np.int16)
            # Reshape for layout (samples, channels)
            expected_samples = len(out_data) // bytes_per_sample
            new_ndarray = new_ndarray.reshape(expected_samples, CHANNELS)

            new_frame = av.AudioFrame.from_ndarray(new_ndarray, format='s16', layout='mono' if CHANNELS==1 else 'stereo')
            new_frame.sample_rate = RECEIVE_SAMPLE_RATE
            new_frame.pts = frame.pts # Attempt to preserve PTS

            return new_frame

        except Exception as e:
            logger.error(f"Error constructing output audio frame: {e}", exc_info=True)
            # Return silence on error
            silence_samples = silence_chunk_size // bytes_per_sample
            silent_ndarray = np.zeros((silence_samples, CHANNELS), dtype=np.int16)
            error_frame = av.AudioFrame.from_ndarray(silent_ndarray, format='s16', layout='mono' if CHANNELS==1 else 'stereo')
            error_frame.sample_rate = RECEIVE_SAMPLE_RATE
            return error_frame

    def on_ended(self):
        """Callback when the WebRTC connection ends."""
        logger.info("AudioProcessor.on_ended called.")
        # Stop is handled by the main UI checking webrtc_ctx.state.playing now
        # stop_asyncio_thread() # Avoid calling stop directly from here


# --- Streamlit UI ---
st.set_page_config(layout="wide") # Optional: Use wider layout

st.title("Live Voice Agent Interaction")
st.markdown("Connect your microphone and speak to the agent.")

col1, col2 = st.columns([0.7, 0.3])

with col1:
    status_placeholder = st.empty()
    transcript_placeholder = st.empty()

with col2:
    # Debug button
    if st.button("Clear Session State (Debug)"):
        logger.info("Clearing Streamlit session state.")
        # Stop thread first if running
        stop_asyncio_thread()
        # Clear state
        for key in list(st.session_state.keys()):
            if key != 'query_params': # Preserve query params if any
                 del st.session_state[key]
        st.rerun()

    st.write("---")
    st.write("**Instructions:**")
    st.write("1. Allow microphone access when prompted.")
    st.write("2. Use the 'Start/Stop' button below the video window.")
    st.write("3. Speak clearly.")
    st.write("4. Stop the connection when finished.")


# Get or create manager instance at the start of the script run
manager = ensure_manager()
if not manager:
     st.error("Failed to ensure LiveConnectManager is initialized. Cannot proceed.")
     st.stop()


webrtc_ctx = webrtc_streamer(
    key="live-voice-interaction", # Unique key for the component instance
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
    audio_processor_factory=LiveConnectAudioProcessor, # Factory to create processor instance
    async_processing=True, # Recommended for non-blocking processing
    # Desired latency can be adjusted, affects buffer sizes maybe
    # desired_latency=0.2, # seconds
    # Sendonly=False, Recvonly=False by default for SENDRECV
)

# --- Manage background thread based on WebRTC component state ---
if webrtc_ctx.state.playing:
    status_placeholder.write("Status: ✅ Connected and Listening...")
    start_asyncio_thread() # Start thread if not running and component is playing

    # Display transcriptions dynamically
    full_transcript = st.session_state.get("full_transcript", "")
    while True: # Loop as long as we are playing to update transcript
         if not webrtc_ctx.state.playing:
             logger.info("WebRTC component stopped playing, breaking transcript loop.")
             break # Exit loop if component stops

         new_text = None
         if st.session_state.get('asyncio_thread_started', False):
             loop = st.session_state.get('event_loop')
             manager = st.session_state.get('live_connect_manager')
             if loop and loop.is_running() and manager:
                  try:
                      future = asyncio.run_coroutine_threadsafe(manager.get_transcription(), loop)
                      new_text = future.result(timeout=0.01)
                  except (asyncio.TimeoutError, queue.Empty): # Handle potential queue empty if used internally
                      pass # No new text is okay
                  except Exception as e:
                      logger.error(f"Error getting transcription in UI loop: {e}")
             else:
                  # Log periodically if loop isn't running while playing
                  if time.time() % 10 < 0.1:
                       logger.warning("Transcript check: Component playing but loop/manager not running.")

         if new_text:
             full_transcript += new_text # Append only if not None/empty
             st.session_state.full_transcript = full_transcript # Update session state

         # Update placeholder - use markdown for potential auto-scroll/height handling?
         transcript_placeholder.text_area("Transcript:", full_transcript, height=300, key="transcript_display")

         time.sleep(0.1) # Short sleep to prevent tight loop, adjust interval as needed

else:
    # Component is not playing (Initial state, stopped, or errored)
    status_placeholder.write("Status: ⏸️ Disconnected / Stopped. (Allow mic & click 'Start' below)")
    stop_asyncio_thread() # Ensure thread is stopped if component is not playing
    st.session_state.full_transcript = "" # Clear transcript when stopped
    transcript_placeholder.text_area("Transcript:", "", height=300, key="transcript_display_cleared")


logger.debug(f"Streamlit script execution finished. Playing: {webrtc_ctx.state.playing}")