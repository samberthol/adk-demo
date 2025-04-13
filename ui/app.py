import streamlit as st
# Make sure to import the correct classes now
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import asyncio
import threading
import queue
import av
import logging
import time

# Assuming live_connect_manager.py is correct and doesn't import pyaudio
from live_connect_manager import LiveConnectManager, SEND_SAMPLE_RATE, RECEIVE_SAMPLE_RATE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define configuration directly (no ClientSettings)
RTC_CONFIGURATION = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
MEDIA_STREAM_CONSTRAINTS = {"audio": True, "video": False}

# --- Refactored Asyncio Event Loop Management using Session State ---

def ensure_manager():
    """Gets or creates the LiveConnectManager instance in session state."""
    if 'live_connect_manager' not in st.session_state:
        logger.info("Creating LiveConnectManager instance.")
        st.session_state.live_connect_manager = LiveConnectManager()
    return st.session_state.live_connect_manager

def run_manager_in_thread(loop, manager):
    """Target function for the background thread."""
    asyncio.set_event_loop(loop)
    st.session_state.manager_task_running = True
    try:
        logger.info("Asyncio thread: Starting manager session...")
        loop.run_until_complete(manager.start_session())
        logger.info("Asyncio thread: Manager session started. Running related tasks...")
        # Keep the loop running if start_session only starts background tasks
        # Or run a main task like the queue processing if start_session doesn't block
        # Assuming start_session starts background loops and returns.
        # We need a way to keep this loop alive until stopped.
        # Let's create a future that only completes on stop signal.
        st.session_state.stop_future = loop.create_future()
        loop.run_until_complete(st.session_state.stop_future)

    except Exception as e:
        logger.error(f"Exception in asyncio thread: {e}", exc_info=True)
    finally:
        logger.info("Asyncio thread: Loop finishing.")
        # Avoid closing loop here explicitly, let thread exit clean up?
        # loop.close()
        st.session_state.manager_task_running = False
        logger.info("Asyncio thread: Exiting.")


def start_asyncio_thread():
    """Starts the asyncio thread if not already running."""
    if 'manager_thread' not in st.session_state or not st.session_state.manager_thread.is_alive():
        logger.info("Starting asyncio background thread...")
        manager = ensure_manager() # Get or create manager
        st.session_state.event_loop = asyncio.new_event_loop() # Create new loop for the thread
        st.session_state.manager_thread = threading.Thread(
            target=run_manager_in_thread,
            args=(st.session_state.event_loop, manager),
            daemon=True
        )
        st.session_state.manager_thread.start()
        time.sleep(1) # Give thread time to start loop and manager
    else:
        logger.debug("Asyncio thread already running.")


def stop_asyncio_thread():
    """Signals the asyncio thread and tasks to stop."""
    if 'manager_thread' in st.session_state and st.session_state.manager_thread.is_alive():
        logger.info("Requesting asyncio thread stop...")
        loop = st.session_state.get('event_loop')
        manager = st.session_state.get('live_connect_manager')

        if loop and loop.is_running() and manager:
            # Stop the manager's session first
            if manager._is_running:
                logger.info("Calling manager.stop_session()...")
                future = asyncio.run_coroutine_threadsafe(manager.stop_session(), loop)
                try:
                    future.result(timeout=5) # Wait for graceful shutdown
                    logger.info("Manager stop_session() completed.")
                except Exception as e:
                    logger.error(f"Error or timeout waiting for manager.stop_session: {e}")

            # Signal the main loop runner to exit (if using the future pattern)
            stop_future = st.session_state.get('stop_future')
            if stop_future and not stop_future.done():
                 loop.call_soon_threadsafe(stop_future.set_result, True)

        # Wait for the thread to actually finish
        st.session_state.manager_thread.join(timeout=7)
        if st.session_state.manager_thread.is_alive():
            logger.warning("Asyncio thread did not stop gracefully.")
        else:
            logger.info("Asyncio thread joined.")

    # Clean up state
    st.session_state.pop('manager_thread', None)
    st.session_state.pop('event_loop', None)
    st.session_state.pop('stop_future', None)
    st.session_state.pop('manager_task_running', None)
    logger.info("Asyncio thread state cleared.")


# --- Streamlit Audio Processor ---
class LiveConnectAudioProcessor(AudioProcessorBase):
    def __init__(self) -> None:
        super().__init__()
        self._buffer = bytearray()
        self._target_chunk_size = int(SEND_SAMPLE_RATE * 1 * 2 * 0.010) # ~320 bytes
        self.loop = st.session_state.get('event_loop') # Get loop existing at init time
        self.manager = st.session_state.get('live_connect_manager') # Get manager
        logger.info(f"AudioProcessor initialized. Loop: {self.loop}, Manager: {self.manager}")

    async def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Processes audio frames from WebRTC, sends to LiveConnect, returns audio for playback."""
        if not self.loop or not self.manager or not self.manager._is_running:
             logger.warning("Asyncio loop/manager not ready in recv.")
             # Return silence if not ready
             silence_chunk_size = int(RECEIVE_SAMPLE_RATE * 1 * 2 * 0.010)
             silence_samples = silence_chunk_size // 2
             silent_ndarray = np.zeros((silence_samples, 1), dtype=np.int16)
             error_frame = av.AudioFrame.from_ndarray(silent_ndarray, format='s16', layout='mono')
             error_frame.sample_rate = RECEIVE_SAMPLE_RATE
             return error_frame

        # 1. Process incoming audio frame
        try:
            # TODO: Resampling if needed based on frame.sample_rate vs SEND_SAMPLE_RATE
            in_data = frame.to_ndarray(format='s16').tobytes()
            self._buffer.extend(in_data)

            while len(self._buffer) >= self._target_chunk_size:
                chunk_to_send = self._buffer[:self._target_chunk_size]
                del self._buffer[:self._target_chunk_size]
                # Safely call the async method from this sync context
                asyncio.run_coroutine_threadsafe(
                     self.manager.send_audio_chunk(chunk_to_send), self.loop
                )
                # Don't wait for the future here to keep recv responsive

        except Exception as e:
            logger.error(f"Error processing incoming audio frame: {e}", exc_info=True)

        # 2. Get audio chunk received from LiveConnect for playback
        out_data = None
        try:
            future = asyncio.run_coroutine_threadsafe(self.manager.get_received_audio_chunk(), self.loop)
            out_data = future.result(timeout=0.01) # Very short timeout, expect immediate data or None
        except (asyncio.TimeoutError, queue.Empty):
             pass # No data ready is normal
        except Exception as e:
             logger.error(f"Error getting received audio chunk: {e}")

        # Use silence if no data
        if not out_data:
            silence_chunk_size = int(RECEIVE_SAMPLE_RATE * 1 * 2 * 0.010)
            out_data = b'\0' * silence_chunk_size

        # 3. Construct the output AudioFrame
        try:
            num_samples = len(out_data) // 2
            new_ndarray = np.frombuffer(out_data, dtype=np.int16)
            if len(out_data) % 2 != 0: # Should not happen with s16
                 new_ndarray = np.frombuffer(out_data[:-1], dtype=np.int16) # Drop last byte if odd

            # Ensure correct shape (samples, channels=1)
            new_ndarray = new_ndarray.reshape(-1, 1)

            new_frame = av.AudioFrame.from_ndarray(new_ndarray, format='s16', layout='mono')
            new_frame.sample_rate = RECEIVE_SAMPLE_RATE
            new_frame.pts = frame.pts
            return new_frame

        except Exception as e:
            logger.error(f"Error constructing output audio frame: {e}", exc_info=True)
            # Return silence on error
            silence_chunk_size = int(RECEIVE_SAMPLE_RATE * 1 * 2 * 0.010)
            silence_samples = silence_chunk_size // 2
            silent_ndarray = np.zeros((silence_samples, 1), dtype=np.int16)
            error_frame = av.AudioFrame.from_ndarray(silent_ndarray, format='s16', layout='mono')
            error_frame.sample_rate = RECEIVE_SAMPLE_RATE
            return error_frame

    def on_ended(self):
        logger.info("AudioProcessor.on_ended called.")
        # The main UI loop should handle stopping based on webrtc_ctx.state

# --- Streamlit UI ---
st.title("Live Voice Agent Interaction")

if st.button("Clear State (Debug)"):
     # Add button to manually clear state if needed during dev
     for key in list(st.session_state.keys()):
          del st.session_state[key]
     st.rerun()


transcript_placeholder = st.empty()
status_placeholder = st.empty()

# Get or create manager instance at the start of the script run
ensure_manager()

webrtc_ctx = webrtc_streamer(
    key="live-voice-interaction", # Changed key to avoid potential conflicts
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints=MEDIA_STREAM_CONSTRAINTS,
    audio_processor_factory=LiveConnectAudioProcessor,
    async_processing=True,
)

# Manage thread based on WebRTC state
if webrtc_ctx.state.playing:
    status_placeholder.write("Status: Connecting / Listening...")
    start_asyncio_thread() # Start thread if not running

    # Display transcriptions
    full_transcript = st.session_state.get("full_transcript", "")
    while True: # Loop to update transcript display
        if 'text_received_queue' in st.session_state: # Check if queue exists
             manager = st.session_state.get('live_connect_manager')
             if manager:
                  loop = st.session_state.get('event_loop')
                  if loop and loop.is_running():
                       try:
                           future = asyncio.run_coroutine_threadsafe(manager.get_transcription(), loop)
                           text = future.result(timeout=0.01)
                           if text:
                                full_transcript += text # Append only if not None/empty
                                st.session_state.full_transcript = full_transcript # Update session state
                       except (asyncio.TimeoutError, queue.Empty):
                            pass # No new text
                       except Exception as e:
                            logger.error(f"Error getting transcription: {e}")
                  else:
                       logger.warning("Transcript check: Loop not running.")


        # Update placeholder regardless of new text to reflect current state
        transcript_placeholder.text_area("Transcript:", full_transcript, height=200, key="transcript_display")

        # Check if we should break the loop
        if not webrtc_ctx.state.playing:
            logger.info("WebRTC no longer playing, breaking transcript loop.")
            break
        time.sleep(0.2) # Update interval for transcript display

else:
    status_placeholder.write("Status: Disconnected. Click Start button above.")
    stop_asyncio_thread() # Stop thread if running
    st.session_state.full_transcript = "" # Clear transcript
    transcript_placeholder.text_area("Transcript:", "", height=200, key="transcript_display_cleared")

st.write("---")