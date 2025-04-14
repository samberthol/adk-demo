# ui/api.py
import asyncio
import os
import sys
import logging
import time
import uuid
import json
from pathlib import Path
import io # For handling byte streams

# --- FastAPI / Starlette Imports ---
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState
from fastapi.responses import HTMLResponse

# --- Google AI / ADK Imports ---
from google import genai
from google.genai import types
# Removed Live API specific imports as they won't be used without v1alpha client
# from google.genai.types import LiveConnectConfig, LiveClientMessage, ActivityStart, ActivityEnd, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig
from google.genai.types import Content, Part # Keep Content/Part for ADK interaction

# Import HttpOptions needed for the check, even if not used in client init anymore
try:
    from google.genai.types import HttpOptions
except ImportError as e:
     # Keep logger definition consistent
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
     logger = logging.getLogger("fastapi_app_init_error")
     logger.error(f"Critical Import Error for google.genai.types.HttpOptions: {e}. Check library version.")
     HttpOptions = None # Set to None to handle gracefully if import fails

# --- ADK Agent Imports (Keep these) ---
try:
    from agents.meta.agent import meta_agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
except ImportError as e:
     # Ensure logger is configured before use here too
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
     logger = logging.getLogger("fastapi_app_init_error")
     logging.error(f"FastAPI: Failed to import agent modules/ADK components: {e}")
     sys.exit(f"FastAPI startup failed: {e}")

# --- Transcoding Import (Keep if potentially needed for other audio tasks, though not used here now) ---
try:
    import ffmpeg
except ImportError:
    # Ensure logger is configured
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger("fastapi_app_init_error")
    logging.warning("ffmpeg-python not installed. Audio input cannot be transcoded.") # Changed to warning
    ffmpeg = None # Set to None to handle gracefully later

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fastapi_app")

APP_NAME = "gcp_multi_agent_demo_api"
USER_ID_PREFIX = "fastapi_user_"
ADK_SESSION_PREFIX = f'adk_session_{APP_NAME}_'
# GEMINI_LIVE_MODEL_NAME = "models/gemini-2.0-flash-live-001" # Keep commented out or remove, as Live API isn't used
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
# STREAMING_INTERIM_RESULTS = True # No longer relevant for this endpoint

# Audio constants - Keep commented out or remove if ffmpeg is definitely not used elsewhere
# TARGET_SAMPLE_RATE = 16000
# TARGET_CHANNELS = 1
# TARGET_FORMAT = 's16le'

# --- ADK Initialization ---
session_service = InMemorySessionService()
adk_runner = Runner(agent=meta_agent, app_name=APP_NAME, session_service=session_service)
active_adk_sessions = {}

# --- Google Generative AI Configuration (Client initialized without v1alpha) ---
client = None # Initialize client variable globally
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY environment variable not set.")
# Removed HttpOptions check as it's not used for client init here
# elif HttpOptions is None:
#     logger.error("Cannot configure client: HttpOptions failed to import. Check google-genai version.")
else:
    try:
        # Initialize the client directly, passing only the API key
        # This uses the default API version, NOT v1alpha
        client = genai.Client(
            api_key=GOOGLE_API_KEY
        )
        logger.info("Google Generative AI client configured using default API version.")
    except Exception as e:
        logger.error(f"Failed to configure Google Generative AI client: {e}")
        # client remains None if initialization fails

# --- ADK Interaction Functions (Sync for simplicity) ---
# (get_or_create_adk_session_sync remains the same)
def get_or_create_adk_session_sync(user_id: str) -> str:
    if user_id in active_adk_sessions:
        session_id = active_adk_sessions[user_id]
        try: # Check/Recreate logic
            existing = session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
            if not existing:
                logger.warning(f"ADK session {session_id} not found. Recreating.")
                session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id, state={})
        except Exception as e: logger.error(f"Error checking/recreating session {session_id}: {e}"); raise
        return session_id
    else: # Create new
        session_id = f"{ADK_SESSION_PREFIX}{int(time.time())}_{os.urandom(4).hex()}"
        active_adk_sessions[user_id] = session_id
        try: session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id, state={})
        except Exception as e:
           logger.exception(f"--- FastAPI ADK Sync: FATAL ERROR creating session {session_id} for {user_id}:")
           if user_id in active_adk_sessions:
               del active_adk_sessions[user_id]
           raise # Re-raise the exception
        return session_id

# (run_adk_turn_sync remains the same - it interacts with ADK, not directly with Gemini Live here)
def run_adk_turn_sync(user_id: str, session_id: str, user_message_text: str) -> str:
    logger.info(f"ADK Run: Session {session_id}, Query: '{user_message_text[:100]}...'")
    content = Content(role='user', parts=[Part(text=user_message_text)])
    final_response_text = "[Agent did not respond]"
    try:
        for event in adk_runner.run(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts and hasattr(event.content.parts[0], 'text'):
                    final_response_text = event.content.parts[0].text
                else: final_response_text = "[Agent finished with no text output]"
                break
    except Exception as e: logger.exception(f"ADK Run EXCEPTION for session {session_id}:"); final_response_text = f"Error: {e}"
    return final_response_text

# --- FastAPI App ---
app = FastAPI()

# --- Frontend HTML/JS - Modified to indicate disabled functionality ---
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>FastAPI Gemini Live Audio (DISABLED)</title> <style> body { font-family: sans-serif; } </style>
    </head>
    <body>
        <h1>Voice Interaction (Gemini Live & ADK Agent) - DISABLED</h1>
        <p><i>Note: Real-time voice interaction is currently disabled due to configuration changes.</i></p>
        <p>Status: <span id="status">Initializing</span></p>
        <button id="start" disabled>Start Recording</button> <button id="stop" disabled>Stop Recording</button>
        <h2>Agent Interaction Log:</h2>
        <div id="interaction" style="height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; white-space: pre-wrap;"></div>

        <script>
            // Keep JS functions for basic WS connection and logging, but remove audio parts

            const statusSpan = document.getElementById('status');
            const interactionDiv = document.getElementById('interaction');
            const startButton = document.getElementById('start'); // Keep reference, but it stays disabled
            const stopButton = document.getElementById('stop');

            let websocket;
            // Remove MediaRecorder, audio chunks, AudioContext related variables
            // let mediaRecorder;
            // let audioChunks = [];
            // let audioContext;
            // let audioQueue = [];
            // let isPlaying = false;
            // let nextStartTime = 0;
            // const playbackSampleRate = 24000;

            function logInteraction(message, type = 'info') {
                const p = document.createElement('p');
                let prefix = '';
                if (type === 'user') prefix = '<strong>You:</strong> ';
                else if (type === 'agent') prefix = '<strong>Agent:</strong> ';
                else if (type === 'system') prefix = '<em>System:</em> ';
                else if (type === 'interim') prefix = '<em>You (interim):</em> '; // Keep for potential future use
                p.innerHTML = prefix + message;
                interactionDiv.appendChild(p);
                interactionDiv.scrollTop = interactionDiv.scrollHeight;
                console.log(`${type}: ${message}`);
            }

            // Remove function sendControlMessage - no longer needed
            // Remove audio playback functions: initAudioContext, playAudioChunk, schedulePlayback

            function connectWebSocket() {
                const wsUri = `wss://${location.host}/ws/audio_gemini`; // Keep endpoint name for now
                logInteraction(`Attempting WebSocket connection to: ${wsUri}`, 'system');
                statusSpan.textContent = "Connecting...";
                // startButton remains disabled

                try {
                    websocket = new WebSocket(wsUri);
                    // websocket.binaryType = 'arraybuffer'; // No longer expecting binary audio

                    websocket.onopen = function(evt) {
                        statusSpan.textContent = "Connected (Voice Disabled)";
                        logInteraction("WebSocket Connected. Voice interaction disabled.", 'system');
                        // startButton remains disabled
                    };
                    websocket.onclose = function(evt) {
                        statusSpan.textContent = "Disconnected";
                        logInteraction(`WebSocket Disconnected: Code=${evt.code}, Reason=${evt.reason || 'N/A'}, WasClean=${evt.wasClean}`, 'system');
                        // startButton.disabled = true; stopButton.disabled = true; // Already disabled
                    };
                    websocket.onerror = function(evt) {
                        statusSpan.textContent = "Error";
                        logInteraction('WebSocket Error occurred. See browser console.', 'system');
                        console.error('WebSocket Error:', evt);
                        // startButton.disabled = true; stopButton.disabled = true; // Already disabled
                    };
                    websocket.onmessage = function(evt) {
                        // Remove ArrayBuffer handling
                        if (typeof evt.data === 'string') {
                            try {
                                const msg = JSON.parse(evt.data);
                                // Simplify message handling - only expect info/error/status from server now
                                if (msg.type === 'agent_response') logInteraction(msg.response, 'agent');
                                else if (msg.type === 'status' || msg.type === 'info') logInteraction(msg.message, 'system');
                                else if (msg.type === 'error') logInteraction(`Error: ${msg.message}`, 'system');
                                else logInteraction(`Unknown JSON message type: ${JSON.stringify(msg)}`, 'system');
                            } catch (e) {
                                logInteraction(`Received non-JSON text message: ${evt.data}`, 'system');
                                console.error("Failed to parse text message:", e);
                            }
                        } else {
                            console.warn("Received unexpected message type:", typeof evt.data);
                        }
                    };
                } catch (err) {
                       statusSpan.textContent = "Error";
                       logInteraction(`Error creating WebSocket: ${err}`, 'system');
                       console.error("Error creating WebSocket:", err);
                       // startButton.disabled = true; stopButton.disabled = true; // Already disabled
                }
            }

            // Remove startButton.onclick and stopButton.onclick handlers

            // Initial connection attempt when page loads
            connectWebSocket();
        </script>
    </body>
</html>
"""

@app.get("/")
async def get_test_page():
    """Serves the simple HTML page with WebSocket audio handling."""
    return HTMLResponse(html)

# --- Helper function for transcoding ---
# Keep commented out or remove entirely if not used elsewhere
# async def transcode_audio_ffmpeg(input_bytes: bytes) -> bytes | None:
#     """Transcodes audio bytes using ffmpeg-python."""
#     if not ffmpeg: logger.error("ffmpeg-python library not available."); return None
#     try:
#         process = (
#             ffmpeg
#             .input('pipe:0')
#             .output('pipe:1', format=TARGET_FORMAT, acodec='pcm_s16le', ac=TARGET_CHANNELS, ar=TARGET_SAMPLE_RATE)
#             .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True)
#         )
#         stdout, stderr = await asyncio.to_thread(process.communicate, input=input_bytes)
#         if process.returncode != 0:
#             logger.error(f"FFmpeg failed: {process.returncode}\n{stderr.decode()}")
#             return None
#         return stdout
#     except ffmpeg.Error as e:
#         logger.error(f"ffmpeg-python error: {e}\n{getattr(e, 'stderr', b'').decode()}", exc_info=True)
#         return None
#     except Exception as e:
#         logger.error(f"Unexpected transcoding error: {e}", exc_info=True)
#         return None

# --- WebSocket Endpoint (Simplified - No Live API interaction) ---
@app.websocket("/ws/audio_gemini")
async def websocket_endpoint_gemini(websocket: WebSocket):
    """Handles WebSocket connections. Gemini Live audio interaction is DISABLED."""
    await websocket.accept()
    client_id = f"{USER_ID_PREFIX}{uuid.uuid4()}"
    logger.info(f"WebSocket connection accepted: {client_id} (Live API Disabled)")
    adk_session_id = None
    # Remove Live API specific variables
    # live_session = None
    # send_task = None
    # receive_task = None
    # audio_queue = asyncio.Queue()

    try:
        # Check if the default client was initialized successfully
        if client is None:
            logger.error(f"[{client_id}] Cannot start session: Google AI Client not initialized.")
            await websocket.send_text(json.dumps({"type": "error", "message": "Server error: Google AI Client not initialized."}))
            await websocket.close(code=1011) # Internal error
            return

        if not GOOGLE_API_KEY: raise ValueError("Server not configured with GOOGLE_API_KEY.")

        adk_session_id = get_or_create_adk_session_sync(client_id)
        await websocket.send_text(json.dumps({"type": "info", "message": f"ADK Session Ready: {adk_session_id}"}))
        await websocket.send_text(json.dumps({"type": "info", "message": "Gemini Live API interaction is disabled in this configuration."}))

        # --- Remove Gemini Live API Interaction Block ---
        # async with client.aio.live.connect(...) as live_session:
             # ... (All code inside this block is removed) ...
        # --- End Removal ---

        # Simplified loop: Just keep connection open, potentially receive text messages for ADK?
        # This part needs defining based on what you *want* the WebSocket to do now.
        # For now, it just waits for disconnect.
        logger.info(f"[{client_id}] Waiting for messages or disconnect (Live API inactive).")
        while True: # Loop until disconnect
            data = await websocket.receive()

            if data['type'] == 'websocket.disconnect':
                logger.info(f"[{client_id}] WebSocket disconnect message received.")
                break # Exit loop on disconnect

            if data['type'] == 'text':
                logger.info(f"[{client_id}] Received text message: {data['text']}")
                # Example: You could potentially pass this text to the ADK agent here
                # try:
                #     msg_data = json.loads(data['text'])
                #     if msg_data.get("type") == "user_text_input":
                #         user_text = msg_data.get("text", "")
                #         if user_text and adk_session_id:
                #             await websocket.send_text(json.dumps({"type": "info", "message": "Sending text to agent..."}))
                #             loop = asyncio.get_running_loop()
                #             agent_response = await loop.run_in_executor(None, run_adk_turn_sync, client_id, adk_session_id, user_text)
                #             await websocket.send_text(json.dumps({"type": "agent_response", "response": agent_response}))
                # except json.JSONDecodeError:
                #      logger.warning(f"[{client_id}] Received non-JSON text: {data['text']}")
                # except Exception as e:
                #      logger.error(f"[{client_id}] Error processing text msg: {e}")
                await websocket.send_text(json.dumps({"type": "info", "message": "Text received, but endpoint is not configured for text interaction via WS."}))


            elif data['type'] == 'bytes':
                 logger.warning(f"[{client_id}] Received bytes, but audio processing is disabled.")
                 # Ignore bytes

        logger.info(f"[{client_id}] Exited main WS receive loop.")

    # --- Exception Handling & Cleanup (Simplified) ---
    except WebSocketDisconnect: logger.info(f"WS client {client_id} disconnected.")
    except ValueError as e: # Catch value errors during setup (e.g., missing key)
        logger.error(f"[{client_id}] Startup failed due to ValueError: {e}")
        try: await websocket.send_text(json.dumps({"type": "error", "message": f"Server Config Error: {e}"}))
        except Exception: pass
    except Exception as e: # Catch any other unexpected errors
        logger.error(f"Unexpected error in WS handler {client_id}: {e}", exc_info=True)
        try:
            if websocket.client_state != WebSocketState.DISCONNECTED:
                 await websocket.send_text(json.dumps({"type": "error", "message": f"Server error: {str(e)}"}))
        except Exception: pass # Ignore errors during error reporting
    finally:
        logger.info(f"Closing WS connection & cleaning up for {client_id}.")
        # Remove cancellation of tasks that no longer exist
        # if send_task and not send_task.done(): send_task.cancel()
        # if receive_task and not receive_task.done(): receive_task.cancel()
        # Remove audio queue signal
        # try: await audio_queue.put(None)
        # except Exception: pass
        # Remove waiting for tasks
        # try:
        #     tasks = [t for t in [send_task, receive_task] if t]
        #     if tasks: await asyncio.gather(*tasks, return_exceptions=True)
        # except Exception as gather_err: logger.error(f"[{client_id}] Error during task cleanup: {gather_err}")

        # Clean up ADK session link
        if client_id in active_adk_sessions: del active_adk_sessions[client_id]; logger.info(f"Removed ADK session link for {client_id}.")
        # Ensure WebSocket is closed
        try:
            ws_state = getattr(websocket, 'client_state', None)
            if ws_state and ws_state != WebSocketState.DISCONNECTED:
                 await websocket.close()
                 logger.info(f"Closed WebSocket for {client_id} from server side.")
        except Exception as e: logger.error(f"Error closing WS for {client_id}: {e}")

# --- Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server directly with Uvicorn...")
    if client is None: # Check if default client initialization failed
        logger.error("FATAL: Google AI Client could not be initialized. API calls will fail.")
    # Keep ffmpeg check if it might be used elsewhere
    if ffmpeg is None:
        logger.warning("WARNING: ffmpeg-python not found, audio transcoding will fail if attempted.")
    uvicorn.run("ui.api:app", host="0.0.0.0", port=8000, reload=True)