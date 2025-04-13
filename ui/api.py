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
# Recommendation: Consider moving API keys and model names to environment variables or a config file.
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
GEMINI_LIVE_MODEL_NAME = os.environ.get("GEMINI_LIVE_MODEL", "models/gemini-2.0-flash-live-001") # Ensure this model supports Live API

try:
    # Import genai library first
    from google import genai
    # Import types needed for configuration, content, and ACTIVITY SIGNALS
    from google.genai import types
    from google.genai.types import (
        Content, Part, LiveConnectConfig, SpeechConfig, VoiceConfig,
        PrebuiltVoiceConfig, RealtimeInputConfig, AutomaticActivityDetection,
        LiveClientMessage, ActivityStart, ActivityEnd, Blob, LiveClientRealtimeInput
    )
    # Define basic config using imported types
    # MODIFY: Add realtime_input_config to disable automatic detection
    GEMINI_LIVE_CONFIG = LiveConnectConfig(
        response_modalities=["TEXT", "AUDIO"], # Request both text and audio back
        speech_config=SpeechConfig(
            voice_config=VoiceConfig(
                prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Puck") # Example voice
            )
        ),
        # Add configuration to disable automatic activity detection
        realtime_input_config=RealtimeInputConfig(
             automatic_activity_detection=AutomaticActivityDetection(disabled=True)
             # You could add activity_handling and turn_coverage here if needed
        )
        # Add other configs like session resumption if needed
    )
except ImportError as e:
    logging.error(f"Critical Import Error for google.genai or its types: {e}. Check library version.")
    # Set config to None to prevent startup if imports fail
    GEMINI_LIVE_CONFIG = None
    # Ensure genai is defined even if types fail, for later checks
    try: from google import genai
    except ImportError: genai = None

try:
    from agents.meta.agent import meta_agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
except ImportError as e:
     logging.error(f"FastAPI: Failed to import agent modules/ADK components: {e}")
     sys.exit(f"FastAPI startup failed: {e}")

# --- Transcoding Import ---
try:
    import ffmpeg
except ImportError:
    logging.error("ffmpeg-python not installed. Audio input cannot be transcoded.")
    ffmpeg = None # Set to None to handle gracefully later

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] %(message)s')
logger = logging.getLogger("fastapi_app") # Main app logger
ws_logger = logging.getLogger("websocket_handler") # Specific logger for WS handler

APP_NAME = "gcp_multi_agent_demo_api"
USER_ID_PREFIX = "fastapi_user_"
ADK_SESSION_PREFIX = f'adk_session_{APP_NAME}_'
STREAMING_INTERIM_RESULTS = True

# Audio constants matching Gemini Live API requirements
# Recommendation: Move these to a config file or environment variables.
TARGET_SAMPLE_RATE = int(os.environ.get("TARGET_SAMPLE_RATE", 16000)) # Input requires 16kHz
TARGET_CHANNELS = int(os.environ.get("TARGET_CHANNELS", 1))
TARGET_FORMAT = os.environ.get("TARGET_FORMAT", 's16le') # Raw PCM signed 16-bit little-endian

# --- ADK Initialization ---
session_service = InMemorySessionService()
adk_runner = Runner(agent=meta_agent, app_name=APP_NAME, session_service=session_service)
active_adk_sessions = {} # Simple in-memory tracking

# --- Google Generative AI Configuration ---
if not GOOGLE_API_KEY:
     logger.error("GOOGLE_API_KEY environment variable not set.")
elif genai: # Check if genai was imported successfully
     try: genai.configure(api_key=GOOGLE_API_KEY)
     except Exception as e: logger.error(f"Failed to configure Google Generative AI: {e}")
else:
    logger.error("Google Generative AI library (google.genai) could not be imported. Cannot configure.")


# --- ADK Interaction Functions (Sync for simplicity) ---
def get_or_create_adk_session_sync(user_id: str) -> str:
    """Gets existing or creates a new ADK session ID for a user."""
    if user_id in active_adk_sessions:
        session_id = active_adk_sessions[user_id]
        try: # Check if session still exists in the service (might expire or be cleared)
            existing = session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id)
            if not existing:
                logger.warning(f"ADK session {session_id} for user {user_id} not found in service. Recreating.")
                session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id, state={})
        except Exception as e:
            logger.error(f"Error checking/recreating ADK session {session_id} for {user_id}: {e}", exc_info=True)
            # Decide if we should raise or try to create a new one
            raise # Re-raise for now, indicates potential issue with session service
        return session_id
    else: # Create a new session
        # Generate a more unique session ID
        session_id = f"{ADK_SESSION_PREFIX}{uuid.uuid4()}"
        active_adk_sessions[user_id] = session_id
        try:
            logger.info(f"Creating new ADK session {session_id} for user {user_id}")
            session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id, state={})
        except Exception as e:
           logger.exception(f"--- FastAPI ADK Sync: FATAL ERROR creating session {session_id} for {user_id}:")
           if user_id in active_adk_sessions:
               del active_adk_sessions[user_id] # Clean up tracking if creation failed
           raise # Re-raise the exception
        return session_id

def run_adk_turn_sync(user_id: str, session_id: str, user_message_text: str) -> str:
    """Runs a turn with the ADK agent synchronously."""
    # Recommendation: Be mindful of potential latency here if the agent is slow.
    logger.info(f"ADK Run: User={user_id}, Session={session_id}, Query='{user_message_text[:100]}...'")
    content = Content(role='user', parts=[Part(text=user_message_text)])
    final_response_text = "[Agent did not provide a text response]" # Default response
    try:
        # Assuming adk_runner.run is synchronous and yields events
        for event in adk_runner.run(user_id=user_id, session_id=session_id, new_message=content):
            # Check if the event indicates the final response from the agent
            if event.is_final_response():
                # Check if the final response contains text content
                if event.content and event.content.parts and hasattr(event.content.parts, 'text'):
                    final_response_text = event.content.parts.text
                    logger.info(f"ADK Run: User={user_id}, Session={session_id}, Agent Response='{final_response_text[:100]}...'")
                else:
                    final_response_text = "[Agent finished turn without text output]"
                    logger.warning(f"ADK Run: User={user_id}, Session={session_id}, Agent finished with no text.")
                break # Stop processing events once final response is found
    except Exception as e:
        logger.exception(f"ADK Run EXCEPTION for User={user_id}, Session={session_id}:")
        final_response_text = f"[Error during agent interaction: {e}]"
    return final_response_text

# --- FastAPI App ---
app = FastAPI()

# --- Frontend HTML/JS with Audio Playback AND Manual Activity Signals ---
# Recommendation: Consider serving this from a separate static file for better maintainability.
# Recommendation: Enhance frontend error handling (mic permissions, websocket errors, etc.).
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>FastAPI Gemini Live Audio (Manual Activity)</title> <style> body { font-family: sans-serif; } </style>
    </head>
    <body>
        <h1>Voice Interaction (Gemini Live & ADK Agent)</h1>
        <p>Status: <span id="status">Initializing</span></p>
        <button id="start">Start Recording</button>
        <button id="stop" disabled>Stop Recording</button>
        <h2>Agent Interaction:</h2>
        <div id="interaction" style="height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; white-space: pre-wrap;"></div>

        <script>
            const statusSpan = document.getElementById('status');
            const interactionDiv = document.getElementById('interaction');
            const startButton = document.getElementById('start');
            const stopButton = document.getElementById('stop');
            let websocket;
            let mediaRecorder;
            let audioChunks =;

            // --- Audio Playback Setup (No changes needed here) ---
            let audioContext;
            let audioQueue =;
            let isPlaying = false;
            let nextStartTime = 0;
            const playbackSampleRate = 24000; // Gemini default output rate

            function initAudioContext() {
                if (!audioContext) {
                    try {
                        audioContext = new (window.AudioContext |
| window.webkitAudioContext)({ sampleRate: playbackSampleRate });
                        console.log(`AudioContext initialized with sample rate ${audioContext.sampleRate}Hz.`);
                    } catch (e) {
                        console.error("Error creating AudioContext:", e);
                        logInteraction("Error: Cannot initialize audio playback.", "system");
                    }
                }
                // Resume context if it was suspended by browser policy
                if (audioContext && audioContext.state === 'suspended') {
                    audioContext.resume().then(() => {
                        console.log("AudioContext resumed successfully.");
                    }).catch(e => console.error("Error resuming AudioContext:", e));
                }
                return audioContext;
            }

            async function playAudioChunk(arrayBuffer) {
                 if (!audioContext |
| arrayBuffer.byteLength === 0) return;
                 // Ensure context is running
                 if (audioContext.state === 'suspended') { await audioContext.resume(); }
                 if (audioContext.state!== 'running') {
                     console.warn("AudioContext not running, cannot play audio.");
                     return;
                 }

                 try {
                    // Assuming incoming data is PCM s16le (adjust if Gemini sends different format)
                    const pcm16Data = new Int16Array(arrayBuffer);
                    const float32Data = new Float32Array(pcm16Data.length);
                    for (let i = 0; i < pcm16Data.length; i++) { float32Data[i] = pcm16Data[i] / 32768.0; } // Convert s16 to float -1.0 to 1.0

                    const audioBuffer = audioContext.createBuffer(1, float32Data.length, playbackSampleRate); // 1 channel, length, sampleRate
                    audioBuffer.copyToChannel(float32Data, 0); // Copy data to the buffer channel

                    audioQueue.push(audioBuffer);
                    schedulePlayback();
                 } catch (e) {
                    console.error("Error processing or queueing audio chunk:", e);
                    logInteraction("Error playing received audio.", "system");
                 }
            }

            function schedulePlayback() {
                if (isPlaying |
| audioQueue.length === 0 ||!audioContext |
| audioContext.state!== 'running') return;
                isPlaying = true;
                const bufferToPlay = audioQueue.shift();
                const source = audioContext.createBufferSource();
                source.buffer = bufferToPlay;
                source.connect(audioContext.destination);

                const currentTime = audioContext.currentTime;
                const startTime = Math.max(currentTime, nextStartTime); // Play immediately or after the previous chunk finishes

                console.log(`Scheduling audio chunk to start at: ${startTime.toFixed(2)}s (duration: ${bufferToPlay.duration.toFixed(2)}s)`);
                source.start(startTime);

                nextStartTime = startTime + bufferToPlay.duration; // Schedule next chunk start time
                source.onended = () => {
                    console.log("Audio chunk finished playing.");
                    isPlaying = false;
                    schedulePlayback(); // Check if more chunks are waiting
                };
            }
            // --- End Audio Playback Setup ---


            function logInteraction(message, type = 'info') {
                const p = document.createElement('p');
                let prefix = '';
                if (type === 'user') prefix = '<strong>You:</strong> ';
                else if (type === 'agent') prefix = '<strong>Agent:</strong> ';
                else if (type === 'system') prefix = '<em>System:</em> ';
                else if (type === 'interim') prefix = '<em>You (interim):</em> ';
                p.innerHTML = prefix + message;
                interactionDiv.appendChild(p);
                interactionDiv.scrollTop = interactionDiv.scrollHeight; // Auto-scroll
                console.log(`${type}: ${message}`);
            }

            // Function to send control messages
            function sendControlMessage(action) {
                if (websocket && websocket.readyState === WebSocket.OPEN) {
                    try {
                        websocket.send(JSON.stringify({ "type": "control", "action": action }));
                        logInteraction(`Sent ${action} signal.`, 'system');
                    } catch (e) {
                        console.error(`Error sending ${action}:`, e);
                        logInteraction(`Error sending ${action} signal.`, 'system');
                    }
                } else {
                     logInteraction(`Cannot send ${action}: WebSocket not open.`, 'system');
                }
            }


            function connectWebSocket() {
                // Use wss:// for secure connections if served over https
                const wsProtocol = window.location.protocol === "https:"? "wss:" : "ws:";
                const wsUri = `${wsProtocol}//${location.host}/ws/audio_gemini`;
                logInteraction(`Attempting WebSocket connection to: ${wsUri}`, 'system');
                statusSpan.textContent = "Connecting...";
                startButton.disabled = true;
                stopButton.disabled = true; // Disable stop initially too

                try {
                    websocket = new WebSocket(wsUri);
                    websocket.binaryType = 'arraybuffer'; // Expect binary audio data

                    websocket.onopen = function(evt) {
                        statusSpan.textContent = "Connected";
                        logInteraction("WebSocket Connected. Ready to record.", 'system');
                        startButton.disabled = false; // Enable start only when connected
                        stopButton.disabled = true; // Stop remains disabled until recording starts
                         if (!audioContext) initAudioContext(); // Init audio context on connect
                    };
                    websocket.onclose = function(evt) {
                        statusSpan.textContent = "Disconnected";
                        logInteraction(`WebSocket Disconnected: Code=${evt.code}, Reason=${evt.reason |
| 'N/A'}, WasClean=${evt.wasClean}`, 'system');
                        startButton.disabled = true; stopButton.disabled = true;
                        if (mediaRecorder && mediaRecorder.state!== 'inactive') {
                            try { mediaRecorder.stop(); } catch (e) { console.warn("Error stopping media recorder on WS close:", e); }
                        }
                        // Optional: Attempt to reconnect after a delay
                        // setTimeout(connectWebSocket, 5000);
                    };
                    websocket.onerror = function(evt) {
                        statusSpan.textContent = "Error";
                        logInteraction('WebSocket Error occurred. See browser console.', 'system');
                        console.error('WebSocket Error:', evt);
                        startButton.disabled = true; stopButton.disabled = true;
                    };
                    websocket.onmessage = function(evt) {
                        if (evt.data instanceof ArrayBuffer) {
                            // Received binary data (assume it's audio from the agent)
                            console.log(`Received audio chunk: ${evt.data.byteLength} bytes`);
                            playAudioChunk(evt.data);
                        } else if (typeof evt.data === 'string') {
                            // Received text data (JSON messages)
                            try {
                                const msg = JSON.parse(evt.data);
                                if (msg.type === 'interim_transcript') logInteraction(msg.transcript, 'interim');
                                else if (msg.type === 'final_transcript') logInteraction(msg.transcript, 'user');
                                else if (msg.type === 'agent_response') logInteraction(msg.response, 'agent');
                                else if (msg.type === 'status' |
| msg.type === 'info') logInteraction(msg.message, 'system');
                                else if (msg.type === 'error') logInteraction(`Error from server: ${msg.message}`, 'system');
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
                     startButton.disabled = true; stopButton.disabled = true;
                }
            }

            startButton.onclick = async () => {
                // Ensure audio context is ready and resumed
                if (!audioContext) { if (!initAudioContext()) return; }
                if (audioContext.state === 'suspended') {
                    try { await audioContext.resume(); } catch (e) { logInteraction("Failed to resume audio context.", "system"); return; }
                }
                if (audioContext.state!== 'running') {
                    logInteraction("Audio context not running. Cannot start recording.", "system"); return;
                }

                if (!websocket |
| websocket.readyState!== WebSocket.OPEN) {
                     logInteraction("WebSocket not open. Cannot start recording.", 'system'); return;
                }
                // --- MODIFICATION: Send start_activity signal ---
                sendControlMessage("start_activity");
                // --- End Modification ---

                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    // Use a common, widely supported mimeType if opus isn't available everywhere
                    const options = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
                       ? { mimeType: 'audio/webm;codecs=opus' }
                        : MediaRecorder.isTypeSupported('audio/ogg;codecs=opus')
                           ? { mimeType: 'audio/ogg;codecs=opus' }
                            : {}; // Let browser choose default if opus isn't supported
                    if (!options.mimeType &&!MediaRecorder.isTypeSupported('')) { // Check if *any* recording is supported
                         logInteraction(`Error: Browser does not support WebM/Opus or Ogg/Opus, and no default available.`, 'system');
                         sendControlMessage("end_activity"); // Send end if start was sent but recorder failed
                         return;
                    }
                    logInteraction(`Using mimeType: ${options.mimeType |
| 'browser default'}`, 'system');

                    mediaRecorder = new MediaRecorder(stream, options);
                    audioChunks =; // Clear previous chunks if any

                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0 && websocket && websocket.readyState === WebSocket.OPEN) {
                            websocket.send(event.data); // Send raw audio chunk (WebM or Ogg)
                        }
                    };

                    mediaRecorder.onstop = () => {
                        logInteraction("Recording stopped.", 'system');
                        startButton.disabled = false; stopButton.disabled = true;
                        // --- MODIFICATION: Send end_activity signal ---
                        sendControlMessage("end_activity");
                        // --- End Modification ---
                        // Clean up tracks
                        stream.getTracks().forEach(track => track.stop());
                    };

                    mediaRecorder.onerror = (event) => {
                        logInteraction(`MediaRecorder Error: ${event.error}`, 'system');
                        console.error("MediaRecorder Error:", event.error);
                        // Attempt to stop cleanly if possible, which should trigger onstop->end_activity
                        if (mediaRecorder.state === 'recording') {
                            try { mediaRecorder.stop(); } catch (e) { console.warn("Error stopping recorder after error:", e); }
                        }
                        startButton.disabled = false; stopButton.disabled = true;
                        // Ensure tracks are stopped even if onstop doesn't fire
                        stream.getTracks().forEach(track => track.stop());
                    };

                    mediaRecorder.start(250); // Send chunks every 250ms
                    logInteraction("Recording started...", 'system');
                    startButton.disabled = true; stopButton.disabled = false; // Enable stop button

                } catch (err) {
                    logInteraction("Error accessing microphone or starting recorder: " + err.message, 'system');
                    console.error("Mic/Recorder Error:", err);
                     // --- MODIFICATION: Send end_activity signal if start failed after sending start_activity ---
                    sendControlMessage("end_activity");
                    // --- End Modification ---
                    startButton.disabled = false; // Re-enable start button on error
                    stopButton.disabled = true;
                }
            };

            stopButton.onclick = () => {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop(); // This triggers onstop handler above which now sends end_activity
                } else {
                    logInteraction("Not recording.", "system"); // User feedback
                }
            };

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
async def transcode_audio_ffmpeg(input_bytes: bytes, client_id: str) -> bytes | None:
    """Transcodes audio bytes using ffmpeg-python. Logs include client_id."""
    # Recommendation: Investigate performance implications of per-chunk ffmpeg process for high load.
    # Consider persistent processes or checking if API supports Opus directly.
    if not ffmpeg:
        ws_logger.error(f"[{client_id}] ffmpeg-python library not available. Cannot transcode.")
        return None
    try:
        # Start ffmpeg process asynchronously
        process = await asyncio.create_subprocess_exec(
            'ffmpeg',
            '-f', 'webm', # Assume input is webm (adjust if frontend changes mimeType)
            '-i', 'pipe:0', # Input from stdin
            '-f', TARGET_FORMAT, # Output format (e.g., s16le)
            '-acodec', 'pcm_s16le', # Output codec
            '-ac', str(TARGET_CHANNELS), # Output channels (e.g., 1)
            '-ar', str(TARGET_SAMPLE_RATE), # Output sample rate (e.g., 16000)
            'pipe:1', # Output to stdout
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        # Write input and read output/error streams
        stdout, stderr = await process.communicate(input=input_bytes)
        # Check return code
        if process.returncode!= 0:
            ws_logger.error(f"[{client_id}] FFmpeg failed (code {process.returncode}): {stderr.decode(errors='ignore')}")
            return None
        # ws_logger.info(f"[{client_id}] FFmpeg transcoding successful: {len(input_bytes)} -> {len(stdout)} bytes") # Verbose log
        return stdout
    except FileNotFoundError:
        ws_logger.error(f"[{client_id}] ffmpeg command not found. Ensure ffmpeg is installed and in PATH.")
        return None
    except Exception as e:
        ws_logger.error(f"[{client_id}] Unexpected transcoding error: {e}", exc_info=True)
        return None


# --- WebSocket Endpoint ---
@app.websocket("/ws/audio_gemini")
async def websocket_endpoint_gemini(websocket: WebSocket):
    """Handles WebSocket connections for audio streaming using client.aio.live."""
    await websocket.accept()
    client_id = f"{USER_ID_PREFIX}{uuid.uuid4()}"
    ws_logger.info(f"[{client_id}] WebSocket connection accepted from {websocket.client.host}:{websocket.client.port}")
    adk_session_id = None
    live_session = None
    send_task = None
    receive_task = None
    audio_queue = asyncio.Queue() # Queue for incoming audio chunks from client

    # Helper to safely send messages to WebSocket client
    async def safe_send_text(message: str):
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_text(message)
            except WebSocketDisconnect:
                ws_logger.warning(f"[{client_id}] Client disconnected before text message could be sent.")
            except Exception as ws_send_err:
                ws_logger.error(f"[{client_id}] Error sending text message to WebSocket: {ws_send_err}", exc_info=True)
        else:
            ws_logger.warning(f"[{client_id}] WebSocket no longer connected; skipping text send.")

    async def safe_send_bytes(data: bytes):
        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.send_bytes(data)
            except WebSocketDisconnect:
                ws_logger.warning(f"[{client_id}] Client disconnected before bytes message could be sent.")
            except Exception as ws_send_err:
                ws_logger.error(f"[{client_id}] Error sending bytes message to WebSocket: {ws_send_err}", exc_info=True)
        else:
            ws_logger.warning(f"[{client_id}] WebSocket no longer connected; skipping bytes send.")

    # Helper to safely send messages to Gemini Live session
    async def safe_live_send(message: LiveClientMessage):
        if live_session:
            try:
                await live_session.send(message)
            except Exception as send_err:
                ws_logger.error(f"[{client_id}/{adk_session_id}] Error sending message to Gemini: {send_err}", exc_info=True)
                # Optionally, notify client or raise to trigger cleanup
                await safe_send_text(json.dumps({"type": "error", "message": f"Server error communicating with Speech API: {send_err}"}))
                raise # Re-raise to potentially stop the send task

    try:
        # --- Initial Setup Checks ---
        if not GOOGLE_API_KEY: raise ValueError("Server not configured with GOOGLE_API_KEY.")
        if not genai: raise ImportError("Google Generative AI library (google.genai) could not be imported.")
        if GEMINI_LIVE_CONFIG is None: raise ImportError("Required google.genai types could not be imported or config failed.")
        if not ffmpeg: await safe_send_text(json.dumps({"type": "warning", "message": "Audio transcoding might fail: ffmpeg not available on server."}))

        # --- ADK Session ---
        adk_session_id = get_or_create_adk_session_sync(client_id) # Use sync helper
        await safe_send_text(json.dumps({"type": "info", "message": f"ADK Session Ready: {adk_session_id}"}))

        # --- Gemini Live Connection ---
        client = genai.Client() # Assuming genai.configure was successful
        ws_logger.info(f"[{client_id}/{adk_session_id}] Initializing Gemini live session...")

        async with client.aio.live.connect(
            model=GEMINI_LIVE_MODEL_NAME,
            config=GEMINI_LIVE_CONFIG # Config now includes disabled auto activity detection
        ) as session:
            live_session = session # Assign to outer scope variable for cleanup
            ws_logger.info(f"[{client_id}/{adk_session_id}] Gemini live session established.")
            await safe_send_text(json.dumps({"type": "info", "message": "Manual activity detection active. Use buttons."}))

            # --- Background Tasks ---

            # Task to receive audio from client, transcode, and send to Gemini
            async def send_audio_to_gemini():
                while True:
                    try:
                        webm_chunk = await audio_queue.get()
                        if webm_chunk is None: # Sentinel value to stop the task
                            ws_logger.info(f"[{client_id}/{adk_session_id}] send_audio_to_gemini received stop signal.")
                            break
                        if ffmpeg:
                            pcm_chunk = await transcode_audio_ffmpeg(webm_chunk, client_id)
                            if pcm_chunk:
                                # Send audio using LiveClientMessage
                                await live_session.send(LiveClientMessage(
                                    realtime_input=types.LiveClientRealtimeInput(
                                        media_chunks=[types.Blob(data=pcm_chunk, mime_type="audio/pcm")]
                                    )
                                ))
                                # ws_logger.debug(f"[{client_id}/{adk_session_id}] Sent {len(pcm_chunk)} audio bytes to Gemini.")
                            else:
                                ws_logger.warning(f"[{client_id}/{adk_session_id}] Transcoding failed for audio chunk.")
                                # Optionally notify client about transcoding failure
                                # await safe_send_text(json.dumps({"type": "error", "message": "Server failed to process audio chunk."}))
                        else:
                            # This case should ideally be prevented by initial checks, but log defensively
                            ws_logger.error(f"[{client_id}/{adk_session_id}] Cannot process audio: ffmpeg not available.")
                            # Stop processing if ffmpeg is missing
                            break
                        audio_queue.task_done() # Mark item as processed
                    except asyncio.CancelledError:
                        ws_logger.info(f"[{client_id}/{adk_session_id}] send_audio_to_gemini task cancelled.")
                        break
                    except Exception as e:
                        ws_logger.error(f"[{client_id}/{adk_session_id}] Error in send_audio_to_gemini loop: {e}", exc_info=True)
                        # Consider breaking the loop on persistent errors
                        break
                ws_logger.info(f"[{client_id}/{adk_session_id}] send_audio_to_gemini task finished.")


            # Task to receive from Gemini and interact with ADK/Client
            async def receive_from_gemini():
                final_transcript_buffer = ""
                try:
                    async for response in live_session: # Iterate through responses from Gemini
                        if not response: continue # Skip empty responses

                        # Handle Errors from Gemini first
                        # Note: Consult API docs for exact error structure if different
                        if hasattr(response, 'error') and response.error:
                            ws_logger.error(f"[{client_id}/{adk_session_id}] Gemini live session error: {response.error}")
                            await safe_send_text(json.dumps({"type": "error", "message": f"Speech API Error: {response.error}"}))
                            break # Stop processing on API error

                        # Handle Server Content (Transcripts, Audio)
                        if hasattr(response, 'server_content') and response.server_content:
                            server_content = response.server_content

                            # Handle Text Response (Transcript)
                            if hasattr(server_content, 'input_transcription') and server_content.input_transcription:
                                transcript_data = server_content.input_transcription
                                if hasattr(transcript_data, 'text') and transcript_data.text:
                                    if hasattr(transcript_data, 'finished') and transcript_data.finished:
                                        final_transcript_buffer += transcript_data.text
                                        ws_logger.info(f"[{client_id}/{adk_session_id}] Final Transcript Segment: '{transcript_data.text}' -> Full: '{final_transcript_buffer}'")
                                        await safe_send_text(json.dumps({"type": "final_transcript", "transcript": final_transcript_buffer}))

                                        # --- Trigger ADK Interaction ---
                                        if final_transcript_buffer.strip(): # Only run ADK if there's content
                                            await safe_send_text(json.dumps({"type": "info", "message": "Sending to agent..."}))
                                            loop = asyncio.get_running_loop()
                                            try:
                                                # Recommendation: Be aware this blocks this task until ADK responds.
                                                agent_response = await loop.run_in_executor(
                                                    None, # Use default thread pool executor
                                                    run_adk_turn_sync,
                                                    client_id,
                                                    adk_session_id,
                                                    final_transcript_buffer
                                                )
                                                await safe_send_text(json.dumps({"type": "agent_response", "response": agent_response}))
                                            except Exception as adk_err:
                                                ws_logger.error(f"[{client_id}/{adk_session_id}] Error running ADK turn: {adk_err}", exc_info=True)
                                                await safe_send_text(json.dumps({"type": "error", "message": f"Error interacting with agent: {adk_err}"}))
                                        else:
                                             ws_logger.info(f"[{client_id}/{adk_session_id}] Empty final transcript received, skipping ADK.")
                                        # --- End ADK Interaction ---

                                        final_transcript_buffer = "" # Reset buffer for next utterance
                                    elif STREAMING_INTERIM_RESULTS:
                                        interim_text = final_transcript_buffer + transcript_data.text
                                        # ws_logger.debug(f"[{client_id}/{adk_session_id}] Interim Transcript: '{interim_text}'")
                                        await safe_send_text(json.dumps({"type": "interim_transcript", "transcript": interim_text}))
                                    else: # Not finished, but not streaming interim
                                        final_transcript_buffer += transcript_data.text # Append to buffer anyway

                            # Handle Audio Response (TTS)
                            if hasattr(server_content, 'model_turn') and server_content.model_turn:
                                if hasattr(server_content.model_turn, 'parts'):
                                    for part in server_content.model_turn.parts:
                                        # Check for inline audio data
                                        if hasattr(part, 'inline_data') and part.inline_data and hasattr(part.inline_data, 'data') and part.inline_data.data:
                                            ws_logger.info(f"[{client_id}/{adk_session_id}] Received audio data: {len(part.inline_data.data)} bytes")
                                            await safe_send_bytes(part.inline_data.data) # Send raw audio bytes to client

                        # Handle other potential response types if documented by the API

                except asyncio.CancelledError:
                    ws_logger.info(f"[{client_id}/{adk_session_id}] receive_from_gemini task cancelled.")
                # Catch specific API exceptions if documented, e.g., StopCandidateException
                except types.generation_types.StopCandidateException as stop_ex:
                     ws_logger.info(f"[{client_id}/{adk_session_id}] Gemini stream stopped (StopCandidateException): {stop_ex}")
                     await safe_send_text(json.dumps({"type": "info", "message": "Speech stream ended normally."}))
                except Exception as e:
                    ws_logger.error(f"[{client_id}/{adk_session_id}] Error in receive_from_gemini loop: {e}", exc_info=True)
                    await safe_send_text(json.dumps({"type": "error", "message": f"Server error processing speech response: {e}"}))
                finally:
                    ws_logger.info(f"[{client_id}/{adk_session_id}] receive_from_gemini task finished.")


            # --- Start Background Tasks ---
            send_task = asyncio.create_task(send_audio_to_gemini())
            receive_task = asyncio.create_task(receive_from_gemini())

            # --- Main Loop: Receive data/signals from client WebSocket ---
            while True: # Loop until disconnect or error in tasks
                # Check if background tasks are still running
                if (send_task and send_task.done()) or (receive_task and receive_task.done()):
                    ws_logger.warning(f"[{client_id}/{adk_session_id}] A background task finished unexpectedly. Closing connection.")
                    if send_task and send_task.done() and send_task.exception():
                         ws_logger.error(f"[{client_id}/{adk_session_id}] Send task exception: {send_task.exception()}")
                    if receive_task and receive_task.done() and receive_task.exception():
                         ws_logger.error(f"[{client_id}/{adk_session_id}] Receive task exception: {receive_task.exception()}")
                    break # Exit main loop to trigger cleanup

                try:
                    data = await websocket.receive() # Wait for message from client
                except WebSocketDisconnect:
                    ws_logger.info(f"[{client_id}/{adk_session_id}] WebSocket disconnect received in main loop.")
                    break # Exit loop on disconnect

                if data['type'] == 'websocket.disconnect':
                    ws_logger.info(f"[{client_id}/{adk_session_id}] WebSocket disconnect message processed.")
                    break # Exit loop on disconnect message

                elif data['type'] == 'websocket.receive': # Starlette structure
                    if 'bytes' in data and data['bytes']:
                        # ws_logger.debug(f"[{client_id}/{adk_session_id}] Received {len(data['bytes'])} audio bytes from client.")
                        await audio_queue.put(data['bytes']) # Queue raw WebM/Ogg audio for transcoding
                    elif 'text' in data and data['text']:
                        ws_logger.info(f"[{client_id}/{adk_session_id}] Received text from client: {data['text'][:100]}")
                        try:
                            msg = json.loads(data['text'])
                            if msg.get("type") == "control":
                                action = msg.get("action")
                                if action == "start_activity":
                                    ws_logger.info(f"[{client_id}/{adk_session_id}] Received start_activity signal.")
                                    # Send ActivityStart signal to Gemini
                                    await safe_live_send(LiveClientMessage(activity_start=ActivityStart()))
                                elif action == "end_activity":
                                    ws_logger.info(f"[{client_id}/{adk_session_id}] Received end_activity signal.")
                                    # Send ActivityEnd signal to Gemini
                                    await safe_live_send(LiveClientMessage(activity_end=ActivityEnd()))
                                    # Note: No need to signal end of audio queue here, let send_task handle it.
                                else:
                                    ws_logger.warning(f"[{client_id}/{adk_session_id}] Received unknown control action: {action}")
                                    await safe_send_text(json.dumps({"type": "warning", "message": f"Unknown control action received: {action}"}))
                            else:
                                ws_logger.warning(f"[{client_id}/{adk_session_id}] Received unknown JSON message structure: {data['text'][:100]}")
                                await safe_send_text(json.dumps({"type": "warning", "message": "Unknown message type received."}))
                        except json.JSONDecodeError:
                            ws_logger.warning(f"[{client_id}/{adk_session_id}] Received non-JSON text: {data['text'][:100]}")
                            await safe_send_text(json.dumps({"type": "warning", "message": "Received invalid message format."}))
                        except Exception as e:
                            ws_logger.error(f"[{client_id}/{adk_session_id}] Error processing text message: {e}", exc_info=True)
                            await safe_send_text(json.dumps({"type": "error", "message": "Server error processing your message."}))

            ws_logger.info(f"[{client_id}/{adk_session_id}] Exited main WS receive loop.")

    # --- Exception Handling & Cleanup ---
    except WebSocketDisconnect:
        ws_logger.info(f"[{client_id}/{adk_session_id}] WS client disconnected during setup or main loop.")
    except ImportError as e:
        ws_logger.error(f"[{client_id}] Startup failed due to ImportError: {e}", exc_info=True)
        await safe_send_text(json.dumps({"type": "error", "message": f"Server Import Error: {e}. Cannot start session."}))
    except ValueError as e:
        ws_logger.error(f"[{client_id}] Startup failed due to ValueError: {e}", exc_info=True)
        await safe_send_text(json.dumps({"type": "error", "message": f"Server Config Error: {e}. Cannot start session."}))
    except Exception as e:
        # Catch any other unexpected errors during setup or the live session context manager
        ws_logger.error(f"[{client_id}/{adk_session_id}] Unexpected error in WS handler: {e}", exc_info=True)
        await safe_send_text(json.dumps({"type": "error", "message": f"Unexpected server error: {str(e)}"}))
    finally:
        ws_logger.info(f"[{client_id}/{adk_session_id}] Closing WS connection & cleaning up resources.")

        # 1. Cancel background tasks safely
        if send_task and not send_task.done():
            ws_logger.info(f"[{client_id}/{adk_session_id}] Cancelling send_audio_to_gemini task.")
            send_task.cancel()
        if receive_task and not receive_task.done():
            ws_logger.info(f"[{client_id}/{adk_session_id}] Cancelling receive_from_gemini task.")
            receive_task.cancel()

        # 2. Signal end of audio queue for send_task to exit cleanly if running
        try:
            # Use put_nowait in finally block to avoid blocking if queue is full or task already done
            audio_queue.put_nowait(None)
        except asyncio.QueueFull:
             ws_logger.warning(f"[{client_id}/{adk_session_id}] Audio queue full during cleanup, send task might not stop gracefully.")
        except Exception as q_err:
             ws_logger.error(f"[{client_id}/{adk_session_id}] Error putting None sentinel in audio queue: {q_err}")


        # 3. Wait for tasks to finish cancellation (with timeout)
        tasks = [t for t in [send_task, receive_task] if t]
        if tasks:
            try:
                # Wait briefly for tasks to finish after cancellation
                await asyncio.wait(tasks, timeout=2.0)
            except asyncio.TimeoutError:
                 ws_logger.warning(f"[{client_id}/{adk_session_id}] Timeout waiting for background tasks to cancel.")
            except Exception as gather_err:
                 ws_logger.error(f"[{client_id}/{adk_session_id}] Error during task cleanup gathering: {gather_err}", exc_info=True)

        # 4. Clean up ADK session tracking (optional, depending on session lifecycle needs)
        if client_id in active_adk_sessions:
            try:
                del active_adk_sessions[client_id]
                ws_logger.info(f"[{client_id}/{adk_session_id}] Removed ADK session link.")
            except KeyError:
                 ws_logger.warning(f"[{client_id}/{adk_session_id}] Client ID not found in active ADK sessions during cleanup.")


        # 5. Close WebSocket connection if not already closed
        if websocket.client_state!= WebSocketState.DISCONNECTED:
            try:
                ws_logger.info(f"[{client_id}/{adk_session_id}] Closing WebSocket connection server-side.")
                await websocket.close()
            except Exception as e:
                ws_logger.error(f"[{client_id}/{adk_session_id}] Error closing WebSocket: {e}", exc_info=True)

        ws_logger.info(f"[{client_id}/{adk_session_id}] Cleanup finished.")


# --- Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server directly with Uvicorn...")
    if not GOOGLE_API_KEY: logger.warning("WARNING: GOOGLE_API_KEY env var not set.")
    if not genai: logger.critical("CRITICAL: google.genai library failed to import. API endpoints will likely fail.")
    if ffmpeg is None:
        logger.warning("WARNING: ffmpeg-python not found or ffmpeg executable missing. Audio transcoding will fail.")
    # Use reload=True only for development
    uvicorn.run("ui.api:app", host="0.0.0.0", port=8000, reload=True, log_config=None) # Disable uvicorn default logging to use ours