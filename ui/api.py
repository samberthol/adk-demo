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
from google.genai import types # Keep top-level types import

# Import specific types needed for Live API (v1alpha)
try:
    from google.genai.types import (
        Content, Part, HttpOptions, # HttpOptions for client init
        LiveConnectConfig, SpeechConfig, VoiceConfig, PrebuiltVoiceConfig, # Config types
        LiveClientMessage, ActivityStart, ActivityEnd, # Message types
        Blob, LiveClientRealtimeInput, # Input types
        LiveServerMessage, Transcription, # Server message types
        UsageMetadata # Common type
    )
    # *** Use "Kore" voice based on docs example ***
    GEMINI_LIVE_CONFIG = LiveConnectConfig(
        response_modalities=["TEXT", "AUDIO"], # Keep both TEXT and AUDIO
        speech_config=SpeechConfig(
            voice_config=VoiceConfig(
                # Change voice name here
                prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Kore") # Use "Kore"
            )
        )
    )
    live_types_imported = True
except ImportError as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger_init = logging.getLogger("fastapi_app_init_error")
    logger_init.error(f"Critical Import Error for google.genai.types (potentially Live API related): {e}. Live API will likely fail.")
    live_types_imported = False
    GEMINI_LIVE_CONFIG = None
except Exception as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger_init = logging.getLogger("fastapi_app_init_error")
    logger_init.error(f"Error defining GEMINI_LIVE_CONFIG: {e}")
    live_types_imported = False
    GEMINI_LIVE_CONFIG = None


# --- ADK Agent Imports ---
try:
    from agents.meta.agent import meta_agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
except ImportError as e:
     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
     logger_init = logging.getLogger("fastapi_app_init_error")
     logger_init.error(f"FastAPI: Failed to import agent modules/ADK components: {e}")
     sys.exit(f"FastAPI startup failed due to missing ADK components: {e}")

# --- Transcoding Import ---
try:
    import ffmpeg
except ImportError:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger_init = logging.getLogger("fastapi_app_init_error")
    logger_init.warning("ffmpeg-python not installed. Audio input cannot be transcoded.")
    ffmpeg = None

# --- Basic Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fastapi_app") # Main logger for the app

APP_NAME = "gcp_multi_agent_demo_api"
USER_ID_PREFIX = "fastapi_user_"
ADK_SESSION_PREFIX = f'adk_session_{APP_NAME}_'
GEMINI_LIVE_MODEL_NAME = "models/gemini-2.0-flash-live-001" # Use the requested live model
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
STREAMING_INTERIM_RESULTS = True # For text transcripts

# Audio constants (Keep these for transcoding)
TARGET_SAMPLE_RATE = 16000
TARGET_CHANNELS = 1
TARGET_FORMAT = 's16le' # Raw PCM signed 16-bit little-endian

# --- ADK Initialization ---
session_service = InMemorySessionService()
adk_runner = Runner(agent=meta_agent, app_name=APP_NAME, session_service=session_service)
active_adk_sessions = {}

# --- Google Generative AI Configuration (Using v1alpha for the live model) ---
client = None # Initialize client variable globally
# ... (client initialization remains the same, using v1alpha) ...
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY environment variable not set.")
elif not live_types_imported or HttpOptions is None:
     logger.error("Cannot configure client: Required google.genai types failed to import.")
else:
    try:
        client = genai.Client(
            api_key=GOOGLE_API_KEY,
            http_options=HttpOptions(api_version='v1alpha') # Keep v1alpha
        )
        logger.info("Google Generative AI client configured targeting v1alpha API.")
    except Exception as e:
         logger.error(f"Failed to configure Google Generative AI client (targeting v1alpha): {e}")


# --- ADK Interaction Functions ---
# (get_or_create_adk_session_sync remains the same)
def get_or_create_adk_session_sync(user_id: str) -> str:
    # ... (code remains the same) ...
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

# (run_adk_turn_sync remains the same)
def run_adk_turn_sync(user_id: str, session_id: str, user_message_text: str) -> str:
    # ... (code remains the same) ...
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

# --- Frontend HTML/JS ---
# (HTML remains the same as previous version with agent response box and improved error handling)
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>FastAPI Gemini Live Audio (Manual Activity)</title>
        <style>
             body { font-family: sans-serif; padding: 1em; }
             #interaction { height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px; white-space: pre-wrap; background-color: #f9f9f9;}
             #agent-response-box { min-height: 50px; border: 1px dashed blue; padding: 10px; margin-top: 10px; background-color: #e7f0ff;}
             #agent-response-box strong { color: blue; }
        </style>
    </head>
    <body>
        <h1>Voice Interaction (Gemini Live & ADK Agent)</h1>
        <p>Status: <span id="status">Initializing</span></p>
        <button id="start">Start Recording</button>
        <button id="stop" disabled>Stop Recording</button>
        <h2>Agent Interaction Log:</h2>
        <div id="interaction"></div>
        <h2>Latest Agent Response:</h2>
        <div id="agent-response-box"><strong>Agent:</strong> Waiting for interaction...</div>

        <script>
            // (JavaScript remains the same as previous version)
            // ...
            const statusSpan = document.getElementById('status');
            const interactionDiv = document.getElementById('interaction');
            const agentResponseDiv = document.getElementById('agent-response-box');
            const startButton = document.getElementById('start');
            const stopButton = document.getElementById('stop');
            let websocket;
            let mediaRecorder;
            let audioChunks = [];
            let audioContext;
            let audioQueue = [];
            let isPlaying = false;
            let nextStartTime = 0;
            const playbackSampleRate = 24000;

            function initAudioContext() {
                if (!audioContext) {
                    try {
                        audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: playbackSampleRate });
                        console.log(`AudioContext initialized with sample rate ${audioContext.sampleRate}Hz.`);
                    } catch (e) {
                        console.error("Error creating AudioContext:", e);
                        logInteraction("Error: Cannot initialize audio playback.", "system");
                    }
                }
                return audioContext;
            }

            async function playAudioChunk(arrayBuffer) {
                 if (!audioContext || arrayBuffer.byteLength === 0) return;
                try {
                    const pcm16Data = new Int16Array(arrayBuffer);
                    const float32Data = new Float32Array(pcm16Data.length);
                    for (let i = 0; i < pcm16Data.length; i++) {
                        float32Data[i] = pcm16Data[i] / 32768.0;
                    }
                    const audioBuffer = audioContext.createBuffer(1, float32Data.length, playbackSampleRate);
                    audioBuffer.copyToChannel(float32Data, 0);
                    audioQueue.push(audioBuffer);
                    schedulePlayback();
                } catch (e) {
                     console.error("Error processing audio chunk for playback:", e);
                     logInteraction("Error playing received audio.", "system");
                }
            }

            function schedulePlayback() {
                 if (isPlaying || audioQueue.length === 0 || !audioContext) return;
                isPlaying = true;
                const bufferToPlay = audioQueue.shift();
                const source = audioContext.createBufferSource();
                source.buffer = bufferToPlay;
                source.connect(audioContext.destination);
                const currentTime = audioContext.currentTime;
                const startTime = Math.max(currentTime, nextStartTime);
                source.start(startTime);
                console.log(`Scheduled audio chunk to start at: ${startTime.toFixed(2)}s`);
                nextStartTime = startTime + bufferToPlay.duration;
                source.onended = () => { isPlaying = false; schedulePlayback(); };
            }

            function logInteraction(message, type = 'info') {
                const p = document.createElement('p');
                let prefix = '';
                if (type === 'user') prefix = '<strong>You:</strong> ';
                else if (type === 'agent') prefix = '<strong>Agent:</strong> ';
                else if (type === 'system') prefix = '<em>System:</em> ';
                else if (type === 'interim') prefix = '<em>You (interim):</em> ';
                p.innerHTML = prefix + message;
                interactionDiv.appendChild(p);
                interactionDiv.scrollTop = interactionDiv.scrollHeight;
                console.log(`${type}: ${message}`);
            }

            function updateAgentResponseBox(message) {
                 agentResponseDiv.innerHTML = '<strong>Agent:</strong> ' + message;
            }

            function sendControlMessage(action) {
                 if (websocket && websocket.readyState === WebSocket.OPEN) {
                    try {
                        const messagePayload = {};
                        if (action === "start_activity") {
                            messagePayload.activity_start = {};
                        } else if (action === "end_activity") {
                            messagePayload.activity_end = {};
                        } else {
                            logInteraction(`Unknown control action: ${action}`, 'system');
                            return;
                        }
                        websocket.send(JSON.stringify(messagePayload));
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
                 const wsProto = window.location.protocol === "https:" ? "wss:" : "ws:";
                const wsUri = `${wsProto}//${location.host}/ws/audio_gemini`;
                logInteraction(`Attempting WebSocket connection to: ${wsUri}`, 'system');
                statusSpan.textContent = "Connecting...";
                startButton.disabled = true;

                try {
                    websocket = new WebSocket(wsUri);
                    websocket.binaryType = 'arraybuffer';

                    websocket.onopen = function(evt) {
                        statusSpan.textContent = "Connected";
                        logInteraction("WebSocket Connected. Ready to record.", 'system');
                        startButton.disabled = false;
                         if (!audioContext) initAudioContext();
                    };
                    websocket.onclose = function(evt) {
                        statusSpan.textContent = "Disconnected";
                        logInteraction(`WebSocket Disconnected: Code=${evt.code}, Reason=${evt.reason || 'N/A'}, WasClean=${evt.wasClean}`, 'system');
                        startButton.disabled = true; stopButton.disabled = true;
                        if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
                    };
                    websocket.onerror = function(evt) {
                         statusSpan.textContent = "Error";
                        logInteraction('WebSocket Error occurred. See browser console.', 'system');
                        console.error('WebSocket Error:', evt);
                        startButton.disabled = true; stopButton.disabled = true;
                    };
                    websocket.onmessage = function(evt) {
                        if (evt.data instanceof ArrayBuffer) {
                            console.log(`Received audio chunk: ${evt.data.byteLength} bytes`);
                            playAudioChunk(evt.data); // Play received TTS audio
                        } else if (typeof evt.data === 'string') {
                            try {
                                const msg = JSON.parse(evt.data);

                                // --- Improved Message Handling ---
                                if (msg.type === 'error') { // Check for type: 'error' first
                                    logInteraction(`Server Error: ${msg.message || JSON.stringify(msg)}`, 'system');
                                } else if (msg.server_content) {
                                    // ... (handle server_content as before) ...
                                     const content = msg.server_content;
                                    if (content.input_transcription) {
                                        const transcript = content.input_transcription;
                                        const displayText = transcript.text || '';
                                        if (transcript.finished && displayText.trim()) {
                                            logInteraction(displayText, 'user');
                                            logInteraction("Processing request...", 'system');
                                        } else if (!transcript.finished && STREAMING_INTERIM_RESULTS && displayText.trim()) {
                                            const lastInterimElement = interactionDiv.querySelector('p:last-child > em');
                                            if (lastInterimElement && lastInterimElement.textContent === 'You (interim):') {
                                                lastInterimElement.parentNode.innerHTML = '<em>You (interim):</em> ' + displayText;
                                            } else {
                                                logInteraction(displayText, 'interim');
                                            }
                                            interactionDiv.scrollTop = interactionDiv.scrollHeight;
                                        }
                                    }
                                    if (content.model_turn && content.model_turn.parts) {
                                        const textParts = content.model_turn.parts.filter(p => p.text).map(p => p.text).join('');
                                        if (textParts) {
                                            logInteraction(textParts, 'agent');
                                            updateAgentResponseBox(textParts);
                                        }
                                    }
                                } else if (msg.agent_response) {
                                     logInteraction(msg.response, 'agent');
                                     updateAgentResponseBox(msg.response);
                                } else if (msg.tool_call) {
                                    logInteraction(`Agent wants to call tool: ${JSON.stringify(msg.tool_call)}`, 'system');
                                } else if (msg.setup_complete) {
                                    logInteraction("Server setup complete.", 'system');
                                } else if (msg.go_away) {
                                     logInteraction(`Server GoAway: ${msg.go_away.time_left || 'No time specified'}`, 'system');
                                } else if (msg.session_resumption_update) {
                                     logInteraction(`Session resumption update. Resumable: ${msg.session_resumption_update.resumable}`, 'system');
                                } else if (msg.type === 'info' || msg.type === 'status') {
                                     logInteraction(msg.message, 'system');
                                } else {
                                    logInteraction(`Unknown server message structure received.`, 'system');
                                    console.log("Unknown server message:", msg);
                                }
                                // --- End Improved Message Handling ---

                            } catch (e) {
                                logInteraction(`Received non-JSON/unparsed text message: ${evt.data}`, 'system');
                                console.error("Failed to parse text message:", e);
                            }
                        } else {
                            console.warn("Received unexpected message type:", typeof evt.data);
                        }
                    };
                } catch (err) {
                    // ... (connectWebSocket catch remains the same) ...
                     statusSpan.textContent = "Error";
                       logInteraction(`Error creating WebSocket: ${err}`, 'system');
                       console.error("Error creating WebSocket:", err);
                       startButton.disabled = true; stopButton.disabled = true;
                }
            }

            startButton.onclick = async () => {
                 // ... (startButton.onclick remains the same) ...
                 if (!audioContext) { if (!initAudioContext()) return; }
                if (audioContext.state === 'suspended') { await audioContext.resume(); }

                if (!websocket || websocket.readyState !== WebSocket.OPEN) {
                     logInteraction("WebSocket not open. Cannot start recording.", 'system'); return;
                }
                sendControlMessage("start_activity");

                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    const options = { mimeType: 'audio/webm;codecs=opus' };
                    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                         logInteraction(`Warning: Browser does not support ${options.mimeType}. Using default.`, 'system');
                         delete options.mimeType;
                    }

                    mediaRecorder = new MediaRecorder(stream, options);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0 && websocket && websocket.readyState === WebSocket.OPEN) {
                            websocket.send(event.data);
                        }
                    };
                    mediaRecorder.onstop = () => {
                        logInteraction("Recording stopped.", 'system');
                        startButton.disabled = false; stopButton.disabled = true;
                        sendControlMessage("end_activity");
                        stream.getTracks().forEach(track => track.stop());
                    };
                    mediaRecorder.start(250);
                    logInteraction("Recording started...", 'system');
                    startButton.disabled = true; stopButton.disabled = false;
                } catch (err) {
                    logInteraction("Error accessing microphone or starting recorder: " + err, 'system');
                    console.error("Mic/Recorder Error:", err);
                    sendControlMessage("end_activity");
                }
            };

            stopButton.onclick = () => {
                 // ... (stopButton.onclick remains the same) ...
                 if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop();
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
async def transcode_audio_ffmpeg(input_bytes: bytes) -> bytes | None:
    # ... (transcode_audio_ffmpeg remains the same) ...
    """Transcodes audio bytes using ffmpeg-python to PCM S16LE @ 16kHz."""
    if not ffmpeg: logger.error("ffmpeg-python library not available."); return None
    try:
        logger.info(f"[{uuid.uuid4()}] Starting transcoding for {len(input_bytes)} bytes...")
        process = (
            ffmpeg
            .input('pipe:0')
            .output('pipe:1', format=TARGET_FORMAT, acodec='pcm_s16le', ac=TARGET_CHANNELS, ar=TARGET_SAMPLE_RATE)
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=False)
        )
        stdout, stderr = await asyncio.to_thread(process.communicate, input=input_bytes)
        if process.returncode != 0:
            logger.error(f"[{uuid.uuid4()}] FFmpeg failed: {process.returncode}\nStderr: {stderr.decode()}")
            return None
        logger.info(f"[{uuid.uuid4()}] Transcoding successful. Output size: {len(stdout)} bytes.")
        return stdout
    except ffmpeg.Error as e:
        logger.error(f"[{uuid.uuid4()}] ffmpeg-python error: {e}\n{getattr(e, 'stderr', b'').decode()}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"[{uuid.uuid4()}] Unexpected transcoding error: {e}", exc_info=True)
        return None


# --- WebSocket Endpoint ---
@app.websocket("/ws/audio_gemini")
async def websocket_endpoint_gemini(websocket: WebSocket):
    # ... (websocket_endpoint_gemini remains the same, using "Kore" voice config and introspection logging) ...
    await websocket.accept()
    client_id = f"{USER_ID_PREFIX}{uuid.uuid4()}"
    logger.info(f"WebSocket connection accepted: {client_id}")
    adk_session_id = None
    live_session = None
    send_task = None
    receive_task = None
    audio_queue = asyncio.Queue()

    try:
        if client is None: raise ValueError("Google AI Client not initialized.")
        if not live_types_imported or GEMINI_LIVE_CONFIG is None: raise ImportError("Required Live API types or config failed.")

        adk_session_id = get_or_create_adk_session_sync(client_id)
        await websocket.send_text(json.dumps({"type": "info", "message": f"ADK Session Ready: {adk_session_id}"}))

        logger.info(f"[{client_id}] Initializing Gemini live session (model: {GEMINI_LIVE_MODEL_NAME}, config: {GEMINI_LIVE_CONFIG}) using configured client (v1alpha)...") # Log config

        async with client.aio.live.connect(
            model=GEMINI_LIVE_MODEL_NAME,
            config=GEMINI_LIVE_CONFIG # Use the config with "Kore" voice
        ) as live_session:
            logger.info(f"[{client_id}] Gemini live session established.")
            logger.info(f"[{client_id}] live_session type: {type(live_session)}")
            try: logger.info(f"[{client_id}] live_session dir: {dir(live_session)}")
            except Exception as inspect_err: logger.error(f"[{client_id}] Error inspecting live_session: {inspect_err}")

            await websocket.send_text(json.dumps({"type": "info", "message": "Using manual activity detection (start/stop buttons)."}))

            # Task to send audio to Gemini
            async def send_audio_to_gemini():
                # ... (code remains the same) ...
                logger.info(f"[{client_id}] send_audio_to_gemini task started.")
                while True:
                    try:
                        logger.debug(f"[{client_id}] Waiting for audio chunk from queue...")
                        input_chunk = await audio_queue.get()
                        if input_chunk is None:
                            logger.info(f"[{client_id}] Received None signal, exiting send_audio_to_gemini.")
                            break
                        logger.info(f"[{client_id}] Got audio chunk ({len(input_chunk)} bytes) from queue.")

                        transcoded_chunk = None
                        if ffmpeg:
                            transcoded_chunk = await transcode_audio_ffmpeg(input_chunk)
                        else:
                             logger.error(f"[{client_id}] Cannot process audio: ffmpeg not available.")
                             audio_queue.task_done()
                             continue

                        if transcoded_chunk:
                            logger.info(f"[{client_id}] Sending {len(transcoded_chunk)} bytes of PCM audio to Gemini...")
                            await live_session.send(LiveClientMessage(
                                realtime_input=LiveClientRealtimeInput(
                                    media_chunks=[Blob(data=transcoded_chunk, mime_type="audio/pcm")]
                                )
                            ))
                            logger.info(f"[{client_id}] Successfully sent audio chunk to Gemini.")
                        else:
                            logger.warning(f"[{client_id}] Transcoding failed, audio chunk skipped.")

                        audio_queue.task_done()
                    except asyncio.CancelledError:
                        logger.info(f"[{client_id}] send_audio_to_gemini task cancelled.")
                        break
                    except Exception as e:
                        logger.error(f"[{client_id}] Error in send_audio_to_gemini loop: {e}", exc_info=True)
                        break

            # Task to receive responses from Gemini
            async def receive_from_gemini():
                # ... (code remains the same, using live_session.receive()) ...
                logger.info(f"[{client_id}] receive_from_gemini task started.")
                final_transcript_buffer = ""
                try:
                    logger.info(f"[{client_id}] Attempting 'async for' loop over live_session.receive()...")
                    async for response in live_session.receive(): # Using .receive() based on examples
                        logger.info(f"[{client_id}] Received message structure from Gemini: {type(response)}")

                        if not response:
                            logger.warning(f"[{client_id}] Received empty response object from Gemini.")
                            continue
                        response_dict = {}
                        try:
                            # Build response dict based on message type
                            if response.server_content:
                                content_dict = {}
                                if response.server_content.input_transcription:
                                    transcript = response.server_content.input_transcription
                                    logger.info(f"[{client_id}] Received transcription part: finished={transcript.finished}, text='{transcript.text}'")
                                    content_dict["input_transcription"] = {"text": transcript.text, "finished": transcript.finished}
                                    if transcript.finished:
                                        final_transcript_buffer += transcript.text
                                        logger.info(f"[{client_id}] Final Transcript: '{final_transcript_buffer}'")
                                        if final_transcript_buffer.strip() and adk_session_id:
                                            await websocket.send_text(json.dumps({"type": "info", "message": "Transcript complete. Sending to ADK agent..."}))
                                            loop = asyncio.get_running_loop()
                                            agent_response = await loop.run_in_executor(None, run_adk_turn_sync, client_id, adk_session_id, final_transcript_buffer)
                                            await websocket.send_text(json.dumps({"type": "agent_response", "response": agent_response}))
                                        final_transcript_buffer = ""

                                if response.server_content.model_turn:
                                    parts_list = []
                                    for part in response.server_content.model_turn.parts:
                                        part_dict = {}
                                        if hasattr(part, 'text') and part.text: part_dict['text'] = part.text
                                        if hasattr(part, 'inline_data') and part.inline_data and part.inline_data.data:
                                            logger.info(f"[{client_id}] Received audio data (TTS): {len(part.inline_data.data)} bytes - Forwarding")
                                            await websocket.send_bytes(part.inline_data.data)
                                        if part_dict: parts_list.append(part_dict)
                                    if parts_list: content_dict["model_turn"] = {"parts": parts_list}

                                if content_dict: response_dict["server_content"] = content_dict

                            elif response.tool_call:
                                response_dict["tool_call"] = {"function_calls": [{"name": fc.name, "args": fc.args, "id": fc.id} for fc in response.tool_call.function_calls]}
                            elif response.setup_complete:
                                response_dict["setup_complete"] = {}
                            elif response.go_away:
                                response_dict["go_away"] = {"time_left": str(response.go_away.time_left)}
                            elif response.session_resumption_update:
                                response_dict["session_resumption_update"] = {
                                    "new_handle": response.session_resumption_update.new_handle,
                                    "resumable": response.session_resumption_update.resumable
                                }
                            elif hasattr(response, 'error') and response.error:
                                 error_detail = getattr(response.error, 'message', str(response.error))
                                 logger.error(f"[{client_id}] Gemini live session error reported: {error_detail}")
                                 response_dict["error"] = {"message": f"Live API Error: {error_detail}" }
                            else:
                                logger.warning(f"[{client_id}] Received unhandled/unknown server message structure: {response}")
                                continue

                            if hasattr(response, 'usage_metadata') and response.usage_metadata:
                                response_dict["usage_metadata"] = {
                                     "prompt_token_count": response.usage_metadata.prompt_token_count,
                                     "total_token_count": response.usage_metadata.total_token_count
                                 }

                            if response_dict:
                                logger.info(f"[{client_id}] Sending processed response to client: {json.dumps(response_dict)}")
                                await websocket.send_text(json.dumps(response_dict))

                        except Exception as processing_err:
                             logger.error(f"[{client_id}] Error processing/serializing server message: {processing_err}", exc_info=True)


                except AttributeError as ae:
                     logger.error(f"[{client_id}] AttributeError in receive_from_gemini loop (likely 'receive' missing): {ae}", exc_info=True)
                     try: await websocket.send_text(json.dumps({"type": "error", "message": f"Server Error: Failed iterating responses ({ae}). Check SDK version/API compatibility."}))
                     except Exception: pass
                except asyncio.CancelledError:
                    logger.info(f"[{client_id}] receive_from_gemini task cancelled.")
                except Exception as e:
                    logger.error(f"[{client_id}] Error in receive_from_gemini loop: {e}", exc_info=True)
                    try: await websocket.send_text(json.dumps({"type": "error", "message": f"Error processing speech stream: {e}"}))
                    except WebSocketDisconnect: pass
                    except Exception: pass
                finally:
                    logger.info(f"[{client_id}] receive_from_gemini task finished.")


            # Start background tasks
            logger.info(f"[{client_id}] Starting background tasks...")
            send_task = asyncio.create_task(send_audio_to_gemini())
            receive_task = asyncio.create_task(receive_from_gemini())

            # Main loop to receive data/signals from client WebSocket
            # (Main loop remains the same)
            logger.info(f"[{client_id}] Entering main WS receive loop...")
            while True:
                # ... (code remains the same) ...
                data = await websocket.receive()
                log_msg_type = data.get('type')
                if log_msg_type == 'websocket.receive':
                    inner_data = data.get('bytes') or data.get('text')
                    log_msg_type = 'bytes' if inner_data and isinstance(inner_data, bytes) else 'text' if inner_data else 'unknown'
                logger.info(f"[{client_id}] Received message from client, type: {log_msg_type}")


                if data['type'] == 'websocket.disconnect':
                    logger.info(f"[{client_id}] WebSocket disconnect message received.")
                    break

                if data['type'] == 'bytes':
                    logger.info(f"[{client_id}] Received {len(data['bytes'])} bytes from client. Putting in queue.")
                    await audio_queue.put(data['bytes'])
                elif data['type'] == 'text':
                    try:
                        msg = json.loads(data['text'])
                        if msg.get("activity_start") is not None:
                             logger.info(f"[{client_id}] Received start_activity signal.")
                             await live_session.send(LiveClientMessage(activity_start=ActivityStart()))
                        elif msg.get("activity_end") is not None:
                             logger.info(f"[{client_id}] Received end_activity signal.")
                             await live_session.send(LiveClientMessage(activity_end=ActivityEnd()))
                        else:
                             logger.warning(f"[{client_id}] Received unknown text message structure: {data['text']}")
                    except json.JSONDecodeError:
                        logger.warning(f"[{client_id}] Received non-JSON text: {data['text']}")
                    except AttributeError as ae:
                         logger.error(f"[{client_id}] Error sending message to live session (API incompatibility?): {ae}")
                         await websocket.send_text(json.dumps({"type": "error", "message": f"Server error interacting with Live API: {ae}"}))
                         break
                    except Exception as e:
                        logger.error(f"[{client_id}] Error processing text msg: {e}")


            logger.info(f"[{client_id}] Exited main WS receive loop.")

    # --- Exception Handling & Cleanup ---
    # (Cleanup remains the same)
    # ...
    except WebSocketDisconnect: logger.info(f"WS client {client_id} disconnected.")
    except ImportError as e:
        logger.error(f"[{client_id}] Startup failed due to ImportError: {e}")
        try: await websocket.send_text(json.dumps({"type": "error", "message": f"Server Import Error: {e}"}))
        except Exception: pass
    except ValueError as e:
        logger.error(f"[{client_id}] Startup failed due to ValueError: {e}")
        try: await websocket.send_text(json.dumps({"type": "error", "message": f"Server Config Error: {e}"}))
        except Exception: pass
    except AttributeError as ae:
         logger.error(f"[{client_id}] Failed to connect to Live API (likely API version/model incompatibility): {ae}", exc_info=True)
         try: await websocket.send_text(json.dumps({"type": "error", "message": "Server error: Cannot connect to Live API. Check API version/model compatibility."}))
         except Exception: pass
    except Exception as e:
        logger.error(f"Unexpected error in WS handler {client_id} (potentially during connect): {e}", exc_info=True)
        try:
            if websocket.client_state != WebSocketState.DISCONNECTED:
                 await websocket.send_text(json.dumps({"type": "error", "message": f"Server connection/handler error: {str(e)}"}))
        except Exception: pass
    finally:
        # (Cleanup remains the same)
        # ...
        logger.info(f"Closing WS connection & cleaning up for {client_id}.")
        if send_task and not send_task.done(): send_task.cancel()
        if receive_task and not receive_task.done(): receive_task.cancel()
        try: await audio_queue.put(None)
        except Exception: pass
        try:
            tasks = [t for t in [send_task, receive_task] if t]
            if tasks: await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as gather_err: logger.error(f"[{client_id}] Error during task cleanup: {gather_err}")
        if client_id in active_adk_sessions: del active_adk_sessions[client_id]; logger.info(f"Removed ADK session link for {client_id}.")
        try:
            ws_state = getattr(websocket, 'client_state', None)
            if ws_state and ws_state != WebSocketState.DISCONNECTED:
                 await websocket.close()
                 logger.info(f"Closed WebSocket for {client_id} from server side.")
        except Exception as e: logger.error(f"Error closing WS for {client_id}: {e}")


# --- Uvicorn Runner ---
# (Runner remains the same)
# ...
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server directly with Uvicorn...")
    if client is None:
        logger.error("FATAL: Google AI Client could not be initialized. API calls might fail.")
    if ffmpeg is None:
        logger.warning("WARNING: ffmpeg-python not found, audio transcoding will fail.")
    uvicorn.run("ui.api:app", host="0.0.0.0", port=8000, reload=True)