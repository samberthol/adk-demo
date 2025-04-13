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
import google.generativeai as genai
try:
    # Import types needed for configuration and content
    from google.genai.types import (
        Content, Part, LiveConnectConfig, SpeechConfig, VoiceConfig,
        PrebuiltVoiceConfig # Add others as needed, e.g., FinishReason
    )
    # Define basic config using imported types
    GEMINI_LIVE_CONFIG = LiveConnectConfig(
        response_modalities=["TEXT", "AUDIO"], # Request both text and audio back
        speech_config=SpeechConfig(
            voice_config=VoiceConfig(
                prebuilt_voice_config=PrebuiltVoiceConfig(voice_name="Puck") # Example voice
            )
        )
        # Add other configs like session resumption if needed
    )
except ImportError as e:
    logging.error(f"Critical Import Error for google.genai.types: {e}. Check library version.")
    # Set config to None to prevent startup if imports fail
    GEMINI_LIVE_CONFIG = None

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
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fastapi_app")

APP_NAME = "gcp_multi_agent_demo_api"
USER_ID_PREFIX = "fastapi_user_"
ADK_SESSION_PREFIX = f'adk_session_{APP_NAME}_'
GEMINI_LIVE_MODEL_NAME = "models/gemini-2.0-flash-live-001"
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
STREAMING_INTERIM_RESULTS = True

# Audio constants matching Gemini Live API requirements
TARGET_SAMPLE_RATE = 16000 # Input requires 16kHz
TARGET_CHANNELS = 1
TARGET_FORMAT = 's16le' # Raw PCM signed 16-bit little-endian

# --- ADK Initialization ---
session_service = InMemorySessionService()
adk_runner = Runner(agent=meta_agent, app_name=APP_NAME, session_service=session_service)
active_adk_sessions = {}

# --- Google Generative AI Configuration ---
if not GOOGLE_API_KEY:
     logger.error("GOOGLE_API_KEY environment variable not set.")
else:
     try: genai.configure(api_key=GOOGLE_API_KEY)
     except Exception as e: logger.error(f"Failed to configure Google Generative AI: {e}")

# --- ADK Interaction Functions (Sync for simplicity) ---
def get_or_create_adk_session_sync(user_id: str) -> str:
    # ... (function remains the same as last working version) ...
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
           # Correctly indented:
           if user_id in active_adk_sessions:
               del active_adk_sessions[user_id]
           raise # Re-raise the exception
        return session_id

def run_adk_turn_sync(user_id: str, session_id: str, user_message_text: str) -> str:
    # ... (function remains the same as last working version) ...
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

# --- Frontend HTML/JS with Audio Playback ---
# Includes AudioContext logic for playing back PCM audio received from backend
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>FastAPI Gemini Live Audio</title>
        <style> body { font-family: sans-serif; } </style>
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
            let audioChunks = [];

            // --- Audio Playback Setup ---
            let audioContext;
            let audioQueue = []; // Queue for incoming audio buffers
            let isPlaying = false;
            let nextStartTime = 0;
            const playbackSampleRate = 24000; // Gemini Live API output rate

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

            // Function to decode PCM data (received as ArrayBuffer) and play it
            async function playAudioChunk(arrayBuffer) {
                if (!audioContext || arrayBuffer.byteLength === 0) return;

                // Assuming raw 16-bit PCM Little Endian data
                // Need to convert Int16Array to Float32Array in range [-1.0, 1.0]
                const pcm16Data = new Int16Array(arrayBuffer);
                const float32Data = new Float32Array(pcm16Data.length);
                for (let i = 0; i < pcm16Data.length; i++) {
                    float32Data[i] = pcm16Data[i] / 32768.0; // Convert Int16 to Float32 range
                }

                const audioBuffer = audioContext.createBuffer(1, float32Data.length, playbackSampleRate);
                audioBuffer.copyToChannel(float32Data, 0);

                audioQueue.push(audioBuffer);
                schedulePlayback();
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

                // Calculate when this chunk ends to schedule the next one
                nextStartTime = startTime + bufferToPlay.duration;

                source.onended = () => {
                    console.log("Audio chunk finished playing.");
                    isPlaying = false;
                    // Check if more chunks are waiting
                    schedulePlayback();
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
                interactionDiv.scrollTop = interactionDiv.scrollHeight;
                console.log(`${type}: ${message}`);
            }

            function connectWebSocket() {
                const wsUri = `wss://${location.host}/ws/audio_gemini`;
                logInteraction(`Attempting WebSocket connection to: ${wsUri}`, 'system');
                statusSpan.textContent = "Connecting...";
                startButton.disabled = true;

                try {
                    websocket = new WebSocket(wsUri);
                    websocket.binaryType = 'arraybuffer'; // <<< IMPORTANT: Receive audio as ArrayBuffer

                    websocket.onopen = function(evt) {
                        statusSpan.textContent = "Connected";
                        logInteraction("WebSocket Connected. Ready to record.", 'system');
                        startButton.disabled = false;
                        // Initialize AudioContext after user interaction (or connection)
                         if (!audioContext) initAudioContext();
                    };
                    websocket.onclose = function(evt) {
                        statusSpan.textContent = "Disconnected";
                        logInteraction(`WebSocket Disconnected: Code=${evt.code}, Reason=${evt.reason || 'N/A'}, WasClean=${evt.wasClean}`, 'system');
                        startButton.disabled = true; stopButton.disabled = true;
                        if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
                        // Clean up audio context? Maybe not needed immediately.
                    };
                    websocket.onerror = function(evt) {
                        statusSpan.textContent = "Error";
                        logInteraction('WebSocket Error occurred. See browser console.', 'system');
                        console.error('WebSocket Error:', evt);
                        startButton.disabled = true; stopButton.disabled = true;
                    };
                    websocket.onmessage = function(evt) {
                        // Check if message is binary audio data or text JSON
                        if (evt.data instanceof ArrayBuffer) {
                            // Received audio data
                            console.log(`Received audio chunk: ${evt.data.byteLength} bytes`);
                            playAudioChunk(evt.data);
                        } else if (typeof evt.data === 'string') {
                            // Received text data (assume JSON)
                            try {
                                const msg = JSON.parse(evt.data);
                                if (msg.type === 'interim_transcript') logInteraction(msg.transcript, 'interim');
                                else if (msg.type === 'final_transcript') logInteraction(msg.transcript, 'user');
                                else if (msg.type === 'agent_response') logInteraction(msg.response, 'agent');
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
                     startButton.disabled = true; stopButton.disabled = true;
                }
            }

            startButton.onclick = async () => {
                 // Initialize audio context on first user interaction if not already done
                if (!audioContext) {
                    if (!initAudioContext()) return; // Stop if context fails init
                }
                // Resume context if suspended (required by some browsers)
                if (audioContext.state === 'suspended') {
                    await audioContext.resume();
                }

                if (!websocket || websocket.readyState !== WebSocket.OPEN) {
                     logInteraction("WebSocket not open. Cannot start recording.", 'system'); return;
                }
                // ... (rest of startButton.onclick remains the same) ...
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    const options = { mimeType: 'audio/webm;codecs=opus' }; // Or try 'audio/ogg;codecs=opus' if needed
                    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                         logInteraction(`Error: Browser does not support ${options.mimeType}.`, 'system'); return;
                    }
                    mediaRecorder = new MediaRecorder(stream, options);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0 && websocket && websocket.readyState === WebSocket.OPEN) {
                            // Send audio chunk (WebM/Opus) to backend for transcoding
                            websocket.send(event.data);
                        }
                    };
                    mediaRecorder.onstop = () => {
                        logInteraction("Recording stopped.", 'system');
                        startButton.disabled = false; stopButton.disabled = true;
                        if (websocket && websocket.readyState === WebSocket.OPEN) {
                             try { websocket.send(JSON.stringify({ "type": "control", "action": "stop_audio" })); } catch (e) { console.error("Error sending stop_audio:", e); }
                        }
                        stream.getTracks().forEach(track => track.stop());
                    };
                    mediaRecorder.start(250); // Send chunks every 250ms
                    logInteraction("Recording started...", 'system');
                    startButton.disabled = true; stopButton.disabled = false;
                } catch (err) {
                    logInteraction("Error accessing microphone or starting recorder: " + err, 'system');
                    console.error("Mic/Recorder Error:", err);
                }
            };

            stopButton.onclick = () => {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop(); // This triggers onstop handler above
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
    """Transcodes audio bytes using ffmpeg-python."""
    if not ffmpeg:
        logger.error("ffmpeg-python library not available. Cannot transcode.")
        return None
    try:
        process = (
            ffmpeg
            .input('pipe:0') # Input from pipe (stdin)
            .output(
                'pipe:1', # Output to pipe (stdout)
                format=TARGET_FORMAT, #'s16le',
                acodec='pcm_s16le', # Explicit codec
                ac=TARGET_CHANNELS, # 1 channel
                ar=TARGET_SAMPLE_RATE # 16kHz sample rate
            )
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True) # quiet=True suppresses ffmpeg logs
        )
        stdout, stderr = await asyncio.to_thread(process.communicate, input=input_bytes)
        # process.wait() # Not needed with communicate

        if process.returncode != 0:
            logger.error(f"FFmpeg transcoding failed. Return code: {process.returncode}")
            logger.error(f"FFmpeg stderr: {stderr.decode()}")
            return None
        logger.info(f"FFmpeg transcoding successful: {len(input_bytes)} -> {len(stdout)} bytes")
        return stdout

    except ffmpeg.Error as e:
        logger.error(f"ffmpeg-python error during transcoding: {e}")
        if hasattr(e, 'stderr') and e.stderr:
            logger.error(f"ffmpeg stderr: {e.stderr.decode()}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during transcoding: {e}", exc_info=True)
        return None

# --- WebSocket Endpoint ---
@app.websocket("/ws/audio_gemini")
async def websocket_endpoint_gemini(websocket: WebSocket):
    """Handles WebSocket connections for audio streaming."""
    await websocket.accept()
    client_id = f"{USER_ID_PREFIX}{uuid.uuid4()}"
    logger.info(f"WebSocket connection accepted: {client_id}")
    adk_session_id = None
    live_session = None
    send_task = None
    receive_task = None
    audio_queue = asyncio.Queue() # Queue for raw audio bytes from client

    try:
        # --- Initial Setup ---
        if not GOOGLE_API_KEY: raise ValueError("Server not configured with GOOGLE_API_KEY.")
        if GEMINI_LIVE_CONFIG is None: raise ImportError("Required google.genai types could not be imported.")

        adk_session_id = get_or_create_adk_session_sync(client_id)
        await websocket.send_text(json.dumps({"type": "info", "message": f"ADK Session Ready: {adk_session_id}"}))

        model = genai.GenerativeModel(GEMINI_LIVE_MODEL_NAME)

        # --- Gemini Live Connection ---
        logger.info(f"[{client_id}] Initializing Gemini live session...")
        async with model.connect_live(config=GEMINI_LIVE_CONFIG) as live_session:
            logger.info(f"[{client_id}] Gemini live session established.")
            await websocket.send_text(json.dumps({"type": "info", "message": "Speech recognition active."}))

            # Task to receive audio from client, transcode, and send to Gemini
            async def send_audio_to_gemini():
                while True:
                    try:
                        webm_chunk = await audio_queue.get()
                        if webm_chunk is None: break # Sentinel

                        if ffmpeg:
                            # Transcode using ffmpeg-python helper
                            pcm_chunk = await transcode_audio_ffmpeg(webm_chunk)
                            if pcm_chunk:
                                # Send transcoded audio to Gemini Live API
                                # Need to check expected format: raw bytes or structured?
                                # Assuming raw bytes for now based on some examples
                                await live_session.send(input={"data": pcm_chunk, "mime_type": "audio/pcm"})
                                # logger.debug(f"[{client_id}] Sent {len(pcm_chunk)} PCM bytes to Gemini.")
                            else:
                                logger.warning(f"[{client_id}] Transcoding failed for chunk.")
                        else:
                            logger.error(f"[{client_id}] Cannot process audio: ffmpeg-python not available.")
                            # Maybe send error to client? Or just stop?
                        audio_queue.task_done()
                    except asyncio.CancelledError: logger.info(f"[{client_id}] send_audio task cancelled."); break
                    except Exception as e: logger.error(f"[{client_id}] Error in send_audio_to_gemini: {e}", exc_info=True); break

            # Task to receive responses from Gemini and interact with ADK/Client
            async def receive_from_gemini():
                final_transcript_buffer = ""
                try:
                    async for response in live_session:
                        if not response: continue

                        # Handle Text Response (Transcript)
                        if response.text:
                            if response.is_final:
                                final_transcript_buffer += response.text
                                logger.info(f"[{client_id}] Final Transcript: '{final_transcript_buffer}'")
                                await websocket.send_text(json.dumps({"type": "final_transcript", "transcript": final_transcript_buffer}))
                                if final_transcript_buffer.strip():
                                    await websocket.send_text(json.dumps({"type": "info", "message": "Sending to agent..."}))
                                    loop = asyncio.get_running_loop()
                                    agent_response = await loop.run_in_executor(None, run_adk_turn_sync, client_id, adk_session_id, final_transcript_buffer)
                                    await websocket.send_text(json.dumps({"type": "agent_response", "response": agent_response}))
                                final_transcript_buffer = "" # Reset buffer
                            elif STREAMING_INTERIM_RESULTS:
                                interim_text = final_transcript_buffer + response.text
                                await websocket.send_text(json.dumps({"type": "interim_transcript", "transcript": interim_text}))

                        # Handle Audio Response (TTS)
                        if response.data is not None:
                            logger.info(f"[{client_id}] Received audio data: {len(response.data)} bytes")
                            # Send raw audio bytes back to the client for playback
                            await websocket.send_bytes(response.data)

                        # Handle Errors from Gemini
                        if response.error:
                            logger.error(f"[{client_id}] Gemini live session error: {response.error}")
                            await websocket.send_text(json.dumps({"type": "error", "message": f"Speech API Error: {response.error}"}))
                            break # Stop processing on Gemini error

                except asyncio.CancelledError: logger.info(f"[{client_id}] receive_from_gemini task cancelled.")
                except Exception as e:
                    logger.error(f"[{client_id}] Error in receive_from_gemini: {e}", exc_info=True)
                    try: await websocket.send_text(json.dumps({"type": "error", "message": f"Error processing speech: {e}"}))
                    except WebSocketDisconnect: pass
                finally: logger.info(f"[{client_id}] receive_from_gemini finished.")

            # Start background tasks
            send_task = asyncio.create_task(send_audio_to_gemini())
            receive_task = asyncio.create_task(receive_from_gemini())

            # Main loop to receive data from client WebSocket
            stop_audio_received = False
            while not stop_audio_received:
                data = await websocket.receive()
                if data['type'] == 'bytes':
                    await audio_queue.put(data['bytes']) # Put raw WebM/Opus bytes onto queue
                elif data['type'] == 'text':
                    try: # Handle control messages (like stop)
                         msg = json.loads(data['text'])
                         if msg.get("type") == "control" and msg.get("action") == "stop_audio":
                              logger.info(f"[{client_id}] Received stop audio signal.")
                              stop_audio_received = True
                              # Optionally signal end of audio to Gemini if API supports it
                              # await live_session.send(input={"end_of_audio": True})
                         else: logger.warning(f"[{client_id}] Received unknown text: {data['text']}")
                    except json.JSONDecodeError: logger.warning(f"[{client_id}] Received non-JSON text: {data['text']}")
                    except Exception as e: logger.error(f"[{client_id}] Error processing text msg: {e}")

            logger.info(f"[{client_id}] Exited main WS receive loop.")

    # --- Exception Handling & Cleanup ---
    except WebSocketDisconnect: logger.info(f"WS client {client_id} disconnected.")
    except genai.types.generation_types.StopCandidateException as e:
        logger.info(f"[{client_id}] Gemini stream stopped normally: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "info", "message": "Speech stream ended."}))
        except Exception:
            pass  # Ignore errors sending closure message if WS is already closed

    except ImportError as e:
        logger.error(f"[{client_id}] Startup failed due to ImportError: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": f"Server Import Error: {e}"}))
        except Exception:
            pass

    except ValueError as e:
        logger.error(f"[{client_id}] Startup failed due to ValueError: {e}")
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": f"Server Config Error: {e}"}))
        except Exception:
            pass

    except Exception as e:
        logger.error(f"Unexpected error in WS handler {client_id}: {e}", exc_info=True)
        try:
            await websocket.send_text(json.dumps({"type": "error", "message": f"Server error: {str(e)}"}))
        except Exception:
            pass
        
    finally:
        logger.info(f"Closing WS connection & cleaning up for {client_id}.")
        # Cancel background tasks
        if send_task and not send_task.done(): send_task.cancel()
        if receive_task and not receive_task.done(): receive_task.cancel()
        # Send sentinel to audio queue processing task
        try: await audio_queue.put(None)
        except Exception: pass
        # Wait for tasks to finish cancellation
        try:
             tasks = [t for t in [send_task, receive_task] if t]
             if tasks: await asyncio.gather(*tasks, return_exceptions=True)
        except Exception as gather_err: logger.error(f"[{client_id}] Error during task cleanup: {gather_err}")
        # Clean ADK session link
        if client_id in active_adk_sessions: del active_adk_sessions[client_id]; logger.info(f"Removed ADK session link for {client_id}.")
        # Close WebSocket gracefully
        try:
            ws_state = getattr(websocket, 'client_state', None)
            if ws_state and ws_state != WebSocketState.DISCONNECTED: await websocket.close()
        except Exception as e: logger.error(f"Error closing WS for {client_id}: {e}")

# --- Uvicorn Runner ---
if __name__ == "__main__":
    # ... (remains the same) ...
    import uvicorn
    logger.info("Starting FastAPI server directly with Uvicorn...")
    if not GOOGLE_API_KEY: logger.warning("WARNING: GOOGLE_API_KEY env var not set.")
    uvicorn.run("ui.api:app", host="0.0.0.0", port=8000, reload=True)