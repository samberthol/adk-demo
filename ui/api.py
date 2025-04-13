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

try:
    # Import types needed for configuration, content, and ACTIVITY SIGNALS
    from google.genai.types import (
        Content, Part, LiveConnectConfig, SpeechConfig, VoiceConfig,
        PrebuiltVoiceConfig, RealtimeInputConfig, AutomaticActivityDetection, # Added RealtimeInputConfig, AutomaticActivityDetection
        LiveClientMessage, ActivityStart, ActivityEnd # Added LiveClientMessage, ActivityStart, ActivityEnd
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
GEMINI_LIVE_MODEL_NAME = "models/gemini-2.0-flash-live-001" # Ensure this model supports Live API
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

# --- Frontend HTML/JS with Audio Playback AND Manual Activity Signals ---
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
            let audioChunks = [];

            // --- Audio Playback Setup (No changes needed here) ---
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
                const pcm16Data = new Int16Array(arrayBuffer);
                const float32Data = new Float32Array(pcm16Data.length);
                for (let i = 0; i < pcm16Data.length; i++) { float32Data[i] = pcm16Data[i] / 32768.0; }
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
                nextStartTime = startTime + bufferToPlay.duration;
                source.onended = () => { isPlaying = false; schedulePlayback(); };
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
                const wsUri = `wss://${location.host}/ws/audio_gemini`; // Use wss for secure connection
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
                            playAudioChunk(evt.data);
                        } else if (typeof evt.data === 'string') {
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
                if (!audioContext) { if (!initAudioContext()) return; }
                if (audioContext.state === 'suspended') { await audioContext.resume(); }

                if (!websocket || websocket.readyState !== WebSocket.OPEN) {
                     logInteraction("WebSocket not open. Cannot start recording.", 'system'); return;
                }
                // --- MODIFICATION: Send start_activity signal ---
                sendControlMessage("start_activity");
                // --- End Modification ---

                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    const options = { mimeType: 'audio/webm;codecs=opus' };
                    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                         logInteraction(`Error: Browser does not support ${options.mimeType}.`, 'system');
                         // --- MODIFICATION: Send end_activity signal if start failed after sending start_activity ---
                         sendControlMessage("end_activity");
                         // --- End Modification ---
                         return;
                    }
                    mediaRecorder = new MediaRecorder(stream, options);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0 && websocket && websocket.readyState === WebSocket.OPEN) {
                            websocket.send(event.data); // Send raw WebM audio
                        }
                    };
                    mediaRecorder.onstop = () => {
                        logInteraction("Recording stopped.", 'system');
                        startButton.disabled = false; stopButton.disabled = true;
                        // --- MODIFICATION: Send end_activity signal ---
                        sendControlMessage("end_activity");
                        // --- End Modification ---
                        stream.getTracks().forEach(track => track.stop());
                    };
                    mediaRecorder.start(250); // Send chunks every 250ms
                    logInteraction("Recording started...", 'system');
                    startButton.disabled = true; stopButton.disabled = false;
                } catch (err) {
                    logInteraction("Error accessing microphone or starting recorder: " + err, 'system');
                    console.error("Mic/Recorder Error:", err);
                     // --- MODIFICATION: Send end_activity signal if start failed after sending start_activity ---
                    sendControlMessage("end_activity");
                    // --- End Modification ---
                }
            };

            stopButton.onclick = () => {
                if (mediaRecorder && mediaRecorder.state === 'recording') {
                    mediaRecorder.stop(); // This triggers onstop handler above which now sends end_activity
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
    if not ffmpeg: logger.error("ffmpeg-python library not available."); return None
    try:
        process = (
            ffmpeg
            .input('pipe:0')
            .output('pipe:1', format=TARGET_FORMAT, acodec='pcm_s16le', ac=TARGET_CHANNELS, ar=TARGET_SAMPLE_RATE)
            .run_async(pipe_stdin=True, pipe_stdout=True, pipe_stderr=True, quiet=True)
        )
        stdout, stderr = await asyncio.to_thread(process.communicate, input=input_bytes)
        if process.returncode != 0:
            logger.error(f"FFmpeg failed: {process.returncode}\n{stderr.decode()}")
            return None
        # logger.info(f"FFmpeg transcoding successful: {len(input_bytes)} -> {len(stdout)} bytes") # Log might be too verbose
        return stdout
    except ffmpeg.Error as e:
        logger.error(f"ffmpeg-python error: {e}\n{getattr(e, 'stderr', b'').decode()}", exc_info=True)
        return None
    except Exception as e:
        logger.error(f"Unexpected transcoding error: {e}", exc_info=True)
        return None

# --- WebSocket Endpoint ---
@app.websocket("/ws/audio_gemini")
async def websocket_endpoint_gemini(websocket: WebSocket):
    """Handles WebSocket connections for audio streaming using client.aio.live."""
    await websocket.accept()
    client_id = f"{USER_ID_PREFIX}{uuid.uuid4()}"
    logger.info(f"WebSocket connection accepted: {client_id}")
    adk_session_id = None
    live_session = None
    send_task = None
    receive_task = None
    audio_queue = asyncio.Queue()

    try:
        if not GOOGLE_API_KEY: raise ValueError("Server not configured with GOOGLE_API_KEY.")
        if GEMINI_LIVE_CONFIG is None: raise ImportError("Required google.genai types could not be imported.")

        adk_session_id = get_or_create_adk_session_sync(client_id)
        await websocket.send_text(json.dumps({"type": "info", "message": f"ADK Session Ready: {adk_session_id}"}))

        client = genai.Client()
        logger.info(f"[{client_id}] Initializing Gemini live session...")

        async with client.aio.live.connect(
            model=GEMINI_LIVE_MODEL_NAME,
            config=GEMINI_LIVE_CONFIG # Config now includes disabled auto activity detection
        ) as live_session:
            logger.info(f"[{client_id}] Gemini live session established.")
            await websocket.send_text(json.dumps({"type": "info", "message": "Manual activity detection active."}))

            # Task to receive audio, transcode, and send to Gemini
            async def send_audio_to_gemini():
                while True:
                    try:
                        webm_chunk = await audio_queue.get()
                        if webm_chunk is None: break
                        if ffmpeg:
                            pcm_chunk = await transcode_audio_ffmpeg(webm_chunk)
                            if pcm_chunk:
                                # Send audio using LiveClientMessage
                                await live_session.send(LiveClientMessage(
                                    realtime_input=types.LiveClientRealtimeInput(
                                        media_chunks=[types.Blob(data=pcm_chunk, mime_type="audio/pcm")]
                                    )
                                ))
                            else: logger.warning(f"[{client_id}] Transcoding failed.")
                        else: logger.error(f"[{client_id}] Cannot process audio: ffmpeg not available.")
                        audio_queue.task_done()
                    except asyncio.CancelledError: logger.info(f"[{client_id}] send_audio task cancelled."); break
                    except Exception as e: logger.error(f"[{client_id}] Error in send_audio_to_gemini: {e}", exc_info=True); break

            # Task to receive from Gemini and interact with ADK/Client
            async def receive_from_gemini():
                final_transcript_buffer = ""
                try:
                    async for response in live_session:
                        if not response: continue
                        # Handle Text Response (Transcript)
                        if response.server_content and response.server_content.input_transcription:
                            transcript_data = response.server_content.input_transcription
                            if transcript_data.finished:
                                final_transcript_buffer += transcript_data.text
                                logger.info(f"[{client_id}] Final Transcript: '{final_transcript_buffer}'")
                                await websocket.send_text(json.dumps({"type": "final_transcript", "transcript": final_transcript_buffer}))
                                if final_transcript_buffer.strip():
                                     await websocket.send_text(json.dumps({"type": "info", "message": "Sending to agent..."}))
                                     loop = asyncio.get_running_loop()
                                     agent_response = await loop.run_in_executor(None, run_adk_turn_sync, client_id, adk_session_id, final_transcript_buffer)
                                     await websocket.send_text(json.dumps({"type": "agent_response", "response": agent_response}))
                                final_transcript_buffer = "" # Reset buffer
                            elif STREAMING_INTERIM_RESULTS and transcript_data.text:
                                interim_text = final_transcript_buffer + transcript_data.text
                                await websocket.send_text(json.dumps({"type": "interim_transcript", "transcript": interim_text}))

                        # Handle Audio Response (TTS)
                        if response.server_content and response.server_content.model_turn:
                             for part in response.server_content.model_turn.parts:
                                 if part.inline_data and part.inline_data.data:
                                     logger.info(f"[{client_id}] Received audio data: {len(part.inline_data.data)} bytes")
                                     await websocket.send_bytes(part.inline_data.data) # Send raw audio bytes

                        # Handle Errors from Gemini (check if error structure is different in Live API)
                        if response.error: # Assuming error is directly on response, adjust if needed
                            logger.error(f"[{client_id}] Gemini live session error: {response.error}")
                            await websocket.send_text(json.dumps({"type": "error", "message": f"Speech API Error: {response.error}"}))
                            break

                except asyncio.CancelledError: logger.info(f"[{client_id}] receive_from_gemini task cancelled.")
                except Exception as e:
                    logger.error(f"[{client_id}] Error in receive_from_gemini: {e}", exc_info=True)
                    try: await websocket.send_text(json.dumps({"type": "error", "message": f"Error processing speech: {e}"}))
                    except WebSocketDisconnect: pass
                finally: logger.info(f"[{client_id}] receive_from_gemini finished.")


            # Start background tasks
            send_task = asyncio.create_task(send_audio_to_gemini())
            receive_task = asyncio.create_task(receive_from_gemini())

            # Main loop to receive data/signals from client WebSocket
            while True: # Loop until disconnect or explicit break
                data = await websocket.receive()

                if data['type'] == 'websocket.disconnect':
                    logger.info(f"[{client_id}] WebSocket disconnect message received.")
                    break # Exit loop on disconnect

                if data['type'] == 'bytes':
                    await audio_queue.put(data['bytes']) # Queue raw WebM audio for transcoding
                elif data['type'] == 'text':
                    try:
                         msg = json.loads(data['text'])
                         if msg.get("type") == "control":
                              action = msg.get("action")
                              if action == "start_activity":
                                   logger.info(f"[{client_id}] Received start_activity signal.")
                                   # Send ActivityStart signal to Gemini
                                   await live_session.send(LiveClientMessage(activity_start=ActivityStart()))
                              elif action == "end_activity":
                                   logger.info(f"[{client_id}] Received end_activity signal.")
                                   # Send ActivityEnd signal to Gemini
                                   await live_session.send(LiveClientMessage(activity_end=ActivityEnd()))
                                   # Consider if this should also signal end of audio queue? Maybe not needed.
                              else:
                                   logger.warning(f"[{client_id}] Received unknown control action: {action}")
                         else:
                              logger.warning(f"[{client_id}] Received unknown text message structure: {data['text']}")
                    except json.JSONDecodeError:
                         logger.warning(f"[{client_id}] Received non-JSON text: {data['text']}")
                    except Exception as e:
                         logger.error(f"[{client_id}] Error processing text msg: {e}")

            logger.info(f"[{client_id}] Exited main WS receive loop.")

    # --- Exception Handling & Cleanup ---
    except WebSocketDisconnect: logger.info(f"WS client {client_id} disconnected.")
    except genai.types.generation_types.StopCandidateException as e:
        logger.info(f"[{client_id}] Gemini stream stopped normally: {e}")
        try: await websocket.send_text(json.dumps({"type": "info", "message": "Speech stream ended."}))
        except Exception: pass
    except ImportError as e:
        logger.error(f"[{client_id}] Startup failed due to ImportError: {e}")
        try: await websocket.send_text(json.dumps({"type": "error", "message": f"Server Import Error: {e}"}))
        except Exception: pass
    except ValueError as e:
        logger.error(f"[{client_id}] Startup failed due to ValueError: {e}")
        try: await websocket.send_text(json.dumps({"type": "error", "message": f"Server Config Error: {e}"}))
        except Exception: pass
    except Exception as e:
        logger.error(f"Unexpected error in WS handler {client_id}: {e}", exc_info=True)
        try: await websocket.send_text(json.dumps({"type": "error", "message": f"Server error: {str(e)}"}))
        except Exception: pass
    finally:
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
            if ws_state and ws_state != WebSocketState.DISCONNECTED: await websocket.close()
        except Exception as e: logger.error(f"Error closing WS for {client_id}: {e}")

# --- Uvicorn Runner ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server directly with Uvicorn...")
    if not GOOGLE_API_KEY: logger.warning("WARNING: GOOGLE_API_KEY env var not set.")
    # Ensure ffmpeg availability check if critical
    if ffmpeg is None:
        logger.warning("WARNING: ffmpeg-python not found, audio transcoding will fail.")
    uvicorn.run("ui.api:app", host="0.0.0.0", port=8000, reload=True)