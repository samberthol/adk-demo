# ui/api.py
import asyncio
import os
import sys
import logging
import time
import uuid
import json # Added previously
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect # Keep these
from starlette.websockets import WebSocketState # Import WebSocketState from starlette
from fastapi.responses import HTMLResponse
import google.generativeai as genai

# Use correct import path
from google.genai.types import Content, Part

# Add project root to Python path
# ... (path setup as before) ...
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import ADK components
# ... (imports as before) ...
try:
    from agents.meta.agent import meta_agent
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
except ImportError as e:
     logging.error(f"FastAPI: Failed to import agent modules or ADK components. Error: {e}")
     sys.exit(f"FastAPI startup failed: Cannot import ADK components: {e}")


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("fastapi_app")

# --- Configuration ---
# Using a distinct APP_NAME might be good practice if ADK session service were shared (it isn't here)
APP_NAME = "gcp_multi_agent_demo_api"
USER_ID_PREFIX = "fastapi_user_"
ADK_SESSION_PREFIX = f'adk_session_{APP_NAME}_'
GEMINI_LIVE_MODEL_NAME = "models/gemini-2.0-flash-live-001"
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
STREAMING_INTERIM_RESULTS = True

# --- ADK Initialization (Remains the same) ---
# ... (session_service, adk_runner, active_adk_sessions as before) ...
session_service = InMemorySessionService()
logger.info("--- FastAPI ADK Init: InMemorySessionService instantiated. ---")
adk_runner = Runner(
    agent=meta_agent,
    app_name=APP_NAME,
    session_service=session_service
)
logger.info(f"--- FastAPI ADK Init: Runner instantiated for agent '{meta_agent.name}'. ---")
active_adk_sessions = {} # Key: user_id, Value: adk_session_id

# --- Google Generative AI Configuration (Remains the same) ---
# ... (genai.configure as before) ...
if not GOOGLE_API_KEY:
     logger.error("GOOGLE_API_KEY environment variable not set. Google Generative AI cannot function.")
else:
     try:
         genai.configure(api_key=GOOGLE_API_KEY)
         logger.info("Google Generative AI configured successfully.")
     except Exception as e:
         logger.error(f"Failed to configure Google Generative AI: {e}")

# --- ADK Interaction Functions (Using Sync for simplicity, remains the same) ---
# ... (get_or_create_adk_session_sync, run_adk_turn_sync as before) ...
def get_or_create_adk_session_sync(user_id: str) -> str:
    if user_id in active_adk_sessions:
        session_id = active_adk_sessions[user_id]
        logger.info(f"--- FastAPI ADK Sync: Reusing existing ADK session for {user_id}: {session_id} ---")
        try:
            existing = session_service.get_session(app_name=APP_NAME, user_id=user_id, session_id=session_id) # Requires app_name, user_id? Check ADK docs. Assuming yes for now.
            if not existing:
                logger.warning(f"--- FastAPI ADK Sync: Session {session_id} for {user_id} not found in service. Recreating.")
                session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id, state={})
        except Exception as e:
             logger.error(f"--- FastAPI ADK Sync: Error checking/recreating session {session_id} for {user_id}: {e}")
             if user_id in active_adk_sessions: del active_adk_sessions[user_id]
             return get_or_create_adk_session_sync(user_id) # Recurse
        return session_id
    else:
        session_id = f"{ADK_SESSION_PREFIX}{int(time.time())}_{os.urandom(4).hex()}"
        active_adk_sessions[user_id] = session_id
        logger.info(f"--- FastAPI ADK Sync: Generated new ADK session for {user_id}: {session_id} ---")
        try:
            session_service.create_session(app_name=APP_NAME, user_id=user_id, session_id=session_id, state={})
            logger.info(f"--- FastAPI ADK Sync: Successfully created new session {session_id} in SessionService.")
        except Exception as e:
            logger.exception(f"--- FastAPI ADK Sync: FATAL ERROR creating session {session_id} for {user_id}:")
            if user_id in active_adk_sessions: del active_adk_sessions[user_id]
            raise
        return session_id

def run_adk_turn_sync(user_id: str, session_id: str, user_message_text: str) -> str:
    logger.info(f"--- FastAPI ADK Run Sync: Session {session_id}, User: {user_id}, Query: '{user_message_text[:100]}...' ---")
    content = Content(role='user', parts=[Part(text=user_message_text)])
    final_response_text = "[Agent did not respond]"
    try:
        for event in adk_runner.run(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts and hasattr(event.content.parts[0], 'text'):
                    final_response_text = event.content.parts[0].text
                else: final_response_text = "[Agent finished with no text output]"
                break
    except Exception as e:
        logger.exception(f"--- FastAPI ADK Run Sync: Session {session_id} - !! EXCEPTION during agent execution:")
        final_response_text = f"Sorry, an error occurred: {e}"
    logger.info(f"--- FastAPI ADK Run Sync: Session {session_id} - Turn complete. Response: '{final_response_text[:100]}...' ---")
    return final_response_text


# --- FastAPI App ---
app = FastAPI()

# Basic HTML page - remove debug logs
html = """
<!DOCTYPE html>
<html>
    <head>
        <title>FastAPI Gemini Live Audio</title>
    </head>
    <body>
        <h1>WebSocket Audio Streaming (Gemini Live & ADK Agent)</h1>
        <p>Status: <span id="status">Initializing</span></p>
        <button id="start">Start Recording</button>
        <button id="stop" disabled>Stop Recording</button>
        <h2>Agent Interaction:</h2>
        <div id="interaction" style="height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; margin-bottom: 10px;"></div>
        <script>
            // --- Cleaned Script ---
            const statusSpan = document.getElementById('status');
            const interactionDiv = document.getElementById('interaction');
            const startButton = document.getElementById('start');
            const stopButton = document.getElementById('stop');
            let websocket;
            let mediaRecorder;
            let audioChunks = [];

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
                // Use wss:// for secure connections - location.host will be this service's URL
                const wsUri = `wss://${location.host}/ws/audio_gemini`;
                logInteraction(`Attempting WebSocket connection to: ${wsUri}`, 'system');
                statusSpan.textContent = "Connecting...";
                startButton.disabled = true;

                try {
                    websocket = new WebSocket(wsUri);

                    websocket.onopen = function(evt) {
                        statusSpan.textContent = "Connected";
                        logInteraction("WebSocket Connected. Ready to record.", 'system');
                        startButton.disabled = false;
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
                        try {
                            const msg = JSON.parse(evt.data);
                            if (msg.type === 'interim_transcript') logInteraction(msg.transcript, 'interim');
                            else if (msg.type === 'final_transcript') logInteraction(msg.transcript, 'user');
                            else if (msg.type === 'agent_response') logInteraction(msg.response, 'agent');
                            else if (msg.type === 'status' || msg.type === 'info') logInteraction(msg.message, 'system');
                            else if (msg.type === 'error') logInteraction(`Error: ${msg.message}`, 'system');
                            else logInteraction(`Unknown message type: ${JSON.stringify(msg)}`, 'system');
                        } catch (e) {
                            logInteraction(`Received non-JSON message or parse error: ${evt.data}`, 'system');
                            console.error("Failed to parse message:", e);
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
                if (!websocket || websocket.readyState !== WebSocket.OPEN) {
                     logInteraction("WebSocket not open. Cannot start recording.", 'system'); return;
                }
                try {
                    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                    const options = { mimeType: 'audio/webm;codecs=opus' };
                    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                         logInteraction(`Error: Browser does not support ${options.mimeType}.`, 'system'); return;
                    }
                    mediaRecorder = new MediaRecorder(stream, options);
                    audioChunks = [];
                    mediaRecorder.ondataavailable = (event) => {
                        if (event.data.size > 0 && websocket && websocket.readyState === WebSocket.OPEN) websocket.send(event.data);
                    };
                    mediaRecorder.onstop = () => {
                        logInteraction("Recording stopped.", 'system');
                        startButton.disabled = false; stopButton.disabled = true;
                        if (websocket && websocket.readyState === WebSocket.OPEN) {
                             try { websocket.send(JSON.stringify({ "type": "control", "action": "stop_audio" })); } catch (e) { console.error("Error sending stop_audio:", e); }
                        }
                        stream.getTracks().forEach(track => track.stop());
                    };
                    mediaRecorder.start(250);
                    logInteraction("Recording started...", 'system');
                    startButton.disabled = true; stopButton.disabled = false;
                } catch (err) {
                    logInteraction("Error accessing microphone or starting recorder: " + err, 'system');
                     console.error("Mic/Recorder Error:", err);
                }
            };

            stopButton.onclick = () => {
                if (mediaRecorder && mediaRecorder.state === 'recording') mediaRecorder.stop();
            };

            // Initial connection attempt when page loads
            connectWebSocket();
            // --- END Cleaned Script ---
        </script>
    </body>
</html>
"""

@app.get("/")
async def get_test_page():
    """Serves the simple HTML page to test WebSocket audio streaming."""
    return HTMLResponse(html)


@app.websocket("/ws/audio_gemini")
async def websocket_endpoint_gemini(websocket: WebSocket):
    # ... (Websocket endpoint logic remains the same) ...
    # REMEMBER: Need to implement transcoding in send_audio_to_gemini task
    await websocket.accept()
    client_id = f"{USER_ID_PREFIX}{uuid.uuid4()}"
    logger.info(f"WebSocket (Gemini) connection accepted from {websocket.client.host}:{websocket.client.port}, assigned ID: {client_id}")
    adk_session_id = None
    if not GOOGLE_API_KEY:
        await websocket.send_text('{"type": "error", "message": "Server not configured with Google API Key."}')
        await websocket.close(code=1011); return
    try:
        adk_session_id = get_or_create_adk_session_sync(client_id)
        await websocket.send_text(f'{{"type": "info", "message": "ADK Session Ready: {adk_session_id}"}}')
    except Exception as e:
        logger.error(f"Failed to initialize ADK session for {client_id}: {e}")
        await websocket.send_text('{"type": "error", "message": "Could not initialize agent session."}')
        await websocket.close(code=1011); return

    live_session = None; send_task = None; receive_task = None
    audio_queue = asyncio.Queue()
    try:
        model = genai.GenerativeModel(GEMINI_LIVE_MODEL_NAME)
        config = genai.types.LiveConnectConfig(response_modalities=["text"])
        async with model.connect_live(config=config) as live_session:
            logger.info(f"[{client_id} / {adk_session_id}] Gemini live session established.")
            await websocket.send_text('{"type": "info", "message": "Speech recognition active."}')

            async def send_audio_to_gemini():
                while True:
                    try:
                        audio_chunk = await audio_queue.get()
                        if audio_chunk is None: break
                        logger.warning(f"[{client_id}] Skipping audio send - Transcoding needed from WebM/Opus to PCM.")
                        # TODO: Add transcoding here before sending to live_session.send(...)
                        audio_queue.task_done()
                    except asyncio.CancelledError: break
                    except Exception as e: logger.error(f"[{client_id}] Error in send_audio_to_gemini task: {e}", exc_info=True); break

            async def receive_from_gemini():
                final_transcript_buffer = ""
                try:
                    async for response in live_session:
                        if not response: continue
                        if response.text:
                            if response.is_final:
                                final_transcript_buffer += response.text
                                logger.info(f"[{client_id}] Final Transcript: '{final_transcript_buffer}'")
                                await websocket.send_text(f'{{"type": "final_transcript", "transcript": "{final_transcript_buffer}"}}')
                                if final_transcript_buffer.strip():
                                    await websocket.send_text('{"type": "info", "message": "Sending to agent..."}')
                                    loop = asyncio.get_running_loop()
                                    agent_response = await loop.run_in_executor(None, run_adk_turn_sync, client_id, adk_session_id, final_transcript_buffer)
                                    await websocket.send_text(f'{{"type": "agent_response", "response": "{agent_response}"}}')
                                else: logger.info(f"[{client_id}] Ignoring empty final transcript.")
                                final_transcript_buffer = ""
                            elif STREAMING_INTERIM_RESULTS:
                                interim_text = final_transcript_buffer + response.text
                                await websocket.send_text(f'{{"type": "interim_transcript", "transcript": "{interim_text}"}}')
                        elif response.error:
                            logger.error(f"[{client_id}] Gemini live session error: {response.error}")
                            await websocket.send_text(f'{{"type": "error", "message": "Speech API Error: {response.error}"}}'); break
                except asyncio.CancelledError: pass
                except Exception as e:
                    logger.error(f"[{client_id}] Error in receive_from_gemini task: {e}", exc_info=True)
                    try: await websocket.send_text(f'{{"type": "error", "message": "Error processing speech response: {e}"}}')
                    except WebSocketDisconnect: pass
                finally: logger.info(f"[{client_id}] Receive from Gemini task finished.")

            send_task = asyncio.create_task(send_audio_to_gemini())
            receive_task = asyncio.create_task(receive_from_gemini())
            stop_audio_received = False
            while not stop_audio_received:
                try:
                    data = await websocket.receive()
                    if data['type'] == 'bytes': await audio_queue.put(data['bytes'])
                    elif data['type'] == 'text':
                        try:
                             msg = json.loads(data['text'])
                             if msg.get("type") == "control" and msg.get("action") == "stop_audio":
                                  logger.info(f"[{client_id}] Received stop audio signal.")
                                  stop_audio_received = True
                             else: logger.warning(f"[{client_id}] Received unknown text message: {data['text']}")
                        except json.JSONDecodeError: logger.warning(f"[{client_id}] Received non-JSON text message: {data['text']}")
                        except Exception as e: logger.error(f"[{client_id}] Error processing text message: {e}")
                except WebSocketDisconnect: logger.info(f"[{client_id}] WebSocket disconnected by client."); break
                except asyncio.CancelledError: break
                except Exception as e: logger.error(f"[{client_id}] Error receiving from WebSocket: {e}", exc_info=True); break
            logger.info(f"[{client_id}] Exited main WebSocket receive loop.")
    except WebSocketDisconnect: logger.info(f"WebSocket client {client_id} disconnected during setup or run.")
    except genai.types.generation_types.StopCandidateException as e:
         logger.info(f"[{client_id}] Gemini stream stopped normally: {e}")
         try:
             await websocket.send_text('{"type": "info", "message": "Speech stream ended."}')
         except WebSocketDisconnect:
             pass # Ignore error if client disconnected before we could send message
    except Exception as e:
        logger.error(f"An unexpected error occurred in WebSocket handler for {client_id}: {e}", exc_info=True)
        try: await websocket.send_text(f'{{"type": "error", "message": "Server error: {str(e)}"}}')
        except Exception: pass
    finally:
        logger.info(f"Closing WebSocket connection and cleaning up for {client_id}.")
        if send_task and not send_task.done(): send_task.cancel()
        if receive_task and not receive_task.done(): receive_task.cancel()
        try: await audio_queue.put(None)
        except Exception: pass
        try:
             if send_task or receive_task: await asyncio.gather(send_task, receive_task, return_exceptions=True)
        except Exception as gather_err: logger.error(f"[{client_id}] Error during task gathering cleanup: {gather_err}")
        if client_id in active_adk_sessions: del active_adk_sessions[client_id]; logger.info(f"Removed ADK session association for {client_id}.")
        try:
            ws_state = getattr(websocket, 'client_state', None)
            if ws_state and ws_state != WebSocketState.DISCONNECTED: await websocket.close()
        except RuntimeError as e:
            if "WebSocket is not connected" not in str(e): logger.error(f"Error during WebSocket close for {client_id}: {e}")
        except Exception as e: logger.error(f"Unexpected error during WebSocket close for {client_id}: {e}")
 
# --- Uvicorn Runner (if run directly) ---
if __name__ == "__main__":
    import uvicorn
    logger.info("Starting FastAPI server directly with Uvicorn...")
    if not GOOGLE_API_KEY: logger.warning("WARNING: GOOGLE_API_KEY env var not set.")
    uvicorn.run("ui.api:app", host="0.0.0.0", port=8000, reload=True)