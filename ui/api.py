# ui/api.py
import asyncio
import os
import sys
import logging
import time
import uuid
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse # For a simple test page

# Import Google Generative AI library
import google.generativeai as genai
from google.genai.types import Content, Part

# Add project root to Python path
project_root = str(Path(__file__).parent.parent)
if project_root not in sys.path:
    sys.path.append(project_root)

# Import ADK components
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
APP_NAME = "gcp_multi_agent_demo_api"
USER_ID_PREFIX = "fastapi_user_"
ADK_SESSION_PREFIX = f'adk_session_{APP_NAME}_'

# Google Generative AI settings
GEMINI_LIVE_MODEL_NAME = "models/gemini-2.0-flash-live-001" # Confirmed model name
GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY') # Use general key if specific live key not needed

# Audio Configuration (Check Gemini Live docs for exact requirements)
# Common defaults, but API might specify others. Assuming PCM 16-bit 16kHz mono.
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
# The API likely expects raw PCM S16LE bytes.
# The frontend JS sends WEBM/Opus. Transcoding will be NEEDED.
# Adding a dependency like ffmpeg-python or using a simpler approach if possible.
# For now, we'll log a warning and skip sending audio if transcoding isn't implemented.
# TODO: Implement transcoding from WEBM/Opus to PCM S16LE.

STREAMING_INTERIM_RESULTS = True # Gemini Live supports interim results

# --- ADK Initialization ---
session_service = InMemorySessionService()
logger.info("--- FastAPI ADK Init: InMemorySessionService instantiated. ---")
adk_runner = Runner(
    agent=meta_agent,
    app_name=APP_NAME,
    session_service=session_service
)
logger.info(f"--- FastAPI ADK Init: Runner instantiated for agent '{meta_agent.name}'. ---")
active_adk_sessions = {} # Key: user_id, Value: adk_session_id

# --- Google Generative AI Configuration ---
if not GOOGLE_API_KEY:
    logger.error("GOOGLE_API_KEY environment variable not set. Google Generative AI cannot function.")
    # Optionally raise an error or exit if the key is essential at startup
else:
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Google Generative AI configured successfully.")
    except Exception as e:
        logger.error(f"Failed to configure Google Generative AI: {e}")
        # Handle configuration error

# --- ADK Interaction Functions ---
# NOTE: These ADK functions run the *synchronous* ADK methods.
# For better performance in FastAPI, consider adapting them to use
# session_service.get_session_async, create_session_async, and runner.run_async
# This requires careful handling of event loops between FastAPI and ADK.
# Using sync methods for now for simplicity, but be aware of potential blocking.

def get_or_create_adk_session_sync(user_id: str) -> str:
    """Gets existing or creates a new ADK session ID for a FastAPI user (Sync)."""
    if user_id in active_adk_sessions:
        session_id = active_adk_sessions[user_id]
        logger.info(f"--- FastAPI ADK Sync: Reusing existing ADK session for {user_id}: {session_id} ---")
        try:
            # Using synchronous ADK methods
            existing = session_service.get_session(APP_NAME, user_id, session_id)
            if not existing:
                logger.warning(f"--- FastAPI ADK Sync: Session {session_id} for {user_id} not found in service. Recreating.")
                session_service.create_session(APP_NAME, user_id, session_id, {})
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
            session_service.create_session(APP_NAME, user_id, session_id, {})
            logger.info(f"--- FastAPI ADK Sync: Successfully created new session {session_id} in SessionService.")
        except Exception as e:
            logger.exception(f"--- FastAPI ADK Sync: FATAL ERROR creating session {session_id} for {user_id}:")
            if user_id in active_adk_sessions: del active_adk_sessions[user_id]
            raise
        return session_id

def run_adk_turn_sync(user_id: str, session_id: str, user_message_text: str) -> str:
    """Runs a single turn with the ADK agent (Sync)."""
    logger.info(f"--- FastAPI ADK Run Sync: Session {session_id}, User: {user_id}, Query: '{user_message_text[:100]}...' ---")
    content = Content(role='user', parts=[Part(text=user_message_text)])
    final_response_text = "[Agent did not respond]"
    try:
        # Using synchronous runner method
        for event in adk_runner.run(user_id=user_id, session_id=session_id, new_message=content):
            if event.is_final_response():
                if event.content and event.content.parts and hasattr(event.content.parts[0], 'text'):
                    final_response_text = event.content.parts[0].text
                    logger.info(f"--- FastAPI ADK Run Sync: Session {session_id} - Final response received.")
                else:
                    final_response_text = "[Agent finished with no text output]"
                    logger.warning(f"--- FastAPI ADK Run Sync: Session {session_id} - Final event has no text content.")
                break
    except Exception as e:
        logger.exception(f"--- FastAPI ADK Run Sync: Session {session_id} - !! EXCEPTION during agent execution:")
        final_response_text = f"Sorry, an error occurred: {e}"

    logger.info(f"--- FastAPI ADK Run Sync: Session {session_id} - Turn complete. Response: '{final_response_text[:100]}...' ---")
    return final_response_text


# --- FastAPI App ---
app = FastAPI()

# Basic HTML page for testing
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
            // --- ADDED CONSOLE LOGGING ---
            console.log("SCRIPT START: Script block executing."); // <-- Log 1

            const statusSpan = document.getElementById('status');
            const interactionDiv = document.getElementById('interaction');
            const startButton = document.getElementById('start');
            const stopButton = document.getElementById('stop');
            let websocket;
            let mediaRecorder;
            let audioChunks = [];

            function logInteraction(message, type = 'info') {
                // ... (logInteraction function) ...
                const p = document.createElement('p');
                let prefix = '';
                if (type === 'user') prefix = '<strong>You:</strong> ';
                else if (type === 'agent') prefix = '<strong>Agent:</strong> ';
                else if (type === 'system') prefix = '<em>System:</em> ';
                else if (type === 'interim') prefix = '<em>You (interim):</em> ';
                p.innerHTML = prefix + message;
                interactionDiv.appendChild(p);
                interactionDiv.scrollTop = interactionDiv.scrollHeight;
                console.log(`${type}: ${message}`); // Also logs here
            }

            function connectWebSocket() {
                console.log("SCRIPT TRACE: Entering connectWebSocket function."); // <-- Log 2
                const wsUri = `wss://${location.host}/ws/audio_gemini`;
                logInteraction(`Attempting WebSocket connection to: ${wsUri}`, 'system');
                statusSpan.textContent = "Connecting...";
                startButton.disabled = true;

                try {
                    console.log("SCRIPT TRACE: Attempting 'new WebSocket()'."); // <-- Log 3
                    websocket = new WebSocket(wsUri);
                    console.log("SCRIPT TRACE: 'new WebSocket()' called."); // <-- Log 4

                    websocket.onopen = function(evt) {
                        console.log("SCRIPT TRACE: websocket.onopen fired."); // <-- Log 5
                        statusSpan.textContent = "Connected";
                        logInteraction("WebSocket Connected. Ready to record.", 'system');
                        startButton.disabled = false;
                    };
                    websocket.onclose = function(evt) {
                        console.error("SCRIPT TRACE: websocket.onclose fired.", evt); // <-- Log errors
                        statusSpan.textContent = "Disconnected";
                        logInteraction(`WebSocket Disconnected: Code=<span class="math-inline">\{evt\.code\}, Reason\=</span>{evt.reason || 'N/A'}, WasClean=${evt.wasClean}`, 'system');
                        startButton.disabled = true; stopButton.disabled = true;
                        if (mediaRecorder && mediaRecorder.state !== 'inactive') mediaRecorder.stop();
                    };
                    websocket.onerror = function(evt) {
                        console.error("SCRIPT TRACE: websocket.onerror fired.", evt); // <-- Log errors
                        statusSpan.textContent = "Error";
                        logInteraction('WebSocket Error occurred. See browser console.', 'system');
                        startButton.disabled = true; stopButton.disabled = true;
                    };
                    websocket.onmessage = function(evt) {
                       // ... (onmessage handling) ...
                    };
                } catch (err) {
                     console.error("SCRIPT TRACE: Error caught during WebSocket creation.", err); // <-- Log errors
                     statusSpan.textContent = "Error";
                     logInteraction(`Error creating WebSocket: ${err}`, 'system');
                     startButton.disabled = true; stopButton.disabled = true;
                }
            }

            startButton.onclick = async () => {
                console.log("SCRIPT TRACE: startButton.onclick fired."); // <-- Log button click
                // ... (onclick handler) ...
            };

            stopButton.onclick = () => {
                console.log("SCRIPT TRACE: stopButton.onclick fired."); // <-- Log button click
                // ... (stopButton onclick) ...
            };

            // Initial connection attempt when page loads
            console.log("SCRIPT TRACE: Calling connectWebSocket() now."); // <-- Log 6
            connectWebSocket();
            console.log("SCRIPT TRACE: connectWebSocket() called."); // <-- Log 7
            // --- END ADDED CONSOLE LOGGING ---
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
    """Handles WebSocket connections for audio streaming using Gemini Live API and ADK Agent."""
    await websocket.accept()
    client_id = f"{USER_ID_PREFIX}{uuid.uuid4()}"
    logger.info(f"WebSocket (Gemini) connection accepted from {websocket.client.host}:{websocket.client.port}, assigned ID: {client_id}")

    adk_session_id = None
    if not GOOGLE_API_KEY:
        await websocket.send_text('{"type": "error", "message": "Server not configured with Google API Key."}')
        await websocket.close(code=1011)
        return

    try:
        # Using the synchronous ADK function for now
        adk_session_id = get_or_create_adk_session_sync(client_id)
        await websocket.send_text(f'{{"type": "info", "message": "ADK Session Ready: {adk_session_id}"}}')
    except Exception as e:
        logger.error(f"Failed to initialize ADK session for {client_id}: {e}")
        await websocket.send_text('{"type": "error", "message": "Could not initialize agent session."}')
        await websocket.close(code=1011)
        return

    # --- Gemini Live API Interaction ---
    live_session = None
    send_task = None
    receive_task = None
    audio_queue = asyncio.Queue() # Queue to hold audio bytes from WebSocket

    try:
        # Initialize the specific Gemini model for live content
        model = genai.GenerativeModel(GEMINI_LIVE_MODEL_NAME)

        # Configuration for the LiveConnect session
        # Requesting audio output might not be necessary if only interacting via ADK text
        config = genai.types.LiveConnectConfig(
             response_modalities=["text"], # Only need text transcript
             # Add speech config if needed, e.g., for language hints, but defaults are often sufficient
             # speech_config=genai.types.SpeechConfig(...)
        )

        # Establish the connection
        logger.info(f"[{client_id} / {adk_session_id}] Initializing Gemini live session...")
        # Use the async context manager for clean setup/teardown
        async with model.connect_live(config=config) as live_session:
            logger.info(f"[{client_id} / {adk_session_id}] Gemini live session established.")
            await websocket.send_text('{"type": "info", "message": "Speech recognition active."}')

            # --- Task to send audio from WebSocket to Gemini ---
            async def send_audio_to_gemini():
                while True:
                    try:
                        audio_chunk = await audio_queue.get()
                        if audio_chunk is None: # Sentinel to stop
                            break
                        # IMPORTANT: Transcoding needed here!
                        # The chunk from the queue is WEBM/Opus. Gemini needs PCM.
                        # For now, we skip sending until transcoding is added.
                        logger.warning(f"[{client_id}] Skipping audio send - Transcoding needed from WebM/Opus to PCM.")
                        # TODO: Add transcoding (e.g., using ffmpeg-python)
                        # Example placeholder for sending *IF* it was PCM:
                        # await live_session.send(input={"data": pcm_chunk, "mime_type": "audio/pcm"})
                        audio_queue.task_done()
                    except asyncio.CancelledError:
                        logger.info(f"[{client_id}] Send audio task cancelled.")
                        break
                    except Exception as e:
                        logger.error(f"[{client_id}] Error in send_audio_to_gemini task: {e}", exc_info=True)
                        break # Exit on error

            # --- Task to receive responses from Gemini and interact with ADK ---
            async def receive_from_gemini():
                final_transcript_buffer = ""
                try:
                    async for response in live_session:
                        if not response: continue

                        if response.text:
                            #logger.debug(f"[{client_id}] Received text: '{response.text}', is_final={response.is_final}")
                            if response.is_final:
                                final_transcript_buffer += response.text # Append final part
                                logger.info(f"[{client_id}] Final Transcript: '{final_transcript_buffer}'")
                                await websocket.send_text(f'{{"type": "final_transcript", "transcript": "{final_transcript_buffer}"}}')

                                # Send final transcript to ADK agent (using sync method for now)
                                if final_transcript_buffer.strip():
                                    await websocket.send_text('{"type": "info", "message": "Sending to agent..."}')
                                    # Run sync ADK call in executor to avoid blocking FastAPI loop
                                    loop = asyncio.get_running_loop()
                                    agent_response = await loop.run_in_executor(
                                        None, # Use default executor
                                        run_adk_turn_sync, # Sync function
                                        client_id, adk_session_id, final_transcript_buffer # Args
                                    )
                                    await websocket.send_text(f'{{"type": "agent_response", "response": "{agent_response}"}}')
                                else:
                                     logger.info(f"[{client_id}] Ignoring empty final transcript for ADK.")

                                final_transcript_buffer = "" # Reset buffer after processing
                            elif STREAMING_INTERIM_RESULTS:
                                interim_text = final_transcript_buffer + response.text # Show context
                                await websocket.send_text(f'{{"type": "interim_transcript", "transcript": "{interim_text}"}}')
                        elif response.error:
                            logger.error(f"[{client_id}] Gemini live session error: {response.error}")
                            await websocket.send_text(f'{{"type": "error", "message": "Speech API Error: {response.error}"}}')
                            break # Exit on session error
                except asyncio.CancelledError:
                    logger.info(f"[{client_id}] Receive from Gemini task cancelled.")
                except Exception as e:
                    logger.error(f"[{client_id}] Error in receive_from_gemini task: {e}", exc_info=True)
                    try:
                        await websocket.send_text(f'{{"type": "error", "message": "Error processing speech response: {e}"}}')
                    except WebSocketDisconnect: pass
                finally:
                     logger.info(f"[{client_id}] Receive from Gemini task finished.")


            # Start sender and receiver tasks
            send_task = asyncio.create_task(send_audio_to_gemini())
            receive_task = asyncio.create_task(receive_from_gemini())

            # --- Main loop to receive data from WebSocket client ---
            stop_audio_received = False
            while not stop_audio_received:
                try:
                    data = await websocket.receive()
                    if data['type'] == 'bytes':
                        # Assume this is audio data (WebM/Opus)
                        await audio_queue.put(data['bytes'])
                    elif data['type'] == 'text':
                        # Check for control messages (like stopping audio)
                        try:
                             msg = json.loads(data['text'])
                             if msg.get("type") == "control" and msg.get("action") == "stop_audio":
                                  logger.info(f"[{client_id}] Received stop audio signal from client.")
                                  stop_audio_received = True
                                  # Signal end of audio to Gemini if session supports it
                                  # await live_session.send(input={"end_of_audio": True}) # Check API doc for exact mechanism
                             else:
                                 logger.warning(f"[{client_id}] Received unknown text message: {data['text']}")
                        except json.JSONDecodeError:
                             logger.warning(f"[{client_id}] Received non-JSON text message: {data['text']}")
                        except Exception as e:
                             logger.error(f"[{client_id}] Error processing text message: {e}")

                except WebSocketDisconnect:
                    logger.info(f"[{client_id}] WebSocket disconnected by client.")
                    break # Exit main loop
                except asyncio.CancelledError:
                     logger.info(f"[{client_id}] Main WebSocket receive loop cancelled.")
                     break
                except Exception as e:
                    logger.error(f"[{client_id}] Error receiving from WebSocket: {e}", exc_info=True)
                    break # Exit on error

            logger.info(f"[{client_id}] Exited main WebSocket receive loop.")

    except WebSocketDisconnect:
        logger.info(f"WebSocket client {client_id} disconnected during setup or run.")
    except genai.types.generation_types.StopCandidateException as e:
         logger.info(f"[{client_id}] Gemini stream stopped normally: {e}")
         try: await websocket.send_text('{"type": "info", "message": "Speech stream ended."}')
         except WebSocketDisconnect: pass
    except Exception as e:
        logger.error(f"An unexpected error occurred in WebSocket handler for {client_id}: {e}", exc_info=True)
        try:
            # Ensure message is JSON formatted
            await websocket.send_text(f'{{"type": "error", "message": "Server error: {str(e)}"}}')
        except WebSocketDisconnect: pass
        except Exception as send_err:
             logger.error(f"[{client_id}] Error sending final error message: {send_err}")
    finally:
        logger.info(f"Closing WebSocket connection and cleaning up for {client_id}.")
        # Clean up tasks and queues
        if send_task and not send_task.done(): send_task.cancel()
        if receive_task and not receive_task.done(): receive_task.cancel()
        # Send sentinel to unblock queue getter if send_task is waiting
        try: await audio_queue.put(None)
        except Exception: pass

        # Wait briefly for tasks to cancel (optional)
        try:
             if send_task or receive_task:
                  await asyncio.gather(send_task, receive_task, return_exceptions=True)
        except Exception as gather_err:
             logger.error(f"[{client_id}] Error during task gathering cleanup: {gather_err}")

        # Close live session if it was created and context manager didn't handle it (it should)
        # if live_session and hasattr(live_session, 'close'): # Check needed? Context manager should close.
        #    try: await live_session.close()
        #    except Exception as close_err: logger.error(f"[{client_id}] Error closing live session: {close_err}")

        if client_id in active_adk_sessions:
            del active_adk_sessions[client_id]
            logger.info(f"Removed ADK session association for {client_id}.")
        try:
            if websocket.client_state != WebSocketState.DISCONNECTED:
                 await websocket.close()
        except RuntimeError as e:
            # Ignore errors if websocket is already closed
            if "WebSocket is not connected" in str(e): pass
            else: logger.error(f"Error during WebSocket close for {client_id}: {e}")
        except Exception as e:
             logger.error(f"Unexpected error during WebSocket close for {client_id}: {e}")


# --- Uvicorn Runner (if run directly) ---
if __name__ == "__main__":
    import uvicorn
    # Import WebSocketState for the finally block check
    from starlette.websockets import WebSocketState

    logger.info("Starting FastAPI server directly with Uvicorn...")
    if not GOOGLE_API_KEY:
        logger.warning("WARNING: GOOGLE_API_KEY environment variable is not set. Gemini API calls will fail.")
    # Enable reload for development, disable for production deployment
    uvicorn.run("ui.api:app", host="0.0.0.0", port=8000, reload=True) # Host 0.0.0.0 makes it accessible externally