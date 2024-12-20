import asyncio
import json
import os
import uuid
from dweam.models import ParamsUpdate
from pydantic import ValidationError
import torch
from typing_extensions import assert_never
from time import time
import hmac
import hashlib
import base64
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Any

from structlog.stdlib import BoundLogger
from fastapi import Depends, FastAPI, HTTPException, Query, Request, Path
from fastapi.responses import JSONResponse
from aiortc import RTCPeerConnection, RTCSessionDescription, VideoStreamTrack, RTCIceServer, RTCConfiguration
import numpy as np
from av.video.frame import VideoFrame
from av.frame import Frame
import pygame
from fastapi.staticfiles import StaticFiles
import yaml
from dweam.constants import JS_TO_PYGAME_KEY_MAP, JS_TO_PYGAME_BUTTON_MAP
from fastapi.middleware.cors import CORSMiddleware
from dweam.game import Game, GameInfo
from dweam.log_config import get_logger
from dweam.utils.entrypoint import load_games
from contextlib import asynccontextmanager

from dweam.utils.turn import create_turn_credentials, get_turn_stun_urls

log = get_logger()
pcs = set()

# load game entrypoints
games = load_games(log)

def logger_dependency() -> BoundLogger:
    global log
    return log


def is_local_only() -> bool:
    local_var = os.environ.get('LOCAL_ONLY', '')
    if local_var == "0":
        return False
    return bool(local_var)


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

    # Close peer connections on shutdown
    coros = [pc.close() for pc in pcs]
    await asyncio.gather(*coros)
    pcs.clear()

app = FastAPI(lifespan=lifespan)

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:4321",  # Development
        "http://localhost",       # Production
        # Add other allowed origins as needed
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Video stream track to capture Pygame output
class GameVideoTrack(VideoStreamTrack):
    """
    A video stream track that captures frames from a Pygame application.
    """
    def __init__(self, game: Game):
        super().__init__()
        self.game = game

    async def recv(self) -> Frame:
        # Control frame rate
        # TODO don't hardcode 30 FPS
        await asyncio.sleep(1 / 30)  # 30 FPS

        # Capture the frame from Pygame
        # TODO allow numpy arrays/PIL images directly
        surface = await self.game.get_next_frame()
        frame = pygame.surfarray.array3d(surface)
        frame = np.fliplr(frame)
        frame = np.rot90(frame)
        new_frame = VideoFrame.from_ndarray(frame, format='rgb24')

        # Assign timestamp
        new_frame.pts, new_frame.time_base = await self.next_timestamp()
        return new_frame


# Endpoint to serve the entire games list
@app.get('/game_info')
async def get_games() -> list[GameInfo]:
    return list(game 
                for game_list in games.values() 
                for game in game_list.values())

# Endpoint to serve the entire games list
@app.get('/game_info/{type}')
async def get_games_by_type(type: str) -> list[GameInfo]:
    if type not in games:
        raise HTTPException(status_code=404, detail="Game type not found")
    return list(games[type].values())


# Endpoint to serve a singular game based on query parameter
@app.get('/game_info/{type}/{id}')
async def get_game(type: str, id: str) -> GameInfo:
    if type not in games:
        raise HTTPException(status_code=404, detail="Game type not found")
    if id not in games[type]:
        raise HTTPException(status_code=404, detail="Game not found")
    return games[type][id]


@dataclass
class GameHeartbeat:
    """Tracks when a game was last active"""
    game: Game
    last_heartbeat: datetime
    peer_connection: RTCPeerConnection
    cleanup_scheduled: bool = False

    @property
    def is_stale(self) -> bool:
        """Check if the session hasn't received a heartbeat recently"""
        return datetime.now() - self.last_heartbeat > timedelta(seconds=5)

# Global session management
active_games: dict[str, GameHeartbeat] = {}

async def cleanup_game(session_id: str, log: BoundLogger) -> None:
    """Clean up a game and its resources"""
    if session_id not in active_games:
        log.warning("Received cleanup request for unknown session", 
                    session_id=session_id)
        return

    heartbeat = active_games[session_id]
    if heartbeat.cleanup_scheduled:
        return
    
    heartbeat.cleanup_scheduled = True
    log.info("Cleaning up game", session_id=session_id)
    
    try:
        heartbeat.game.stop()  # Uses Game's built-in stop method
        if heartbeat.peer_connection.connectionState != "closed":
            await asyncio.wait_for(heartbeat.peer_connection.close(), timeout=3.0)
    except Exception as e:
        log.error("Error during game cleanup", 
                 session_id=session_id, 
                 error=str(e))
    finally:
        active_games.pop(session_id, None)
        pcs.discard(heartbeat.peer_connection)
        log.debug("Removed peer connection", session_id=session_id)

# WebRTC server endpoint
@app.post("/offer/{type}/{id}")
async def offer(
    request: Request,
    type: str = Path(...),
    id: str = Path(...),
    log: BoundLogger = Depends(logger_dependency),
    local_only: bool = Depends(is_local_only),
):
    if type not in games:
        raise HTTPException(status_code=404, detail="Game type not found")
    if id not in games[type]:
        raise HTTPException(status_code=404, detail="Game not found")
    game_info = games[type][id]
    if game_info._implementation is None:
        log.error("Game implementation not found", type=type, id=id)
        raise HTTPException(status_code=500, detail="Game implementation not found")

    params = await request.json()
    offer = RTCSessionDescription(sdp=params["sdp"], type=params["type"])

    if local_only:
        log.info("Running in local mode without TURN/STUN servers")
        ice_servers = []  # Empty list for direct connections
    else:
        turn_secret = os.environ.get('TURN_SECRET_KEY')
        if turn_secret is None:
            log.error("Running in server mode but TURN_SECRET_KEY is not set")
            raise HTTPException(status_code=500, detail="TURN server configuration missing")

        turn_credentials = create_turn_credentials(turn_secret)
        turn_url, stun_url = get_turn_stun_urls()

        ice_servers = [
            RTCIceServer(
                urls=[stun_url],
                username=turn_credentials["username"],
                credential=turn_credentials["credential"]
            ),
            RTCIceServer(
                urls=[turn_url],
                username=turn_credentials["username"],
                credential=turn_credentials["credential"]
            )
        ]
        log.info("Running in server mode with TURN/STUN servers", turn_url=turn_url)

    config = RTCConfiguration(iceServers=ice_servers)
    pc = RTCPeerConnection(configuration=config)
    session_id = str(uuid.uuid4())[:8]
    log = log.bind(session_id=session_id)

    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps" if torch.backends.mps.is_available() else 
                          "cpu")

    game_app = game_info._implementation(
        log=log,
        game=game_info,
        fps=30,  # TODO unhardcode
        device=device,
    )
    
    # Start the game using its built-in method
    game_app.start()
    
    heartbeat = GameHeartbeat(
        game=game_app,
        peer_connection=pc,
        last_heartbeat=datetime.now()
    )
    active_games[session_id] = heartbeat
    pcs.add(pc)  # Keep this for backward compatibility

    # Add Pygame video track to the peer connection
    log.debug("Adding video track to peer connection")
    pc.addTrack(GameVideoTrack(game_app))

    # Handle data channel for controls
    @pc.on("datachannel")
    def on_datachannel(channel):
        log.debug("Data channel opened", channel=channel.label)
        
        @channel.on("message")
        async def on_message(message):
            if session_id not in active_games:
                log.warning("Received message for inactive session", 
                            session_id=session_id)
                return
                
            data = json.loads(message)
            if data['type'] == 'heartbeat':
                active_games[session_id].last_heartbeat = datetime.now()
            else:
                handle_game_input(log, data)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        log.debug("Connection state change", 
                 state=pc.connectionState, 
                 session_id=session_id)
                 
        if pc.connectionState in ("failed", "closed"):
            await cleanup_game(session_id, log)

    # Set remote description and create answer
    await pc.setRemoteDescription(offer)
    answer = await pc.createAnswer()

    # Log the SDP answer before setting it
    log.debug("Generated SDP answer", sdp=answer.sdp)

    await pc.setLocalDescription(answer)

    # Return the answer
    return JSONResponse(content={
        "sdp": pc.localDescription.sdp,
        "type": pc.localDescription.type,
        "sessionId": session_id
    })

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.get("/turn-credentials")
async def turn_credentials(
    request: Request,
    log: BoundLogger = Depends(logger_dependency),
    local_only: bool = Depends(is_local_only),
):
    if local_only:
        log.info("TURN credentials requested in local mode")
        raise HTTPException(status_code=404, detail="TURN server not available in local mode")

    turn_secret = os.environ.get('TURN_SECRET_KEY')
    if turn_secret is None:
        log.error("TURN_SECRET_KEY not set in server mode")
        raise HTTPException(status_code=500, detail="TURN server configuration missing")

    credentials = create_turn_credentials(turn_secret)
    turn_base_url = request.base_url.hostname
    turn_url, stun_url = get_turn_stun_urls(turn_base_url)

    return {
        **credentials,  # Include username, credential, ttl
        "turn_urls": [turn_url],
        "stun_urls": [stun_url]
    }

# Background cleanup task
async def cleanup_stale_sessions() -> None:
    """Periodically check for and cleanup stale games"""
    while True:
        try:
            await asyncio.sleep(30)  # Check every 30 seconds
            
            stale_sessions = [
                session_id for session_id, heartbeat in active_games.items()
                if heartbeat.is_stale and not heartbeat.cleanup_scheduled
            ]
            
            for session_id in stale_sessions:
                log.info("Cleaning up stale game", session_id=session_id)
                await cleanup_game(session_id, log)
                
        except Exception as e:
            log.error("Error in cleanup task", error=str(e))

@app.on_event("startup")
async def start_cleanup_task():
    asyncio.create_task(cleanup_stale_sessions())

# Move input handling to a separate function
def handle_game_input(log: BoundLogger, data: dict) -> None:
    """Handle game input events
    
    Args:
        data: The input event data
        game_app: The game instance
        log: Logger instance
    """
    # TODO instead of statically pushing inputs via pygame.event.post, 
    #  stick them in the game app's input queue (implement it first)
    if data['type'] == 'keydown':
        pygame_key = JS_TO_PYGAME_KEY_MAP.get(data['key'])
        if pygame_key is not None:
            pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame_key))
        else:
            log.warning("Unmapped key code", key=data['key'])
    elif data['type'] == 'keyup':
        pygame_key = JS_TO_PYGAME_KEY_MAP.get(data['key'])
        if pygame_key is not None:
            pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame_key))
    elif data['type'] == 'mousemove':
        # Create a pygame MOUSEMOTION event
        movement = (data['movementX'], data['movementY'])
        pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, rel=movement))
    elif data['type'] == 'mousedown':
        # Map JavaScript mouse button to Pygame button
        pygame_button = JS_TO_PYGAME_BUTTON_MAP.get(data['button'])
        if pygame_button is not None:
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=pygame_button))
        else:
            log.warning("Unmapped mouse button", button=data['button'])
    elif data['type'] == 'mouseup':
        pygame_button = JS_TO_PYGAME_BUTTON_MAP.get(data['button'])
        if pygame_button is not None:
            pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, button=pygame_button))
        else:
            log.warning("Unmapped mouse button", button=data['button'])
    else:
        log.error("Unknown message type", type=data['type'])
        raise ValueError(f"Unknown message type: {data['type']}")


@app.get('/game/{type}/{id}/params/schema')
async def get_params_schema(
    type: str,
    id: str,
    log: BoundLogger = Depends(logger_dependency)
) -> dict:
    """Get JSON schema for game parameters"""
    game_info = games.get(type, {}).get(id)
    if not game_info:
        raise HTTPException(status_code=404, detail="Game not found")
    
    return game_info._implementation.Params.model_json_schema()


@app.post("/params/{session_id}")
async def update_game_params(
    request: Request,
    session_id: str = Path(...),
    log: BoundLogger = Depends(logger_dependency),
):
    if session_id not in active_games:
        raise HTTPException(status_code=404, detail="Game session not found")
    heartbeat = active_games[session_id]

    params = await request.json()

    # Parse the parameters
    try:
        params_model = heartbeat.game.Params.model_validate(params['params'])
    except ValidationError as e:
        log.error("Invalid game parameters", 
                 session_id=session_id, 
                 error=str(e))
        raise HTTPException(status_code=400, detail=str(e))
    
    try:
        # Update the game parameters
        heartbeat.game.on_params_update(params_model)
        return {"status": "success"}
    except Exception as e:
        log.error("Error updating game parameters", 
                 session_id=session_id, 
                 error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
