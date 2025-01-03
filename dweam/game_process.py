import os
from pathlib import Path
import sys


def setup_logging() -> None:
    """Set up logging to both console and file"""

    pid = os.getpid()
    
    cache_dir = os.environ.get("CACHE_DIR")
    if cache_dir is None:
        cache_dir = Path.home() / ".dweam" / "cache"
    else:
        cache_dir = Path(cache_dir)
        
    log_dir = cache_dir / "worker_logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"game_process_{pid}.log"
    
    # Create file handle with line buffering
    log_handle = open(log_file, 'w', buffering=1)
    
    # Save original stdout/stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    class DualOutput:
        def __init__(self, file1, file2):
            self.file1 = file1
            self.file2 = file2
        
        def write(self, data):
            self.file1.write(data)
            self.file2.write(data)
            self.file1.flush()
            self.file2.flush()
            
        def flush(self):
            self.file1.flush()
            self.file2.flush()
    
    # Replace stdout/stderr with dual-output versions
    sys.stdout = DualOutput(original_stdout, log_handle)
    sys.stderr = DualOutput(original_stderr, log_handle)

    print(f"=== Game Process {pid} Starting ===")
    print(f"Logging to {log_file}")
    print(f"Command line args: {sys.argv}")


setup_logging()


import logging
logging.getLogger("aioice.ice").disabled = True

import asyncio
import json
from typing import Any
from datetime import datetime, timedelta
from dweam.constants import JS_TO_PYGAME_BUTTON_MAP, JS_TO_PYGAME_KEY_MAP
from dweam.log_config import get_logger
from pydantic import TypeAdapter
import pygame
import numpy as np
from av.video.frame import VideoFrame
from aiortc import VideoStreamTrack, RTCPeerConnection, RTCSessionDescription, RTCConfiguration, RTCIceServer, RTCDataChannel
from aiortc.contrib.signaling import object_from_string, object_to_string
import torch
import socket
from dweam.utils.process import patch_subprocess_popen

from dweam.utils.entrypoint import load_games, get_cache_dir
from dweam.commands import (
    Command, Response, SchemaCommand, StopCommand, 
    UpdateParamsCommand, HandleOfferCommand,
    SuccessResponse, ErrorResponse
)


class GameVideoTrack(VideoStreamTrack):
    """A video stream track that captures frames from a Pygame application."""
    def __init__(self, game: Any):
        super().__init__()
        self.game = game

    async def recv(self) -> VideoFrame:
        await asyncio.sleep(1 / 30)  # 30 FPS
        surface = await self.game.get_next_frame()
        frame = pygame.surfarray.array3d(surface)
        frame = np.fliplr(frame)
        frame = np.rot90(frame)
        new_frame = VideoFrame.from_ndarray(frame, format='rgb24')
        new_frame.pts, new_frame.time_base = await self.next_timestamp()
        return new_frame

class GameRTCConnection:
    def __init__(self, game: Any, ice_servers: list[dict] | None = None):
        self.game = game
        self.last_heartbeat = datetime.now()
        self.cleanup_scheduled = False
        
        # Configure ICE servers
        config = RTCConfiguration(
            iceServers=[RTCIceServer(**server) for server in (ice_servers or [])]
        )
        self.pc = RTCPeerConnection(configuration=config)
        self.data_channel: RTCDataChannel | None = None
        
        # Add video track
        self.pc.addTrack(GameVideoTrack(self.game))
        
        @self.pc.on("datachannel")
        def on_datachannel(channel: RTCDataChannel):
            self.data_channel = channel
            
            @channel.on("message")
            def on_message(message):
                if not isinstance(message, str):
                    return
                    
                try:
                    data = json.loads(message)
                    if data["type"] == "heartbeat":
                        self.last_heartbeat = datetime.now()
                    else:
                        self.handle_game_input(data)
                except Exception as e:
                    print(f"Error handling message: {e}", file=sys.stderr)
                    
        @self.pc.on("connectionstatechange")
        async def on_connectionstatechange():
            # print(f"Connection state changed to: {self.pc.connectionState}", file=sys.stderr)
            if self.pc.connectionState in ("failed", "closed", "disconnected"):
                await self.cleanup()

    @property
    def is_stale(self) -> bool:
        """Check if the connection hasn't received a heartbeat recently"""
        return datetime.now() - self.last_heartbeat > timedelta(seconds=5)

    def handle_game_input(self, data: dict):
        """Handle game input events"""
        try:
            if data["type"] == "keydown":
                pygame_key = JS_TO_PYGAME_KEY_MAP.get(data["key"])
                if pygame_key is not None:
                    pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame_key))
            elif data["type"] == "keyup":
                pygame_key = JS_TO_PYGAME_KEY_MAP.get(data["key"])
                if pygame_key is not None:
                    pygame.event.post(pygame.event.Event(pygame.KEYUP, key=pygame_key))
            elif data["type"] == "mousemove":
                movement = (data["movementX"], data["movementY"])
                pygame.event.post(pygame.event.Event(pygame.MOUSEMOTION, rel=movement))
            elif data["type"] == "mousedown":
                pygame_button = JS_TO_PYGAME_BUTTON_MAP.get(data["button"])
                if pygame_button is not None:
                    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=pygame_button))
            elif data["type"] == "mouseup":
                pygame_button = JS_TO_PYGAME_BUTTON_MAP.get(data["button"])
                if pygame_button is not None:
                    pygame.event.post(pygame.event.Event(pygame.MOUSEBUTTONUP, button=pygame_button))
        except Exception as e:
            print(f"Error handling input: {e}", file=sys.stderr)

    async def handle_offer(self, sdp: str, type_: str):
        """Handle incoming WebRTC offer"""
        offer = RTCSessionDescription(sdp=sdp, type=type_)
        await self.pc.setRemoteDescription(offer)
        
        answer = await self.pc.createAnswer()
        await self.pc.setLocalDescription(answer)
        
        return {
            "sdp": self.pc.localDescription.sdp,
            "type": self.pc.localDescription.type
        }

    async def cleanup(self):
        """Cleanup resources"""
        if self.cleanup_scheduled:
            return
            
        self.cleanup_scheduled = True
        if self.pc.connectionState != "closed":
            await self.pc.close()
        # Game cleanup will be handled by the main process

async def main():
    # Patch subprocess to hide windows in release mode
    patch_subprocess_popen()
    
    log = get_logger().bind(process="worker")
    log.info("Starting worker process")

    # Log all command line arguments
    log.info("Command line args", argv=sys.argv)
    
    # Get game type, ID and optional ICE servers from command line args

    try:
        game_type = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        # Windows seems to decode the json implicitly
        game_type = sys.argv[1]

    game_id = sys.argv[2]
    ice_servers = json.loads(sys.argv[3]) if len(sys.argv) > 3 else None
    port = int(sys.argv[4])
    
    log.info("Parsed args", game_type=game_type, game_id=game_id, port=port)
    
    log.info("Attempting to connect to parent", host='127.0.0.1', port=port)
    
    try:
        # Connect to parent process
        reader, writer = await asyncio.open_connection(
            '127.0.0.1',
            port
        )
        log.info("Connected to parent")
    except Exception as e:
        log.exception("Failed to connect to parent")
        raise

    # Load the game implementation
    games = load_games(log)
    log.info("Loaded games")
    log.info("Looking up game", game_type=game_type, game_id=game_id)
    
    # Check if game_type exists
    if game_type not in games:
        log.error("Game type not found", available_types=list(games.keys()))
        raise KeyError(f"Game type '{game_type}' not found")
        
    # Check if game_id exists
    if game_id not in games[game_type]:
        log.error("Game ID not found", 
                 available_ids=list(games[game_type].keys()),
                 game_type=game_type)
        raise KeyError(f"Game ID '{game_id}' not found in {game_type}")
    
    game_info = games[game_type][game_id]
    implementation = game_info.get_implementation()
    
    game = None
    rtc = None
    should_exit = False

    async def check_connection():
        """Check connection state and cleanup if stale"""
        nonlocal should_exit
        while not should_exit:
            await asyncio.sleep(1)  # Check every second
            if rtc and (rtc.is_stale or rtc.pc.connectionState in ("failed", "closed", "disconnected")):
                log.info("Connection stale or closed, cleaning up")
                await rtc.cleanup()
                if game:
                    game.stop()
                    
                writer.close()
                await writer.wait_closed()
                log.info("Connection checker requesting process exit")
                should_exit = True
                return
    
    # Start connection checker
    checker_task = asyncio.create_task(check_connection())
    
    # Process commands
    while not should_exit:
        try:
            line = await reader.readline()
            if not line:
                should_exit = True
                break
                
            command = TypeAdapter(Command).validate_json(line)
            response: Response
            
            try:
                if isinstance(command, SchemaCommand):
                    schema = implementation.Params.model_json_schema()
                    response = SuccessResponse(data=schema)
                    
                elif isinstance(command, UpdateParamsCommand):
                    params = implementation.Params.model_validate(command.data)
                    game.on_params_update(params)
                    response = SuccessResponse()
                    
                elif isinstance(command, HandleOfferCommand):
                    if game is None:
                        game = implementation(
                            log=log,
                            game_id=game_id,
                        )
                        game.start()
                    rtc = GameRTCConnection(game, ice_servers)
                    answer = await rtc.handle_offer(command.data.sdp, command.data.type)
                    response = SuccessResponse(data=answer)
                    
                elif isinstance(command, StopCommand):
                    if rtc:
                        await rtc.cleanup()
                    response = SuccessResponse()
                    should_exit = True
                
            except Exception as e:
                log.exception("Error processing command", command_type=type(command))
                response = ErrorResponse(error=str(e))
                
            writer.write(response.model_dump_json().encode() + b"\n")
            await writer.drain()
            
        except Exception as e:
            print(f"Error processing command: {e}", file=sys.stderr)
            should_exit = True

    # Clean up
    if rtc:
        await rtc.cleanup()
    if game:
        game.stop()
    writer.close()
    await writer.wait_closed()
    checker_task.cancel()
    try:
        await checker_task
    except asyncio.CancelledError:
        pass

if __name__ == "__main__":
    asyncio.run(main()) 
