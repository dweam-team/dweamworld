import asyncio
from asyncio.subprocess import Process
import json
import os
from typing import Optional, Any
from datetime import datetime, timedelta
from pathlib import Path
from asyncio import StreamReader, StreamWriter
import sys
from importlib.resources import files

from pydantic import TypeAdapter

from aiortc import RTCPeerConnection, RTCSessionDescription, RTCIceServer
from av.video.frame import VideoFrame

from dweam.models import GameInfo
from dweam.utils.entrypoint import get_cache_dir
from dweam.log_config import get_logger
from dweam.utils.turn import create_turn_credentials, get_turn_stun_urls
from dweam.constants import JS_TO_PYGAME_KEY_MAP, JS_TO_PYGAME_BUTTON_MAP
from structlog.stdlib import BoundLogger
from dweam.commands import Command, Response, SchemaCommand, StopCommand, UpdateParamsCommand, HandleOfferCommand, OfferData, ErrorResponse
from dweam.utils.process import get_asyncio_subprocess_flags

def is_debug_build() -> bool:
    """Detect if we're running the debug build based on executable name"""
    if getattr(sys, 'frozen', False):
        # We're running in a PyInstaller bundle
        executable_path = sys.executable
        return 'debug' in os.path.basename(executable_path).lower()
    return True  # In development environment, always use debug mode

class GameWorker:
    def __init__(
        self,
        log: BoundLogger,
        game_info: GameInfo,
        session_id: str,
        game_type: str,
        game_id: str,
        venv_path: Path,
    ):
        self.log = log
        self.game_info = game_info
        self.game_type = game_type
        self.game_id = game_id
        self.session_id = session_id
        self.venv_path = venv_path

        self.last_heartbeat = datetime.now()
        self.cleanup_scheduled = False
        
        # Communication handles
        self.process: Optional[Process] = None
        self.reader: Optional[StreamReader] = None
        self.writer: Optional[StreamWriter] = None
        
        # WebRTC
        self.pc: Optional[RTCPeerConnection] = None

        self.last_log_line: str | None = None

    async def _monitor_process_output(self, stream: StreamReader | None, stream_name: str):
        """Monitor output stream of the worker process and log any output"""
        if stream is None:
            self.log.error("Failed to monitor worker stream – stream is None", stream_name=stream_name)
            return
        
        while True:
            line = await stream.readline()
            if not line:
                break
            try:
                # Try UTF-8 first
                output_line = line.decode('utf-8').rstrip()
            except UnicodeDecodeError:
                # Fall back to a more lenient encoding that replaces invalid characters
                output_line = line.decode('utf-8', errors='replace').rstrip()
            
            # Store the last non-empty line
            if output_line.strip():
                self.last_log_line = output_line
                
            self.log.info(f"Worker {stream_name}", line=output_line)

    async def _collect_process_output(self, process: Process) -> tuple[str | None, str | None]:
        stdout_str = None
        stderr_str = None

        if process.stdout:
            try:
                self.log.debug("Reading stdout")
                stdout_data = await asyncio.wait_for(process.stdout.read(), timeout=2.0)
                stdout_str = stdout_data.decode() if stdout_data else None
            except asyncio.TimeoutError:
                self.log.warning("Timeout reading stdout")
                return stdout_str, stderr_str
            except Exception:
                self.log.exception("Error reading stdout")
                return stdout_str, stderr_str

        if process.stderr:
            try:
                self.log.debug("Reading stderr")
                stderr_data = await asyncio.wait_for(process.stderr.read(), timeout=2.0)
                stderr_str = stderr_data.decode() if stderr_data else None
            except asyncio.TimeoutError:
                self.log.warning("Timeout reading stderr")
                return stdout_str, stderr_str
            except Exception:
                self.log.exception("Error reading stderr")
                return stdout_str, stderr_str

        return stdout_str, stderr_str

    async def _establish_connection(self, timeout: float) -> bool:
        """Establish connection with the worker process"""
        if sys.platform == "win32":
            venv_python = self.venv_path / "Scripts" / "python.exe"
        else:
            venv_python = self.venv_path / "bin" / "python"

        # Use importlib.resources to reliably locate the module file
        if getattr(sys, 'frozen', False):
            worker_script = Path(sys._MEIPASS) / "dweam" / "dweam" / "game_process.py"
        else:
            worker_script = files('dweam').joinpath('game_process.py')

        # Create a TCP server socket
        client_connected = asyncio.Event()

        async def handle_client(reader, writer):
            self.reader = reader
            self.writer = writer
            client_connected.set()
        
        # Create server and get the port
        server = await asyncio.start_server(
            handle_client,
            host='127.0.0.1',
            port=0
        )
        port = server.sockets[0].getsockname()[1]

        self.log.info("Got available port", port=port)
        
        # Start the worker process with the port number
        self.log.info(
            "Starting worker process", 
            python=str(venv_python),
            script=str(worker_script),
            game_type=self.game_type,
            game_id=self.game_id
        )

        # Start serving (but don't block)
        async with server:
            self.log.debug("Starting server")
            server_task = asyncio.create_task(server.serve_forever())
            
            try:
                self.process = await asyncio.create_subprocess_exec(
                    str(venv_python),
                    str(worker_script),
                    json.dumps(self.game_type),
                    self.game_id,
                    json.dumps([]),
                    str(port),
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    creationflags=get_asyncio_subprocess_flags()
                )
                self.log.info("Started worker process", pid=self.process.pid)

                # Add immediate process status check with timeout
                try:
                    await asyncio.wait_for(
                        asyncio.create_task(self.process.wait()),
                        timeout=0.1
                    )
                    # If we get here, process exited too quickly
                    stderr, stdout = await self._collect_process_output(self.process)
                    self.log.error(
                        "Process failed to start",
                        returncode=self.process.returncode,
                        stdout=stdout,
                        stderr=stderr
                    )
                    return False
                except asyncio.TimeoutError:
                    # Process is still running, this is good
                    pass
                # Wait for either client connection or process termination
                done, pending = await asyncio.wait([
                    asyncio.create_task(client_connected.wait()),
                    asyncio.create_task(self.process.wait())
                ], timeout=timeout, return_when=asyncio.FIRST_COMPLETED)

                # If process completed first, it means it crashed
                if self.process.returncode is not None:
                    stdout_str, stderr_str = await self._collect_process_output(self.process)
                    self.log.error(
                        "Process terminated before connection",
                        returncode=self.process.returncode,
                        stderr=stderr_str,
                        stdout=stdout_str
                    )
                    return False  # Changed from raise to return False to allow retries

                # If we get here and nothing completed, it was a timeout
                if not done:
                    raise asyncio.TimeoutError("Worker process failed to connect")

            finally:
                # Cancel server task
                server_task.cancel()
                try:
                    await server_task
                except asyncio.CancelledError:
                    pass

            if not self.reader or not self.writer:
                raise RuntimeError("No connection received")

        return True

    async def start(self):
        """Start the worker process and establish communication"""        
        max_retries = 3
        base_timeout = 5.0

        for attempt in range(max_retries):
            try:
                timeout = base_timeout * (2 ** attempt)  # 5s, 10s, 20s

                # Try to establish connection
                try:
                    if not await self._establish_connection(timeout):
                        continue
                    
                    # Start monitoring process output
                    asyncio.create_task(self._monitor_process_output(self.process.stdout, "stdout"))
                    asyncio.create_task(self._monitor_process_output(self.process.stderr, "stderr"))
                    
                    self.log.info("Client connected")
                    return  # Success!

                except Exception:
                    self.log.exception("Error during connection")
                    if attempt == max_retries - 1:
                        raise
                    continue

            except Exception:
                self.log.exception(f"Attempt {attempt + 1} failed")
                if self.process:
                    self.process.kill()
                if attempt == max_retries - 1:
                    raise

        raise RuntimeError(f"Failed to start worker after {max_retries} attempts")

    async def run(self, offer: RTCSessionDescription) -> RTCSessionDescription:
        """Set up and run the WebRTC connection"""
        if not self.process:
            await self.start()

        # Pass the offer to game process and get answer
        response = await self._send_command(HandleOfferCommand(
            cmd="handle_offer",
            data=OfferData(sdp=offer.sdp, type=offer.type)
        ))
        
        # Convert response to RTCSessionDescription
        return RTCSessionDescription(sdp=response["sdp"], type=response["type"])

    async def _send_command(self, command: Command) -> Any:
        """Send a command to the worker process and get the response"""
        if not self.writer or not self.reader:
            raise RuntimeError("Worker process not started")
        
        message = command.model_dump_json() + "\n"
        self.writer.write(message.encode())
        await self.writer.drain()
        
        response = await self.reader.readline()
        if response == b"":
            raise RuntimeError("Worker process closed")
        
        self.log.info("Worker response", response=response)
        result = TypeAdapter(Response).validate_json(response)
        
        if isinstance(result, ErrorResponse):
            raise ValueError(result.error)
        return result.data

    async def get_params_schema(self) -> dict[str, Any]:
        """Get the JSON schema for game parameters"""
        if not self.process:
            await self.start()
        return await self._send_command(SchemaCommand())

    async def update_params(self, params: dict) -> None:
        """Update game parameters"""
        if not self.process:
            await self.start()
        return await self._send_command(UpdateParamsCommand(data=params))

    async def cleanup(self):
        """Clean up worker resources"""
        if self.cleanup_scheduled:
            return
        
        self.cleanup_scheduled = True
        try:
            if self.writer:
                try:
                    await self._send_command(StopCommand())
                except:
                    pass
                self.writer.close()
                await self.writer.wait_closed()
            
            # Close socket
            if hasattr(self, '_socket'):
                self._socket.close()
            
            if self.process and self.process.returncode is None:
                self.process.terminate()
                try:
                    await asyncio.wait_for(self.process.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    self.process.kill()
                    
        except Exception as e:
            self.log.error("Error during cleanup", error=str(e)) 