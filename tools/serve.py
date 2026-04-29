#!/usr/bin/env python3
"""
Local server for the roomgame scanner + game.

By default, serves plain HTTP on port 8080. That is enough for Mac Chrome —
`http://localhost:8080` is treated as a **secure context**, so WebXR (when
the laptop has it) and camera both work without any certificate.

For testing on your Android phone (the WebXR scanner) there are a few options:

    # A. HTTPS on the local Wi-Fi (self-signed cert; Chrome shows a warning)
    python3 tools/serve.py --https

    # B. Cloudflare Tunnel — real HTTPS URL, works anywhere
    cloudflared tunnel --url http://localhost:8080

    # C. ngrok — real HTTPS URL, works anywhere
    ngrok http 8080

    # D. adb reverse — phone over USB, no cert needed
    adb reverse tcp:8080 tcp:8080
    # then open http://localhost:8080/ on the phone

Endpoints:
    GET  /             → static files from web/
    POST /frame        → binary WebXR depth frame from phone (M1: log only)
    POST /save         → write current voxel grid to disk        (M2)
    POST /remesh       → mesh-extract current grid to GLB        (M3)
    POST /carve        → CSG-subtract a sphere from the grid     (M4)
    GET  /stats        → JSON status (active voxel count, bbox)  (M2)
    POST /log          → single line of phone-side console output
    POST /log/dump     → bigger framed log dump from the phone
"""
from __future__ import annotations

import argparse
import http.server
import ipaddress
import json
import os
import re
import socket
import ssl
import struct
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = PROJECT_ROOT / "web"
# Output artifacts (room.npz, room.glb) live inside web/ so the static file
# handler can serve them at /out/room.glb without any extra plumbing.
OUT_DIR = WEB_DIR / "out"
# Captured frames (raw /frame bodies) for offline replay & debugging.
# One file per frame; named frame_NNNNNN.bin in capture order.
FRAMES_DIR = PROJECT_ROOT / "captured_frames"
DEFAULT_CERT_DIR = Path.home() / ".roomgame" / "cert"

# Wire-format header for POST /frame:
#   16 floats viewMatrix             (world_from_view, column-major Float32)
#   16 floats projectionMatrix       (column-major Float32)
#   16 floats normDepthBufferFromNormView (column-major Float32)
#    1 uint32 depth_width
#    1 uint32 depth_height
#    1 float32 rawValueToMeters
#    1 uint32 depth_format            (0=uint16 luminance-alpha, 1=float32)
#    1 uint32 color_width
#    1 uint32 color_height
#    1 uint32 color_format            (0=none, 1=RGBA8)
#    1 uint32 color_byte_length
# Total = 48f + 7 u32 + 1 f32 = 48*4 + 32 = 224 bytes.
# Body = header + depth payload + color payload.
FRAME_HEADER_FMT = "<48f I I f I I I I I"
FRAME_HEADER_SIZE = struct.calcsize(FRAME_HEADER_FMT)
assert FRAME_HEADER_SIZE == 224, FRAME_HEADER_SIZE

FORMAT_UINT16_LA = 0
FORMAT_FLOAT32 = 1
COLOR_NONE = 0
COLOR_RGBA8 = 1


def parse_frame(body: bytes) -> dict:
    """Decode a /frame body into header fields + raw depth buffer + raw color
    buffer (or None if the frame didn't carry color)."""
    if len(body) < FRAME_HEADER_SIZE:
        raise ValueError(f"frame body too short: {len(body)} < {FRAME_HEADER_SIZE}")
    fields = struct.unpack(FRAME_HEADER_FMT, body[:FRAME_HEADER_SIZE])
    view_m = fields[0:16]
    proj_m = fields[16:32]
    norm_m = fields[32:48]
    depth_w = fields[48]
    depth_h = fields[49]
    raw_to_m = fields[50]
    depth_fmt = fields[51]
    color_w = fields[52]
    color_h = fields[53]
    color_fmt = fields[54]
    color_len = fields[55]

    if depth_fmt == FORMAT_UINT16_LA:
        depth_bpp = 2
    elif depth_fmt == FORMAT_FLOAT32:
        depth_bpp = 4
    else:
        raise ValueError(f"unknown depth format code {depth_fmt}")
    depth_expected = depth_w * depth_h * depth_bpp

    body_after_header = body[FRAME_HEADER_SIZE:]
    if len(body_after_header) < depth_expected:
        raise ValueError(
            f"depth payload truncated: got {len(body_after_header)}, "
            f"expected {depth_expected}+color ({depth_w}x{depth_h} @ {depth_bpp} B/px)"
        )
    depth_payload = body_after_header[:depth_expected]

    if color_fmt == COLOR_NONE:
        if color_len != 0:
            raise ValueError(f"color_format=NONE but color_byte_length={color_len}")
        color_payload = None
    elif color_fmt == COLOR_RGBA8:
        expected_color = color_w * color_h * 4
        if expected_color != color_len:
            raise ValueError(f"RGBA8 declared {color_len} bytes, expected {expected_color}")
        rest = body_after_header[depth_expected:]
        if len(rest) < color_len:
            raise ValueError(f"color payload truncated: got {len(rest)}, expected {color_len}")
        color_payload = rest[:color_len]
    else:
        raise ValueError(f"unknown color format code {color_fmt}")

    return {
        "viewMatrix": view_m,
        "projectionMatrix": proj_m,
        "normDepthBufferFromNormView": norm_m,
        "width": depth_w,
        "height": depth_h,
        "rawValueToMeters": raw_to_m,
        "format": depth_fmt,
        "depth": depth_payload,
        "color_width": color_w,
        "color_height": color_h,
        "color_format": color_fmt,
        "color": color_payload,
    }


def print_qr_to_stderr(url: str) -> None:
    """Print a scannable QR code to stderr. Tries `qrencode` (brew) then the
    `qrcode` pip package; if neither is available, prints an install hint."""
    for args in (
        ["qrencode", "-t", "ANSIUTF8", "-m", "1", url],
        ["qrencode", "-t", "UTF8", url],
    ):
        try:
            subprocess.run(args, check=True, stdout=sys.stderr)
            return
        except FileNotFoundError:
            break
        except subprocess.CalledProcessError:
            continue
    try:
        import qrcode  # type: ignore
        qr = qrcode.QRCode(border=1)
        qr.add_data(url)
        qr.make(fit=True)
        qr.print_ascii(out=sys.stderr, invert=True)
        return
    except ImportError:
        pass
    sys.stderr.write(
        "(Install `brew install qrencode` or `pip install qrcode` "
        "to see a scannable QR code here next time.)\n"
    )


def local_ipv4_addresses() -> list[str]:
    addrs: list[str] = []
    try:
        out = subprocess.check_output(["ifconfig"], text=True, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return addrs
    for m in re.finditer(r"\binet (\d+\.\d+\.\d+\.\d+)\b", out):
        ip = m.group(1)
        try:
            if ipaddress.IPv4Address(ip).is_loopback:
                continue
        except ValueError:
            continue
        addrs.append(ip)
    return addrs


def primary_ipv4() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        pass
    addrs = local_ipv4_addresses()
    return addrs[0] if addrs else "127.0.0.1"


def hostname_local() -> str | None:
    try:
        name = socket.gethostname()
    except OSError:
        return None
    if not name:
        return None
    return name if name.endswith(".local") else name + ".local"


def ensure_cert(cert_dir: Path, sans: list[str]) -> tuple[Path, Path]:
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert = cert_dir / "cert.pem"
    key = cert_dir / "key.pem"
    meta = cert_dir / "sans.txt"
    sans_line = "\n".join(sorted(sans))
    if cert.exists() and key.exists() and meta.exists() and meta.read_text() == sans_line:
        return cert, key

    print(f"Generating self-signed cert in {cert_dir} ...", file=sys.stderr)
    san_parts: list[str] = []
    for s in sans:
        try:
            ipaddress.ip_address(s)
            san_parts.append(f"IP:{s}")
        except ValueError:
            san_parts.append(f"DNS:{s}")
    san_ext = "subjectAltName=" + ",".join(san_parts)
    try:
        subprocess.run(
            [
                "openssl", "req", "-x509",
                "-newkey", "rsa:2048", "-nodes",
                "-keyout", str(key), "-out", str(cert),
                "-days", "397",
                "-subj", "/CN=localhost",
                "-addext", san_ext,
                "-addext", "keyUsage=critical,digitalSignature,keyEncipherment",
                "-addext", "extendedKeyUsage=serverAuth",
                "-addext", "basicConstraints=critical,CA:FALSE",
            ],
            check=True,
        )
    except FileNotFoundError:
        print("ERROR: openssl not found. Install it (e.g. `brew install openssl`) "
              "or use a tunnel (cloudflared, ngrok) instead.", file=sys.stderr)
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: openssl failed (exit {e.returncode}).", file=sys.stderr)
        sys.exit(1)
    meta.write_text(sans_line)
    return cert, key


# Module-level state shared across requests.
#
# `voxel_room` is initialised in run_server() if pyopenvdb is importable.
# When it's None, /frame still parses + logs (for transport debugging) but
# /save and /stats return a clear "not initialised" response.
voxel_room: object | None = None
frame_count = 0  # legacy global, kept for /stats; per-session counts live in SESSIONS

# Coarse occupancy cubes — independent representation populated by /frame
# while a recording is active (controlled by /cubes/start and /cubes/stop).
# `cube_grid` is None until /cubes/start is called for the first time.
cube_grid: object | None = None
cube_recording: bool = False

# Per-session frame state. Each "session" is a phone scanner page-load on the
# laptop's wall clock — clients call POST /session/new to mint one and then
# include ?session=<id> on every /frame POST. Frames land under
# `captured_frames/<id>/frames/`. Legacy frames (no session) fall back to
# `captured_frames/frame_NNNNNN.bin` so existing recordings keep working.
import threading as _threading
import time as _time
SESSIONS: dict[str, dict] = {}
SESSIONS_LOCK = _threading.Lock()
DEFAULT_SESSION_ID: str | None = None
# Variant naming for `voxels_<variant>.json` files inside a session dir.
# The set of variants isn't fixed (depth_refine.py, the loop-closure tool,
# future MVS refiners etc. all write under their own name) so we accept
# anything matching VARIANT_RE and let the per-session listing report what's
# actually on disk. The strict regex still blocks path traversal.
SESSION_ID_RE      = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")
SESSION_VARIANT_RE = re.compile(r"^[A-Za-z0-9_\-]{1,32}$")
SESSION_VOXELS_FILE_RE = re.compile(r"^voxels_([A-Za-z0-9_\-]{1,32})\.json$")

# Frame-debug endpoints serve from any directory matching this pattern
# under captured_frames/<session>/. Replacing the previous hard-coded
# allowlist with a regex lets new pipeline outputs (frames_feature_ba/,
# frames_refined_feature_ba/, …) appear in the viewer without a server
# code change. The strict pattern still blocks path traversal.
FRAME_VARIANT_RE = re.compile(r"^frames(?:_[A-Za-z0-9_\-]{1,32})?$")
# Cap on how many of the captured colour pixels we project for the
# frame-voxels endpoint (subsample stride = ceil(sqrt(npix / cap))).
FRAME_VOXELS_PIXEL_CAP = 80_000
# In-memory thumbnail cache. Keyed by (session, variant_dir, idx, kind).
_THUMB_CACHE: dict[tuple, bytes] = {}
_THUMB_CACHE_MAX = 2000  # ~50KB each → ~100MB ceiling

# Cached blend grids for the stereo page. Keyed by
# (session, idx, sigma_quantised, out_w, out_h). Each entry is a float32
# array (≈ 1000×400×4B = 1.6 MB), so 16 entries top out around 25 MB.
_BLEND_GRID_CACHE: dict[tuple, "numpy.ndarray"] = {}
_BLEND_GRID_CACHE_MAX = 16


def _features_meta_filename(variant: str) -> str:
    """Map the voxelview variant string to the corresponding features_meta
    filename written by feature_ray_reconstruct.py. The reconstruct tool
    emits `features_meta.json` for `--frames-variant frames` and
    `features_meta_<suffix>.json` for any other frames variant."""
    if not re.fullmatch(r"features(?:_[A-Za-z0-9_\-]{1,32})?", variant):
        raise ValueError(f"not a features variant: {variant!r}")
    if variant == "features":
        return "features_meta.json"
    return f"features_meta_{variant[len('features_'):]}.json"


def _mint_session_id() -> str:
    return _time.strftime("%Y%m%d_%H%M%S", _time.localtime())


def _ensure_session(session_id: str | None = None) -> str:
    """Create the session if it doesn't exist yet. Pass `session_id=None` to
    mint a fresh wall-clock id; collisions within the same second get a
    `_2`, `_3`, … suffix."""
    with SESSIONS_LOCK:
        if session_id is None:
            base = _mint_session_id()
            sid = base
            n = 2
            while sid in SESSIONS or (FRAMES_DIR / sid).exists():
                sid = f"{base}_{n}"
                n += 1
        else:
            sid = session_id
        if sid not in SESSIONS:
            SESSIONS[sid] = {"count": 0, "created": _time.time()}
            try:
                (FRAMES_DIR / sid / "frames").mkdir(parents=True, exist_ok=True)
            except OSError as e:
                sys.stderr.write(f"session {sid}: mkdir failed: {e}\n")
        return sid


def _next_frame_index(session_id: str) -> int:
    with SESSIONS_LOCK:
        SESSIONS[session_id]["count"] += 1
        return SESSIONS[session_id]["count"]


def init_voxel_room(voxel_size_m: float) -> None:
    """Try to construct the global VoxelRoom; log a clear message if numpy
    is missing and leave the global as None so /frame still works for
    transport-level smoke testing."""
    global voxel_room
    try:
        import voxel_store  # tools/ dir is on sys.path when serve.py runs as a script
    except ImportError as e:
        sys.stderr.write(
            f"voxel_store import failed ({e}). /frame will parse + log but\n"
            "won't fuse points into a grid. Install with:\n"
            "  pip install -r tools/requirements.txt\n"
        )
        return
    voxel_room = voxel_store.VoxelRoom(voxel_size_m=voxel_size_m)
    sys.stderr.write(f"VoxelRoom ready (voxel_size={voxel_size_m} m, chunk={voxel_store.CHUNK_SIZE}³).\n")


class RoomgameHandler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {
        **http.server.SimpleHTTPRequestHandler.extensions_map,
        ".glb": "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".wasm": "application/wasm",
        ".mjs": "application/javascript",
    }

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        super().end_headers()

    def log_message(self, fmt, *args):
        sys.stderr.write(f"[{self.address_string()}] {fmt % args}\n")

    def do_OPTIONS(self):
        self.send_response(204)
        self.end_headers()

    def _read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length", "0") or "0")
        return self.rfile.read(length) if length else b""

    def _send_text(self, code: int, text: str, content_type: str = "text/plain; charset=utf-8") -> None:
        body = text.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, payload: dict) -> None:
        self._send_text(code, json.dumps(payload), "application/json")

    def do_POST(self):
        global frame_count
        from urllib.parse import urlparse
        path = (urlparse(self.path).path or "/").rstrip("/") or "/"
        if path == "/frame":
            self._handle_frame()
            return
        if path == "/session/new":
            sid = _ensure_session()
            sys.stderr.write(f"[{self.address_string()}] new session: {sid}\n")
            self._send_json(200, {"session": sid})
            return
        if path == "/save":
            if voxel_room is None:
                self._send_text(503, "voxel store not initialised — install deps (see tools/requirements.txt)\n")
                return
            try:
                out = OUT_DIR / "room.npz"
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                voxel_room.save(out)
                self._send_json(200, {"saved": str(out)})
            except Exception as e:  # noqa: BLE001
                self._send_text(500, f"{type(e).__name__}: {e}\n")
            return
        if path == "/reset":
            if voxel_room is None:
                self._send_text(503, "voxel store not initialised\n")
                return
            voxel_room.reset()
            sys.stderr.write(f"[{self.address_string()}] RESET — voxel grid wiped\n")
            self._send_json(200, {"reset": True})
            return
        if path == "/remesh":
            if voxel_room is None:
                self._send_text(503, "voxel store not initialised — install deps\n")
                return
            try:
                import meshing  # tools/ on sys.path
                out = OUT_DIR / "room.glb"
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                meta = meshing.remesh_to_glb(voxel_room, out)
                if meta is None:
                    self._send_text(409, "no active voxels to mesh — scan something first\n")
                    return
                self._send_json(200, meta)
            except ImportError as e:
                self._send_text(503, f"missing meshing deps: {e}\n")
            except Exception as e:  # noqa: BLE001
                self._send_text(500, f"{type(e).__name__}: {e}\n")
            return
        if path == "/cubes/start":
            self._handle_cubes_start()
            return
        if path == "/cubes/stop":
            self._handle_cubes_stop()
            return
        if path == "/cubes/reset":
            self._handle_cubes_reset()
            return
        if path == "/log" or path == "/log/dump":
            body = self._read_body()
            try:
                text = body.decode("utf-8", errors="replace").strip()
            except Exception:
                text = repr(body)
            ip = self.address_string()
            if path == "/log/dump":
                bar = "=" * 68
                sys.stderr.write(f"\n{bar}\nPHONE LOG DUMP from {ip}\n{bar}\n{text}\n{bar}\n\n")
            else:
                sys.stderr.write(f"[{ip}] LOG {text}\n")
            self.send_response(204)
            self.end_headers()
            return
        self.send_response(404)
        self.end_headers()

    def do_GET(self):
        from urllib.parse import urlparse
        path = (urlparse(self.path).path or "/").rstrip("/") or "/"
        if path == "/sessions":
            self._handle_sessions_list()
            return
        if path.startswith("/captures/"):
            self._handle_capture_static(path)
            return
        if path == "/cubes/state":
            self._handle_cubes_state()
            return
        if path == "/stats":
            if voxel_room is None:
                self._send_json(200, {"frames": frame_count, "voxels": 0, "bbox": None, "ready": False})
                return
            try:
                stats = voxel_room.stats()
                stats["frames"] = frame_count
                stats["ready"] = True
                self._send_json(200, stats)
            except Exception as e:  # noqa: BLE001
                self._send_text(500, f"{type(e).__name__}: {e}\n")
            return
        return super().do_GET()

    def _handle_cubes_start(self) -> None:
        """Initialise (or re-initialise) the occupancy-cube grid and switch
        ingestion on. Body is an optional JSON object:
            { "cube_size": 0.5,
              "world_min": [-3, -0.3, -3],
              "world_max": [ 3,  2.9, 3],
              "reset": true }
        Any field is optional; defaults match cubes.CubeGrid defaults.
        Re-calling /cubes/start with a different cube_size or bbox replaces
        the grid (counters lost). With the same shape and reset=false the
        existing counters are kept and recording is just re-armed.
        """
        global cube_grid, cube_recording
        body = self._read_body()
        cfg = {}
        if body:
            try:
                cfg = json.loads(body.decode("utf-8"))
            except Exception as e:  # noqa: BLE001
                self._send_text(400, f"bad JSON body: {e}\n")
                return
        try:
            import cubes  # tools/ on sys.path
        except ImportError as e:
            self._send_text(503, f"cubes module unavailable: {e}\n")
            return

        cube_size = float(cfg.get("cube_size", cubes.CubeGrid.DEFAULT_CUBE_SIZE))
        world_min = cfg.get("world_min")
        world_max = cfg.get("world_max")
        reset_flag = bool(cfg.get("reset", True))

        # Decide whether to rebuild. Rebuild on first call, on geometry change,
        # or when reset=true. Otherwise keep the existing counters.
        rebuild = (
            cube_grid is None
            or reset_flag
            or abs(getattr(cube_grid, "cube_size", 0) - cube_size) > 1e-9
            or (world_min is not None and list(world_min) != list(getattr(cube_grid, "world_min", [])))
        )
        if rebuild:
            try:
                cube_grid = cubes.CubeGrid(
                    cube_size=cube_size,
                    world_min=tuple(world_min) if world_min is not None else None,
                    world_max=tuple(world_max) if world_max is not None else None,
                )
            except Exception as e:  # noqa: BLE001
                self._send_text(400, f"CubeGrid init failed: {type(e).__name__}: {e}\n")
                return

        cube_recording = True
        sys.stderr.write(
            f"[{self.address_string()}] CUBES start "
            f"cube_size={cube_grid.cube_size}m shape={cube_grid.shape} "
            f"({cube_grid.n_cubes} cubes) recording=on\n"
        )
        self._send_json(200, {
            "recording": True,
            "cube_size": cube_grid.cube_size,
            "shape": list(cube_grid.shape),
            "world_min": cube_grid.world_min.tolist(),
            "world_max": cube_grid.world_max.tolist(),
            "n_cubes": cube_grid.n_cubes,
            "frames": cube_grid.frames,
        })

    def _handle_cubes_stop(self) -> None:
        global cube_recording
        if cube_grid is None:
            self._send_text(409, "no cube grid initialised — call /cubes/start first\n")
            return
        cube_recording = False
        sys.stderr.write(
            f"[{self.address_string()}] CUBES stop after {cube_grid.frames} frames "
            f"(recording=off)\n"
        )
        self._send_json(200, {"recording": False, "frames": cube_grid.frames})

    def _handle_cubes_reset(self) -> None:
        if cube_grid is None:
            self._send_text(409, "no cube grid initialised\n")
            return
        cube_grid.reset()
        sys.stderr.write(f"[{self.address_string()}] CUBES reset\n")
        self._send_json(200, {"reset": True, "frames": 0})

    def _handle_cubes_state(self) -> None:
        if cube_grid is None:
            self._send_json(200, {"ready": False, "recording": False, "cubes": []})
            return
        # Optional ?threshold=0.25&min=2 query overrides.
        threshold = 0.25
        min_obs = 2
        if "?" in self.path:
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            if "threshold" in qs:
                try:
                    threshold = max(0.0, min(1.0, float(qs["threshold"][0])))
                except ValueError:
                    pass
            if "min" in qs:
                try:
                    min_obs = max(1, int(qs["min"][0]))
                except ValueError:
                    pass
        try:
            payload = cube_grid.state(threshold=threshold, min_observations=min_obs)
        except Exception as e:  # noqa: BLE001
            self._send_text(500, f"{type(e).__name__}: {e}\n")
            return
        payload["ready"] = True
        payload["recording"] = cube_recording
        self._send_json(200, payload)

    def _handle_frame(self) -> None:
        global frame_count, DEFAULT_SESSION_ID
        body = self._read_body()
        try:
            frame = parse_frame(body)
        except Exception as e:  # noqa: BLE001
            self._send_text(400, f"frame parse error: {e}\n")
            return
        frame_count += 1
        ip = self.address_string()
        view = frame["viewMatrix"]
        pose_translation = (view[12], view[13], view[14])  # column-major: last column = translation

        # Resolve the recording session for this frame. New clients pass
        # ?session=<id>; legacy clients (no query) use the server's default
        # session, minted on first such frame.
        from urllib.parse import urlparse, parse_qs
        qs = parse_qs(urlparse(self.path).query)
        session_id = (qs.get("session") or [None])[0]
        if session_id is not None:
            if not SESSION_ID_RE.match(session_id):
                self._send_text(400, f"invalid session id: {session_id!r}\n")
                return
            session_id = _ensure_session(session_id)
        else:
            if DEFAULT_SESSION_ID is None:
                DEFAULT_SESSION_ID = _ensure_session()
                sys.stderr.write(f"[{ip}] minted default session {DEFAULT_SESSION_ID} "
                                 f"for legacy /frame post\n")
            session_id = DEFAULT_SESSION_ID
        idx = _next_frame_index(session_id)

        # Persist the raw body for offline replay + offline reconstruction.
        try:
            session_frames_dir = FRAMES_DIR / session_id / "frames"
            session_frames_dir.mkdir(parents=True, exist_ok=True)
            (session_frames_dir / f"frame_{idx:06d}.bin").write_bytes(body)
        except OSError as e:
            sys.stderr.write(f"[{ip}] frame-save failed: {e}\n")

        # Compute depth statistics so we can see immediately whether ARCore
        # is returning sane values (or mostly zeros = "no depth available").
        depth_summary = ""
        try:
            import numpy as np  # type: ignore
            if frame["format"] == 0:
                arr = np.frombuffer(frame["depth"], dtype=np.uint16).astype(np.float32)
            else:
                arr = np.frombuffer(frame["depth"], dtype=np.float32).astype(np.float32)
            metres = arr * float(frame["rawValueToMeters"])
            nonzero = metres[metres > 0]
            n_nonzero = int(nonzero.size)
            nonzero_pct = 100.0 * n_nonzero / max(1, metres.size)
            if n_nonzero:
                depth_summary = (
                    f" depth(nz={nonzero_pct:.0f}% min={nonzero.min():.2f} "
                    f"max={nonzero.max():.2f} med={float(np.median(nonzero)):.2f})"
                )
            else:
                depth_summary = " depth(all-zero)"
        except Exception as e:  # noqa: BLE001
            depth_summary = f" depth(stats-fail: {e})"

        ingest_summary = ""
        if voxel_room is not None:
            try:
                # ingest_frame returns (n_points_written, drift_check_or_None).
                # n_points_written = -1 when the frame was rejected as drifted.
                n_written, drift = voxel_room.ingest_frame(frame)
                stats = voxel_room.stats()
                if n_written < 0:
                    if drift and "jump_m" in drift:
                        ingest_summary = (
                            f" → REJECTED pose jump {drift['jump_m']*100:.0f}cm "
                            f"(rej_jump={stats['frames_rejected_jump']})"
                        )
                    else:
                        md = drift["median_dist"] if drift else float("nan")
                        mf = drift["match_fraction"] if drift else float("nan")
                        ingest_summary = (
                            f" → REJECTED drift {md*100:.1f}cm@{mf*100:.0f}% "
                            f"(rej_drift={stats['frames_rejected_drift']})"
                        )
                else:
                    drift_tag = ""
                    if drift is not None and "median_dist" in drift:
                        drift_tag = (
                            f" drift={drift['median_dist']*100:.1f}cm@{drift['match_fraction']*100:.0f}%"
                        )
                        if drift.get("correction_applied"):
                            t_mag = float((drift['t'] * drift['t']).sum() ** 0.5)
                            drift_tag += (
                                f" corr=t{t_mag*100:.1f}cm/r{drift['rot_deg']:.2f}°"
                                f" → {drift['median_dist_after']*100:.1f}cm"
                            )
                        elif drift["match_fraction"] >= voxel_room.DRIFT_MIN_MATCH_FRACTION:
                            # Correction was proposed but capped as too large
                            # — likely bad correspondences, not real drift.
                            t_mag = float((drift['t'] * drift['t']).sum() ** 0.5)
                            drift_tag += (
                                f" corr-skipped(t{t_mag*100:.1f}cm/r{drift['rot_deg']:.2f}°)"
                            )
                    ingest_summary = f" → wrote {n_written}pts{drift_tag}, total={stats['voxels']}vox in {stats['chunks']}ch"
            except Exception as e:  # noqa: BLE001
                self._send_text(500, f"{type(e).__name__}: {e}\n")
                return

        cube_summary = ""
        if cube_grid is not None and cube_recording:
            try:
                cs = cube_grid.ingest_frame(frame)
                cube_summary = (
                    f" cubes(vis={cs['visible']} occ+={cs['occupied']} free+={cs['free']})"
                )
            except Exception as e:  # noqa: BLE001
                cube_summary = f" cubes(err: {type(e).__name__}: {e})"

        color_summary = ""
        if frame["color"] is not None:
            color_summary = f" color={frame['color_width']}x{frame['color_height']}"
        else:
            color_summary = " color=none"

        sys.stderr.write(
            f"[{ip}] {session_id} #{idx:06d} {frame['width']}x{frame['height']} "
            f"fmt={frame['format']} rawToM={frame['rawValueToMeters']:.6f} "
            f"pose=({pose_translation[0]:+.2f},{pose_translation[1]:+.2f},{pose_translation[2]:+.2f})"
            f"{depth_summary}{color_summary}{ingest_summary}{cube_summary}\n"
        )
        self._send_json(200, {"ok": True, "session": session_id, "frame": idx})

    def _handle_sessions_list(self) -> None:
        """Enumerate session directories under captured_frames/ and report
        the frame count + which voxel JSON variants are present."""
        result = []
        try:
            if FRAMES_DIR.exists():
                for child in sorted(FRAMES_DIR.iterdir()):
                    if not child.is_dir():
                        continue
                    if not SESSION_ID_RE.match(child.name):
                        continue
                    frames_dir = child / "frames"
                    n_frames = 0
                    if frames_dir.exists():
                        n_frames = sum(
                            1 for p in frames_dir.iterdir()
                            if p.is_file() and p.name.startswith("frame_") and p.suffix == ".bin"
                        )
                    n_frames_refined = 0
                    refined_dir = child / "frames_refined"
                    if refined_dir.exists():
                        n_frames_refined = sum(
                            1 for p in refined_dir.iterdir()
                            if p.is_file() and p.name.startswith("frame_") and p.suffix == ".bin"
                        )
                    variants = []
                    frame_dirs = []
                    for p in child.iterdir():
                        if p.is_file():
                            m = SESSION_VOXELS_FILE_RE.match(p.name)
                            if m:
                                variants.append(m.group(1))
                        elif p.is_dir() and FRAME_VARIANT_RE.fullmatch(p.name):
                            frame_dirs.append(p.name)
                    variants.sort()
                    frame_dirs.sort()
                    result.append({
                        "id": child.name,
                        "n_frames": n_frames,
                        "n_frames_refined": n_frames_refined,
                        "variants": variants,
                        "frame_dirs": frame_dirs,
                    })
        except OSError as e:
            self._send_text(500, f"sessions enumeration failed: {e}\n")
            return
        self._send_json(200, {"sessions": result})

    def _handle_capture_static(self, path: str) -> None:
        """Dispatch /captures/<id>/… requests. The legacy route
        `voxels_<variant>.json` is served as a static file; the
        frame-debug endpoints (manifest, thumb, voxels) hit the helpers
        below."""
        from urllib.parse import urlparse, parse_qs
        purl = urlparse(self.path)
        path = purl.path or "/"

        # /captures/<id>/voxels_<variant>.json — kept for the voxel viewer.
        m = re.fullmatch(
            r"/captures/([A-Za-z0-9_\-]{1,64})/voxels_([A-Za-z0-9_\-]{1,32})\.json",
            path,
        )
        if m:
            session_id, variant = m.group(1), m.group(2)
            f = FRAMES_DIR / session_id / f"voxels_{variant}.json"
            if not f.exists() or not f.is_file():
                self.send_response(404); self.end_headers(); return
            try:
                body = f.read_bytes()
            except OSError as e:
                self._send_text(500, f"read failed: {e}\n"); return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        # /captures/<id>/features_meta.json[?variant=features|features_aligned|...]
        # — the lookup the JS uses to resolve voxel-click → frames + per-frame
        # keypoint UVs. For each `voxels_<v>.json` produced by
        # feature_ray_reconstruct.py there's a sibling `features_meta<suffix>`
        # where suffix derives from `<v>` ('features' → '', 'features_aligned'
        # → '_aligned', etc.). The query param picks which one.
        m = re.fullmatch(
            r"/captures/([A-Za-z0-9_\-]{1,64})/features_meta\.json", path,
        )
        if m:
            qs = parse_qs(purl.query)
            variant = (qs.get("variant") or ["features"])[0]
            f = FRAMES_DIR / m.group(1) / _features_meta_filename(variant)
            if not f.exists() or not f.is_file():
                self.send_response(404); self.end_headers(); return
            try:
                body = f.read_bytes()
            except OSError as e:
                self._send_text(500, f"read failed: {e}\n"); return
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)
            return

        # /captures/<id>/frame-manifest?variant=<vd>
        m = re.fullmatch(
            r"/captures/([A-Za-z0-9_\-]{1,64})/frame-manifest", path,
        )
        if m:
            qs = parse_qs(purl.query)
            variant_dir = (qs.get("variant") or ["frames"])[0]
            self._handle_frame_manifest(m.group(1), variant_dir)
            return

        # /captures/<id>/frame-thumb/<idx>.png?variant=<vd>
        #     &kind=color|depth|phone|model|diff|blend
        #     &long_edge=<n>
        #     &sigma=<frac>          (only for kind=blend)
        m = re.fullmatch(
            r"/captures/([A-Za-z0-9_\-]{1,64})/frame-thumb/(\d+)\.png", path,
        )
        if m:
            qs = parse_qs(purl.query)
            variant_dir = (qs.get("variant") or ["frames"])[0]
            kind = (qs.get("kind") or ["color"])[0]
            try:
                long_edge = int((qs.get("long_edge") or ["600"])[0])
            except ValueError:
                long_edge = 600
            long_edge = max(64, min(2400, long_edge))
            try:
                sigma_frac = float((qs.get("sigma") or ["0.03"])[0])
            except ValueError:
                sigma_frac = 0.03
            sigma_frac = max(0.001, min(0.20, sigma_frac))
            self._handle_frame_thumb(m.group(1), variant_dir, int(m.group(2)),
                                       kind, long_edge=long_edge,
                                       sigma_frac=sigma_frac)
            return

        # /captures/<id>/frame-voxels/<idx>.json?variant=<vd>
        #     &voxel_size=<f>&world_min=x,y,z&shape=Nx,Ny,Nz
        m = re.fullmatch(
            r"/captures/([A-Za-z0-9_\-]{1,64})/frame-voxels/(\d+)\.json", path,
        )
        if m:
            qs = parse_qs(purl.query)
            variant_dir = (qs.get("variant") or ["frames"])[0]
            voxel_size = float((qs.get("voxel_size") or ["0.05"])[0])
            world_min  = [float(x) for x in (qs.get("world_min") or ["-2.5,-0.3,-2.5"])[0].split(",")]
            shape      = [int(x)   for x in (qs.get("shape")     or ["100,100,100"])[0].split(",")]
            self._handle_frame_voxels(
                m.group(1), variant_dir, int(m.group(2)),
                voxel_size, world_min, shape,
            )
            return

        # /captures/<id>/depth-scatter/<idx>.json?variant=<vd>
        #     &max_samples=<n>&sigma=<frac>
        # — paired depth samples on the colour-image grid: both
        # (phone, model_raw) and (phone, blend) plus Pearson + Spearman
        # for each. Sigma is the Gaussian blur fraction used by the blend.
        m = re.fullmatch(
            r"/captures/([A-Za-z0-9_\-]{1,64})/depth-scatter/(\d+)\.json",
            path,
        )
        if m:
            qs = parse_qs(purl.query)
            variant_dir = (qs.get("variant") or ["frames"])[0]
            try:
                max_samples = int((qs.get("max_samples") or ["5000"])[0])
            except ValueError:
                max_samples = 5000
            max_samples = max(100, min(50000, max_samples))
            try:
                sigma_frac = float((qs.get("sigma") or ["0.03"])[0])
            except ValueError:
                sigma_frac = 0.03
            sigma_frac = max(0.001, min(0.20, sigma_frac))
            self._handle_depth_scatter(m.group(1), variant_dir, int(m.group(2)),
                                        max_samples=max_samples,
                                        sigma_frac=sigma_frac)
            return

        # /captures/<id>/frame-feature-voxels/<idx>.json[?variant=features|features_aligned|...]
        # — voxels for features observed in this frame (uses the right
        # features_meta<suffix>.json based on which variant the viewer is on).
        m = re.fullmatch(
            r"/captures/([A-Za-z0-9_\-]{1,64})/frame-feature-voxels/(\d+)\.json",
            path,
        )
        if m:
            qs = parse_qs(purl.query)
            variant = (qs.get("variant") or ["features"])[0]
            self._handle_frame_feature_voxels(m.group(1), int(m.group(2)), variant)
            return

        # /captures/<id>/pixel-cloud-status — reports whether the model_raw
        # cache directory exists for this session and which frame indices
        # are covered. The voxelview uses this to disable the model-depth
        # toggle (and tell the user to run cache_model_raw.py) when the
        # cache hasn't been built.
        m = re.fullmatch(
            r"/captures/([A-Za-z0-9_\-]{1,64})/pixel-cloud-status", path,
        )
        if m:
            self._handle_pixel_cloud_status(m.group(1))
            return

        # /captures/<id>/common-features?frames=1,17,33&pose_dir=frames
        # — features observed in every one of `frames`, with per-frame
        # (u, v) and BA-triangulated world position. The features_meta
        # sidecar is selected to match `pose_dir` (raw → features_meta.json,
        # frames_aligned → features_meta_aligned.json, etc.); without a
        # matching sidecar the response is 404 so the JS can tell the user
        # to run feature_ray_reconstruct.py for that pose set.
        m = re.fullmatch(
            r"/captures/([A-Za-z0-9_\-]{1,64})/common-features", path,
        )
        if m:
            qs = parse_qs(purl.query)
            pose_dir = (qs.get("pose_dir") or ["frames"])[0]
            frames_csv = (qs.get("frames") or [""])[0]
            self._handle_common_features(m.group(1), pose_dir, frames_csv)
            return

        # /captures/<id>/pixel-cloud-autotune?frames=1,17,33&pose_dir=frames
        #     &fit_space=depth|disparity
        # — for each frame in `frames`, fits an affine (a, b) so that the
        # camera-Z depth implied by the cached model_raw at each common
        # feature's pixel matches the BA-triangulated world point's
        # camera-Z. Returns per-frame {a, b, n_features, residual}.
        m = re.fullmatch(
            r"/captures/([A-Za-z0-9_\-]{1,64})/pixel-cloud-autotune", path,
        )
        if m:
            qs = parse_qs(purl.query)
            pose_dir   = (qs.get("pose_dir")   or ["frames"])[0]
            fit_space  = (qs.get("fit_space")  or ["depth"])[0]
            frames_csv = (qs.get("frames") or [""])[0]
            self._handle_pixel_cloud_autotune(
                m.group(1), pose_dir, fit_space, frames_csv,
            )
            return

        # /captures/<id>/pixel-cloud-autotune-voxel?frames=…&pose_dir=…
        #     &fit_space=depth|disparity&voxel_sizes=0.30,0.15,0.05
        #     &init=15:1.0,-0.2;16:0.8,-0.1&stride=8
        # — feature-free auto-tune. For each frame, fits (a, b) by
        # MAXIMISING the pairwise voxel-overlap count between the
        # frames' projected pixel clouds. Coarse-to-fine voxel sizes
        # smooth out the otherwise-piecewise-constant objective; Powell
        # is used at each stage. Camera origins anchor the world frame
        # so there is no gauge ambiguity to break.
        m = re.fullmatch(
            r"/captures/([A-Za-z0-9_\-]{1,64})/pixel-cloud-autotune-voxel",
            path,
        )
        if m:
            qs = parse_qs(purl.query)
            pose_dir   = (qs.get("pose_dir")    or ["frames"])[0]
            fit_space  = (qs.get("fit_space")   or ["depth"])[0]
            frames_csv = (qs.get("frames")      or [""])[0]
            voxel_csv  = (qs.get("voxel_sizes") or ["0.30,0.15,0.05"])[0]
            init_csv   = (qs.get("init")        or [""])[0]
            try:
                stride = max(1, int((qs.get("stride") or ["8"])[0]))
            except ValueError:
                stride = 8
            self._handle_pixel_cloud_autotune_voxel(
                m.group(1), pose_dir, fit_space, frames_csv,
                voxel_csv, init_csv, stride,
            )
            return

        # /captures/<id>/frame-depth-at?frames=15,16,17&us=0.5,0.51,0.52
        #     &vs=0.4,0.4,0.4&kind=phone|model
        # — Bilinear-sample either depth source at the given norm-view
        # UVs in each frame. Lengths of frames/us/vs must match (each
        # entry is one (frame, u, v) sample). Used by the triplet
        # editor to read off depth values at the user's clicked points.
        m = re.fullmatch(
            r"/captures/([A-Za-z0-9_\-]{1,64})/frame-depth-at", path,
        )
        if m:
            qs = parse_qs(purl.query)
            self._handle_frame_depth_at(
                m.group(1),
                kind=(qs.get("kind") or ["phone"])[0],
                pose_dir=(qs.get("pose_dir") or ["frames"])[0],
                frames_csv=(qs.get("frames") or [""])[0],
                us_csv=(qs.get("us") or [""])[0],
                vs_csv=(qs.get("vs") or [""])[0],
            )
            return

        # /captures/<id>/triplet-distances?pose_dir=…
        #     &features=<URL-encoded JSON list>
        # — For each feature in the JSON list (each with `marks: [{frame, u, v}, ...]`)
        # compute an LS-triangulated 3D world point from the rays
        # defined by the user's clicks. For every mark, return the
        # model-depth and phone-depth at the (u, v) plus the camera-Z
        # perpendicular distance from that frame's camera to the
        # triangulated point. Used by the triplet editor's plots.
        m = re.fullmatch(
            r"/captures/([A-Za-z0-9_\-]{1,64})/triplet-distances", path,
        )
        if m:
            qs = parse_qs(purl.query)
            pose_dir = (qs.get("pose_dir") or ["frames"])[0]
            features_json = (qs.get("features") or ["[]"])[0]
            try:
                sigma_frac = float((qs.get("sigma") or ["0.03"])[0])
            except ValueError:
                sigma_frac = 0.03
            sigma_frac = max(0.001, min(0.20, sigma_frac))
            self._handle_triplet_distances(
                m.group(1), pose_dir, features_json,
                sigma_frac=sigma_frac,
            )
            return

        # /captures/<id>/snap-feature?ref_frame=15&ref_u=…&ref_v=…
        #     &target_frame=16&init_u=…&init_v=…&patch=0.025&radius=0.06
        # — Refine `init_u, init_v` on `target_frame` so the local
        # patch best matches a same-size patch on `ref_frame` at
        # `ref_u, ref_v`. Returns refined UV + the NCC score.
        m = re.fullmatch(
            r"/captures/([A-Za-z0-9_\-]{1,64})/snap-feature", path,
        )
        if m:
            qs = parse_qs(purl.query)
            try:
                self._handle_snap_feature(
                    m.group(1),
                    ref_frame=int((qs.get("ref_frame") or ["0"])[0]),
                    target_frame=int((qs.get("target_frame") or ["0"])[0]),
                    ref_u=float((qs.get("ref_u") or ["0.5"])[0]),
                    ref_v=float((qs.get("ref_v") or ["0.5"])[0]),
                    init_u=float((qs.get("init_u") or ["0.5"])[0]),
                    init_v=float((qs.get("init_v") or ["0.5"])[0]),
                    patch=float((qs.get("patch") or ["0.025"])[0]),
                    radius=float((qs.get("radius") or ["0.06"])[0]),
                    pose_dir=(qs.get("pose_dir") or ["frames"])[0],
                )
            except ValueError as e:
                self._send_text(400, f"bad query: {e}\n")
            return

        # /captures/<id>/pixel-cloud-autotune-chamfer?frames=…&pose_dir=…
        #     &fit_space=depth|disparity&thresholds=0.30,0.05
        #     &init=15:1.0,-0.2;16:0.8,-0.1&stride=16
        # — smooth-surrogate auto-tune. Minimises the symmetric
        # truncated-Chamfer distance between every pair of frames'
        # point clouds. Pixels with no nearest neighbour inside the
        # threshold contribute a constant (the threshold²), so partial-
        # overlap regions don't pull the optimiser around. Threshold
        # is annealed coarse-to-fine.
        m = re.fullmatch(
            r"/captures/([A-Za-z0-9_\-]{1,64})/pixel-cloud-autotune-chamfer",
            path,
        )
        if m:
            qs = parse_qs(purl.query)
            pose_dir    = (qs.get("pose_dir")   or ["frames"])[0]
            fit_space   = (qs.get("fit_space")  or ["depth"])[0]
            frames_csv  = (qs.get("frames")     or [""])[0]
            thresh_csv  = (qs.get("thresholds") or ["0.30,0.05"])[0]
            init_csv    = (qs.get("init")       or [""])[0]
            try:
                stride = max(1, int((qs.get("stride") or ["16"])[0]))
            except ValueError:
                stride = 16
            self._handle_pixel_cloud_autotune_chamfer(
                m.group(1), pose_dir, fit_space, frames_csv,
                thresh_csv, init_csv, stride,
            )
            return

        # /captures/<id>/pixel-cloud/<idx>.json — per-pixel rays + depths
        # for one frame, ready for client-side slider-driven affine application.
        # Query params:
        #   pose_dir   = frames | frames_aligned | frames_feature_ba_*    (which
        #                frame.bin to read for V/P/Bd matrices)
        #   depth_kind = phone | model | blend
        #   stride     = pixel skip on the source grid (default 4 for model
        #                and blend, 1 for phone)
        #   near, far  = depth window for client-side filtering hint
        #   sigma      = blend Gaussian (fraction of the colour-image diagonal,
        #                only used for depth_kind=blend; default 0.03 = 3%)
        m = re.fullmatch(
            r"/captures/([A-Za-z0-9_\-]{1,64})/pixel-cloud/(\d+)\.json", path,
        )
        if m:
            qs = parse_qs(purl.query)
            pose_dir   = (qs.get("pose_dir")   or ["frames"])[0]
            depth_kind = (qs.get("depth_kind") or ["phone"])[0]
            try:
                stride = max(1, int((qs.get("stride") or ["0"])[0]))
            except ValueError:
                stride = 0
            try:
                near = float((qs.get("near") or ["0.05"])[0])
                far  = float((qs.get("far")  or ["8.0"])[0])
            except ValueError:
                near, far = 0.05, 8.0
            try:
                sigma_frac = float((qs.get("sigma") or ["0.03"])[0])
            except ValueError:
                sigma_frac = 0.03
            sigma_frac = max(0.001, min(0.20, sigma_frac))
            self._handle_pixel_cloud(
                m.group(1), int(m.group(2)),
                pose_dir, depth_kind, stride, near, far,
                sigma_frac=sigma_frac,
            )
            return

        self.send_response(404); self.end_headers()

    def _handle_frame_manifest(self, session_id: str, variant_dir: str) -> None:
        if not SESSION_ID_RE.match(session_id) or not FRAME_VARIANT_RE.fullmatch(variant_dir):
            self.send_response(404); self.end_headers(); return
        d = FRAMES_DIR / session_id / variant_dir
        if not d.is_dir():
            self.send_response(404); self.end_headers(); return
        frames = []
        try:
            for fp in sorted(d.glob("frame_*.bin")):
                m = re.match(r"frame_(\d+)\.bin", fp.name)
                if not m: continue
                idx = int(m.group(1))
                try:
                    with fp.open("rb") as f:
                        head = f.read(FRAME_HEADER_SIZE)
                except OSError:
                    continue
                if len(head) < FRAME_HEADER_SIZE: continue
                # First 16 floats of the header are viewMatrix (column-major
                # float32). The 4th column is the camera origin in world.
                view = struct.unpack("<16f", head[:64])
                pose = [float(view[12]), float(view[13]), float(view[14])]
                # Forward direction = -view's z column rotated to world.
                # For column-major storage, column 2 of V is at indices 8,9,10
                # — that's the view-space +z direction expressed in world. The
                # camera looks down -z in view, so world-space forward = -col2.
                fwd = [-float(view[8]), -float(view[9]), -float(view[10])]
                # Read enough of the trailing fields for depth/colour dims.
                tail = struct.unpack("<I I f I I I I I", head[192:224])
                dw, dh, _raw, _dfmt, cw, ch_, _cfmt, _clen = tail
                frames.append({
                    "idx": idx,
                    "pose": pose,
                    "forward": fwd,
                    "depth": [int(dw), int(dh)],
                    "color": [int(cw), int(ch_)],
                })
        except OSError as e:
            self._send_text(500, f"manifest scan failed: {e}\n"); return
        self._send_json(200, {"session": session_id, "variant": variant_dir, "frames": frames})

    def _handle_frame_thumb(self, session_id: str, variant_dir: str,
                             idx: int, kind: str, *, long_edge: int = 600,
                             sigma_frac: float = 0.03) -> None:
        if not SESSION_ID_RE.match(session_id) or not FRAME_VARIANT_RE.fullmatch(variant_dir):
            self.send_response(404); self.end_headers(); return
        if kind not in ("color", "depth", "phone", "model", "diff", "blend"):
            self.send_response(404); self.end_headers(); return
        # Cache by long_edge as well so the triplet (full-res) and the
        # voxelview panel (600 px) thumbs don't evict each other.
        # For kind=blend, sigma is also part of the key but quantised so
        # a stream of slider ticks doesn't bloat the cache.
        if kind == "blend":
            sigma_q = round(float(sigma_frac), 3)
            cache_key = (session_id, variant_dir, idx, kind, long_edge, sigma_q)
        else:
            cache_key = (session_id, variant_dir, idx, kind, long_edge)
        png = _THUMB_CACHE.get(cache_key)
        if png is None:
            f = FRAMES_DIR / session_id / variant_dir / f"frame_{idx:06d}.bin"
            if not f.exists():
                self.send_response(404); self.end_headers(); return
            try:
                body = f.read_bytes()
                if kind == "color":
                    png = _render_color_thumb(body, size=long_edge)
                elif kind == "depth":
                    png = _render_depth_thumb(body, size=long_edge)
                elif kind == "phone":
                    png = _render_phone_color_thumb(body, size=long_edge)
                else:  # model / diff / blend — all need the cached prediction
                    arr = _load_model_raw_array(session_id, idx)
                    if arr is None:
                        self._send_text(409,
                            "model_raw cache missing — run "
                            f"tools/cache_model_raw.py --session {session_id}\n")
                        return
                    if kind == "model":
                        png = _render_model_color_thumb(body, arr, size=long_edge)
                    elif kind == "diff":
                        png = _render_diff_color_thumb(body, arr, size=long_edge)
                    else:
                        png = _render_blend_color_thumb(
                            body, arr, size=long_edge, sigma_frac=sigma_frac,
                        )
            except Exception as e:  # noqa: BLE001
                self._send_text(500, f"thumb render failed: {e}\n"); return
            if png is None:
                self.send_response(404); self.end_headers(); return
            if len(_THUMB_CACHE) >= _THUMB_CACHE_MAX:
                # Pop an arbitrary entry to bound memory; don't bother with LRU
                # since the panel's working set is small.
                _THUMB_CACHE.pop(next(iter(_THUMB_CACHE)))
            _THUMB_CACHE[cache_key] = png
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(png)))
        self.send_header("Cache-Control", "public, max-age=86400")
        self.end_headers()
        self.wfile.write(png)

    def _handle_depth_scatter(self, session_id: str, variant_dir: str,
                                idx: int, *, max_samples: int,
                                sigma_frac: float) -> None:
        """Per-pixel paired depth samples for one frame, on the colour-image
        grid: both (phone, model_raw) and (phone, blend_metres) pairs plus
        Pearson + Spearman computed over ALL valid pixels (the wire
        sampling is just a per-axis subset for plotting).

        The blend uses the same hole-aware Gaussian detail-injection as the
        kind=blend thumbnail, so on-screen the scatter and the picture line
        up. `sigma_frac` is the Gaussian sigma as a fraction of the
        sampling grid's diagonal."""
        import math
        import numpy as np
        if not SESSION_ID_RE.match(session_id) or not FRAME_VARIANT_RE.fullmatch(variant_dir):
            self.send_response(404); self.end_headers(); return
        f = FRAMES_DIR / session_id / variant_dir / f"frame_{idx:06d}.bin"
        if not f.exists():
            self._send_text(404, f"frame missing: {variant_dir}/frame_{idx:06d}.bin\n")
            return
        arr = _load_model_raw_array(session_id, idx)
        if arr is None:
            self._send_text(409,
                "model_raw cache missing — run "
                f"tools/cache_model_raw.py --session {session_id}\n")
            return
        try:
            body = f.read_bytes()
            frame = parse_frame(body)
            cw = int(frame["color_width"]); ch = int(frame["color_height"])
            if cw == 0 or ch == 0:
                # No colour buffer → fall back to the depth-buffer dims so the
                # grid sampling still has something sensible to chew on.
                cw = int(frame["width"]); ch = int(frame["height"])
            target = 200_000
            stride = max(1, int(round(((cw * ch) / target) ** 0.5)))
            out_w = max(1, cw // stride); out_h = max(1, ch // stride)
            phone, model = _sample_phone_model_on_color_grid(
                body, out_w=out_w, out_h=out_h, model_raw_arr=arr,
            )
            sigma_px = max(1.0, float(sigma_frac) * math.hypot(out_w, out_h))
            blend, model_metres, fit_a, fit_b, _ = _compute_blend_metres(
                phone, model, sigma_px,
            )
        except Exception as e:  # noqa: BLE001
            self._send_text(500, f"scatter sample failed: {e}\n")
            return

        valid_pm = np.isfinite(phone) & np.isfinite(model)
        valid_pb = np.isfinite(phone) & np.isfinite(blend)
        n_pm = int(valid_pm.sum()); n_pb = int(valid_pb.sum())
        p_for_m = phone[valid_pm].astype(np.float64)
        m_arr   = model[valid_pm].astype(np.float64)
        p_for_b = phone[valid_pb].astype(np.float64)
        b_arr   = blend[valid_pb].astype(np.float64)

        def corr_pair(x, y):
            if x.size < 3 or float(x.std()) < 1e-9 or float(y.std()) < 1e-9:
                return None, None
            r = float(np.corrcoef(x, y)[0, 1])
            rho = float(np.corrcoef(_ranks_with_ties(x), _ranks_with_ties(y))[0, 1])
            return r, rho

        pearson_m, spearman_m = corr_pair(p_for_m, m_arr)
        pearson_b, spearman_b = corr_pair(p_for_b, b_arr)

        rng = np.random.default_rng(0xC0FFEE)
        def subsample_pairs(x, y):
            n = x.size
            if n > max_samples:
                sel = rng.choice(n, size=max_samples, replace=False)
                x = x[sel]; y = y[sel]
            return [[round(float(a), 4), round(float(b), 4)]
                    for a, b in zip(x, y)]

        pairs_model = subsample_pairs(p_for_m, m_arr)
        pairs_blend = subsample_pairs(p_for_b, b_arr)

        self._send_json(200, {
            "session": session_id,
            "variant": variant_dir,
            "idx": idx,
            "color_size": [cw, ch],
            "grid_size": [int(out_w), int(out_h)],
            "stride": int(stride),
            "sigma": float(sigma_frac),
            "fit": {"a": float(fit_a), "b": float(fit_b)},
            "n_valid_model": n_pm, "n_valid_blend": n_pb,
            "pearson_model":  pearson_m,  "spearman_model":  spearman_m,
            "pearson_blend":  pearson_b,  "spearman_blend":  spearman_b,
            "phone_min": float(p_for_m.min()) if p_for_m.size else None,
            "phone_max": float(p_for_m.max()) if p_for_m.size else None,
            "model_min": float(m_arr.min()) if m_arr.size else None,
            "model_max": float(m_arr.max()) if m_arr.size else None,
            "blend_min": float(b_arr.min()) if b_arr.size else None,
            "blend_max": float(b_arr.max()) if b_arr.size else None,
            "pairs_model": pairs_model,
            "pairs_blend": pairs_blend,
        })

    def _handle_frame_feature_voxels(self, session_id: str, idx: int,
                                      variant: str = "features") -> None:
        """Return the voxel triples whose features were observed in frame
        `idx`, plus the camera pose + frustum so the viewer can draw the
        wireframe pyramid. `variant` selects which features_meta file to
        read — `features` → features_meta.json, `features_aligned` →
        features_meta_aligned.json, etc. Falls back gracefully if either
        the meta file or the underlying frame is missing."""
        if not SESSION_ID_RE.match(session_id):
            self.send_response(404); self.end_headers(); return
        sess_dir = FRAMES_DIR / session_id
        try:
            meta_name = _features_meta_filename(variant)
        except ValueError:
            self.send_response(404); self.end_headers(); return
        meta_path = sess_dir / meta_name
        if not meta_path.exists():
            self._send_text(404, f"{meta_name} not found — "
                                  "run feature_ray_reconstruct.py --session "
                                  f"{session_id} [--frames-variant frames_aligned]\n")
            return
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            self._send_text(500, f"features_meta read failed: {e}\n"); return

        triples = []
        for v in meta.get("voxels", []):
            if idx in v.get("frames", ()):
                triples.append(v["idx"])

        # Read the source frame for pose + frustum so the wireframe lines up
        # with the depth-derived overlay.
        f = sess_dir / "frames" / f"frame_{idx:06d}.bin"
        pose: list = []
        frustum: list = []
        if f.exists():
            try:
                import numpy as np
                sys.path.insert(0, str(PROJECT_ROOT / "tools"))
                import fusion  # noqa: E402
                frame = parse_frame(f.read_bytes())
                V = fusion._mat4_from_column_major(frame["viewMatrix"])
                P = fusion._mat4_from_column_major(frame["projectionMatrix"])
                P_inv = np.linalg.inv(P)
                cam = V[:3, 3].astype(np.float64)
                pose = [float(cam[0]), float(cam[1]), float(cam[2])]
                frustum = _frustum_corners(P_inv, V, near=0.1, far=2.0)
            except Exception as e:  # noqa: BLE001
                self._send_text(500, f"frame parse failed: {e}\n"); return

        self._send_json(200, {
            "idx": idx,
            "pose": pose,
            "frustum_world": frustum,
            "indices": triples,
        })

    def _handle_frame_voxels(self, session_id: str, variant_dir: str, idx: int,
                              voxel_size: float, world_min: list,
                              shape: list) -> None:
        if not SESSION_ID_RE.match(session_id) or not FRAME_VARIANT_RE.fullmatch(variant_dir):
            self.send_response(404); self.end_headers(); return
        if voxel_size <= 0 or voxel_size > 10 or len(world_min) != 3 or len(shape) != 3:
            self._send_text(400, "bad grid params\n"); return
        f = FRAMES_DIR / session_id / variant_dir / f"frame_{idx:06d}.bin"
        if not f.exists():
            self.send_response(404); self.end_headers(); return
        try:
            indices, pose, frustum_corners = _frame_to_voxel_indices(
                f.read_bytes(), voxel_size, world_min, shape,
            )
        except Exception as e:  # noqa: BLE001
            self._send_text(500, f"projection failed: {e}\n"); return
        self._send_json(200, {
            "idx": idx,
            "pose": pose,
            "frustum_world": frustum_corners,
            "indices": indices,
        })

    def _handle_pixel_cloud_status(self, session_id: str) -> None:
        """Report whether captured_frames/<id>/model_raw/index.json is on
        disk, plus the list of cached frame indices and the (cw, ch) of
        each — the client uses this to gate the 'model depth' toggle and
        to know each frame's color-grid resolution."""
        if not SESSION_ID_RE.match(session_id):
            self.send_response(404); self.end_headers(); return
        sess_dir = FRAMES_DIR / session_id
        idx_path = sess_dir / "model_raw" / "index.json"
        if not idx_path.exists():
            self._send_json(200, {"ready": False, "frames": {}})
            return
        try:
            payload = json.loads(idx_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            self._send_text(500, f"index.json read failed: {e}\n"); return
        self._send_json(200, {"ready": True, "frames": payload})

    def _handle_pixel_cloud(self, session_id: str, idx: int,
                            pose_dir: str, depth_kind: str,
                            stride: int, near: float, far: float,
                            *, sigma_frac: float = 0.03) -> None:
        if not SESSION_ID_RE.match(session_id) or not FRAME_VARIANT_RE.fullmatch(pose_dir):
            self.send_response(404); self.end_headers(); return
        if depth_kind not in ("phone", "model", "blend"):
            self._send_text(400, f"bad depth_kind {depth_kind!r}\n"); return
        f = FRAMES_DIR / session_id / pose_dir / f"frame_{idx:06d}.bin"
        if not f.exists():
            self._send_text(404, f"missing {pose_dir}/frame_{idx:06d}.bin\n"); return
        # Model-raw cache (needed for depth_kind=model and blend).
        model_raw_path = FRAMES_DIR / session_id / "model_raw" / f"frame_{idx:06d}.f16"
        model_raw_meta = FRAMES_DIR / session_id / "model_raw" / "index.json"
        needs_model = depth_kind in ("model", "blend")
        try:
            payload = _build_pixel_cloud_payload(
                body=f.read_bytes(),
                depth_kind=depth_kind,
                stride=stride,
                near=near,
                far=far,
                model_raw_path=model_raw_path if needs_model else None,
                model_raw_meta=model_raw_meta if needs_model else None,
                idx=idx,
                sigma_frac=sigma_frac,
            )
        except FileNotFoundError as e:
            self._send_text(409, f"{e}\n"); return
        except Exception as e:  # noqa: BLE001
            self._send_text(500, f"pixel-cloud failed: {type(e).__name__}: {e}\n"); return
        self._send_json(200, payload)

    def _handle_pixel_cloud_autotune_chamfer(self, session_id: str,
                                              pose_dir: str, fit_space: str,
                                              frames_csv: str,
                                              thresh_csv: str, init_csv: str,
                                              stride: int) -> None:
        if not SESSION_ID_RE.match(session_id) or not FRAME_VARIANT_RE.fullmatch(pose_dir):
            self.send_response(404); self.end_headers(); return
        if fit_space not in ("depth", "disparity"):
            self._send_text(400, f"bad fit_space {fit_space!r}\n"); return
        frames = sorted(_parse_int_csv(frames_csv))
        if len(frames) < 2:
            self._send_text(400, "need at least 2 frames for chamfer auto-tune\n"); return
        try:
            thresholds = [float(x) for x in thresh_csv.split(",") if x.strip()]
        except ValueError:
            self._send_text(400, f"bad thresholds={thresh_csv!r}\n"); return
        if not thresholds or any(t <= 0 or t > 5 for t in thresholds):
            self._send_text(400, "thresholds must be a comma list of positive metres\n"); return
        init_map: dict[int, tuple[float, float]] = {}
        for part in (init_csv or "").split(";"):
            part = part.strip()
            if not part: continue
            try:
                key, vals = part.split(":")
                a_s, b_s = vals.split(",")
                init_map[int(key)] = (float(a_s), float(b_s))
            except ValueError:
                continue

        model_meta_path = FRAMES_DIR / session_id / "model_raw" / "index.json"
        if not model_meta_path.exists():
            self._send_text(409, "model_raw cache missing — run cache_model_raw.py\n"); return
        try:
            model_meta = json.loads(model_meta_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            self._send_text(500, f"model_raw/index.json read failed: {e}\n"); return
        try:
            result = _autotune_chamfer(
                session_id=session_id, pose_dir=pose_dir,
                frames=frames, fit_space=fit_space,
                thresholds=thresholds, init_map=init_map,
                stride=stride, model_meta=model_meta,
            )
        except FileNotFoundError as e:
            self._send_text(409, f"{e}\n"); return
        except Exception as e:  # noqa: BLE001
            self._send_text(500, f"autotune-chamfer failed: {type(e).__name__}: {e}\n"); return
        self._send_json(200, result)

    def _handle_frame_depth_at(self, session_id: str, *,
                                kind: str, pose_dir: str,
                                frames_csv: str, us_csv: str, vs_csv: str) -> None:
        if not SESSION_ID_RE.match(session_id) or not FRAME_VARIANT_RE.fullmatch(pose_dir):
            self.send_response(404); self.end_headers(); return
        if kind not in ("phone", "model"):
            self._send_text(400, f"bad kind {kind!r}\n"); return
        # Parse parallel arrays. We accept different lengths for frames vs
        # (us, vs) only if frames has length 1 (broadcast over all UVs);
        # otherwise lengths must match.
        try:
            frames = [int(x) for x in frames_csv.split(",") if x.strip()]
            us     = [float(x) for x in us_csv.split(",") if x.strip()]
            vs     = [float(x) for x in vs_csv.split(",") if x.strip()]
        except ValueError as e:
            self._send_text(400, f"bad numeric in query: {e}\n"); return
        if len(us) != len(vs):
            self._send_text(400, "us and vs must have equal length\n"); return
        if not frames or not us:
            self._send_text(400, "frames=, us=, vs= are required\n"); return
        if len(frames) == 1:
            frames = frames * len(us)
        elif len(frames) != len(us):
            self._send_text(400, "frames must be 1 or len(us)\n"); return

        try:
            depths = _sample_depth_at(
                session_id=session_id, pose_dir=pose_dir, kind=kind,
                frames=frames, us=us, vs=vs,
            )
        except FileNotFoundError as e:
            self._send_text(409, f"{e}\n"); return
        except Exception as e:  # noqa: BLE001
            self._send_text(500, f"depth-at failed: {type(e).__name__}: {e}\n"); return
        self._send_json(200, {
            "session": session_id,
            "kind": kind,
            "pose_dir": pose_dir,
            "depths": depths,   # list of float|null in input order
        })

    def _handle_triplet_distances(self, session_id: str,
                                   pose_dir: str, features_json: str,
                                   *, sigma_frac: float = 0.03) -> None:
        if not SESSION_ID_RE.match(session_id) or not FRAME_VARIANT_RE.fullmatch(pose_dir):
            self.send_response(404); self.end_headers(); return
        try:
            features_in = json.loads(features_json)
        except json.JSONDecodeError as e:
            self._send_text(400, f"bad features JSON: {e}\n"); return
        if not isinstance(features_in, list):
            self._send_text(400, "features must be a list\n"); return

        try:
            payload = _triplet_distances_payload(
                session_id=session_id,
                pose_dir=pose_dir,
                features_in=features_in,
                sigma_frac=sigma_frac,
            )
        except FileNotFoundError as e:
            self._send_text(409, f"{e}\n"); return
        except Exception as e:  # noqa: BLE001
            self._send_text(500, f"triplet-distances failed: {type(e).__name__}: {e}\n"); return
        self._send_json(200, payload)

    def _handle_snap_feature(self, session_id: str, *,
                              ref_frame: int, target_frame: int,
                              ref_u: float, ref_v: float,
                              init_u: float, init_v: float,
                              patch: float, radius: float,
                              pose_dir: str) -> None:
        if not SESSION_ID_RE.match(session_id) or not FRAME_VARIANT_RE.fullmatch(pose_dir):
            self.send_response(404); self.end_headers(); return
        # Sanity-check params; everything in [0,1]² for UVs,
        # patch ∈ (0, 0.2], radius ∈ (0, 0.5].
        for name, val, lo, hi in [("ref_u", ref_u, 0, 1), ("ref_v", ref_v, 0, 1),
                                   ("init_u", init_u, 0, 1), ("init_v", init_v, 0, 1),
                                   ("patch", patch, 1e-3, 0.2),
                                   ("radius", radius, 1e-3, 0.5)]:
            if not (lo <= val <= hi):
                self._send_text(400, f"{name}={val} out of [{lo}, {hi}]\n"); return
        ref_path = FRAMES_DIR / session_id / pose_dir / f"frame_{ref_frame:06d}.bin"
        tgt_path = FRAMES_DIR / session_id / pose_dir / f"frame_{target_frame:06d}.bin"
        if not ref_path.exists() or not tgt_path.exists():
            self._send_text(404, "missing frame.bin for ref or target\n"); return
        try:
            refined = _snap_feature_match(
                ref_body=ref_path.read_bytes(),
                tgt_body=tgt_path.read_bytes(),
                ref_uv=(ref_u, ref_v),
                init_uv=(init_u, init_v),
                patch_norm=patch,
                radius_norm=radius,
            )
        except Exception as e:  # noqa: BLE001
            self._send_text(500, f"snap failed: {type(e).__name__}: {e}\n"); return
        self._send_json(200, refined)

    def _handle_pixel_cloud_autotune_voxel(self, session_id: str,
                                            pose_dir: str, fit_space: str,
                                            frames_csv: str,
                                            voxel_csv: str, init_csv: str,
                                            stride: int) -> None:
        if not SESSION_ID_RE.match(session_id) or not FRAME_VARIANT_RE.fullmatch(pose_dir):
            self.send_response(404); self.end_headers(); return
        if fit_space not in ("depth", "disparity"):
            self._send_text(400, f"bad fit_space {fit_space!r}\n"); return
        frames = sorted(_parse_int_csv(frames_csv))
        if len(frames) < 2:
            self._send_text(400, "need at least 2 frames for voxel-overlap auto-tune\n")
            return
        try:
            voxel_sizes = [float(x) for x in voxel_csv.split(",") if x.strip()]
        except ValueError:
            self._send_text(400, f"bad voxel_sizes={voxel_csv!r}\n"); return
        if not voxel_sizes or any(s <= 0 or s > 5 for s in voxel_sizes):
            self._send_text(400, "voxel_sizes must be a comma list of positive metres\n"); return
        # Decode optional init: "idx:a,b;idx:a,b"  →  {idx: (a, b)}
        init_map: dict[int, tuple[float, float]] = {}
        for part in (init_csv or "").split(";"):
            part = part.strip()
            if not part:
                continue
            try:
                key, vals = part.split(":")
                a_s, b_s = vals.split(",")
                init_map[int(key)] = (float(a_s), float(b_s))
            except ValueError:
                continue

        model_meta_path = FRAMES_DIR / session_id / "model_raw" / "index.json"
        if not model_meta_path.exists():
            self._send_text(409, "model_raw cache missing — run cache_model_raw.py\n"); return
        try:
            model_meta = json.loads(model_meta_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            self._send_text(500, f"model_raw/index.json read failed: {e}\n"); return

        try:
            result = _autotune_voxel_overlap(
                session_id=session_id,
                pose_dir=pose_dir,
                frames=frames,
                fit_space=fit_space,
                voxel_sizes=voxel_sizes,
                init_map=init_map,
                stride=stride,
                model_meta=model_meta,
            )
        except FileNotFoundError as e:
            self._send_text(409, f"{e}\n"); return
        except Exception as e:  # noqa: BLE001
            self._send_text(500, f"autotune-voxel failed: {type(e).__name__}: {e}\n"); return
        self._send_json(200, result)

    def _handle_common_features(self, session_id: str,
                                 pose_dir: str, frames_csv: str) -> None:
        if not SESSION_ID_RE.match(session_id) or not FRAME_VARIANT_RE.fullmatch(pose_dir):
            self.send_response(404); self.end_headers(); return
        frame_set = _parse_int_csv(frames_csv)
        if not frame_set:
            self._send_text(400, "frames= required (comma-separated integers)\n"); return
        meta_name = _features_meta_for_pose_dir(pose_dir)
        meta_path = FRAMES_DIR / session_id / meta_name
        if not meta_path.exists():
            self._send_text(
                404,
                f"{meta_name} not found — run "
                f"`tools/feature_ray_reconstruct.py --session {session_id} "
                f"--frames-variant {pose_dir}`\n",
            )
            return
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            self._send_text(500, f"meta read failed: {e}\n"); return
        common = _find_common_features(meta, frame_set)
        # Trim per-feature obs to only the requested frames so the payload
        # is small even on long-tracked features.
        out = []
        for f in common:
            obs_filtered = {k: f["obs"][k] for k in f["obs"] if k in frame_set}
            out.append({"world": f["world"], "obs": obs_filtered})
        self._send_json(200, {
            "session": session_id,
            "pose_dir": pose_dir,
            "meta_file": meta_name,
            "frames": sorted(frame_set),
            "count": len(out),
            "features": out,
        })

    def _handle_pixel_cloud_autotune(self, session_id: str,
                                     pose_dir: str, fit_space: str,
                                     frames_csv: str) -> None:
        if not SESSION_ID_RE.match(session_id) or not FRAME_VARIANT_RE.fullmatch(pose_dir):
            self.send_response(404); self.end_headers(); return
        if fit_space not in ("depth", "disparity"):
            self._send_text(400, f"bad fit_space {fit_space!r}\n"); return
        frame_set = _parse_int_csv(frames_csv)
        if not frame_set:
            self._send_text(400, "frames= required (comma-separated integers)\n"); return

        meta_name = _features_meta_for_pose_dir(pose_dir)
        meta_path = FRAMES_DIR / session_id / meta_name
        if not meta_path.exists():
            self._send_text(
                404,
                f"{meta_name} not found — run "
                f"feature_ray_reconstruct.py --frames-variant {pose_dir}\n",
            )
            return
        model_meta_path = FRAMES_DIR / session_id / "model_raw" / "index.json"
        if not model_meta_path.exists():
            self._send_text(409, "model_raw cache missing — run cache_model_raw.py\n"); return

        try:
            meta = json.loads(meta_path.read_text())
            model_meta = json.loads(model_meta_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            self._send_text(500, f"meta read failed: {e}\n"); return

        common = _find_common_features(meta, frame_set)
        if not common:
            self._send_json(200, {
                "frames": sorted(frame_set),
                "fit_space": fit_space,
                "n_common": 0,
                "per_frame": {},
                "warning": "no features observed in every selected frame",
            })
            return

        per_frame = {}
        for idx in sorted(frame_set):
            try:
                fit = _autotune_one_frame(
                    session_id, pose_dir, idx,
                    common_features=common,
                    model_meta=model_meta,
                    fit_space=fit_space,
                )
            except FileNotFoundError as e:
                per_frame[str(idx)] = {"error": str(e)}
                continue
            except Exception as e:  # noqa: BLE001
                per_frame[str(idx)] = {"error": f"{type(e).__name__}: {e}"}
                continue
            per_frame[str(idx)] = fit
        self._send_json(200, {
            "frames": sorted(frame_set),
            "fit_space": fit_space,
            "n_common": len(common),
            "per_frame": per_frame,
        })


# --------------------------------------------------------------------------
# Frame-debug helpers (called from the per-frame endpoints above).
# --------------------------------------------------------------------------

def _thumb_target_size(cw: int, ch: int, long_edge: int) -> tuple[int, int]:
    """Return (out_w, out_h) so the long edge equals `long_edge` and the
    aspect matches (cw, ch). Both thumbnails (colour + depth) use this so
    they end up the same shape in the panel."""
    if cw >= ch:
        return long_edge, max(1, int(round(long_edge * ch / cw)))
    return max(1, int(round(long_edge * cw / ch))), long_edge


def _render_color_thumb(body: bytes, size: int = 600) -> bytes | None:
    """Decode the captured RGBA payload, vertical-flip from GL convention to
    natural orientation, downscale to a thumbnail with long edge ≤ `size`,
    return PNG bytes."""
    import io
    import numpy as np
    from PIL import Image
    frame = parse_frame(body)
    if frame["color"] is None:
        return None
    cw = int(frame["color_width"]); ch = int(frame["color_height"])
    rgba = np.frombuffer(frame["color"], dtype=np.uint8).reshape(ch, cw, 4)
    rgb = np.ascontiguousarray(rgba[::-1, :, :3])  # GL → CV
    out_w, out_h = _thumb_target_size(cw, ch, size)
    img = Image.fromarray(rgb).resize((out_w, out_h), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _render_depth_thumb(body: bytes, size: int = 600) -> bytes | None:
    """Render the depth map as a grayscale PNG aligned with the colour image:
    we resample the (possibly landscape) depth buffer into the colour image's
    pixel grid via Bd⁻¹ at the target thumbnail resolution. That way colour
    and depth thumbs share orientation and aspect, and pixel-for-pixel they
    correspond to the same view ray.

    Pixels for which Bd⁻¹ lands outside the depth buffer (e.g. the corners
    of a wider-FOV colour image where the depth sensor doesn't cover) get
    rendered black."""
    import io
    import numpy as np
    from PIL import Image
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import fusion  # noqa: E402
    frame = parse_frame(body)
    dw = int(frame["width"]); dh = int(frame["height"])
    cw = int(frame["color_width"]); ch = int(frame["color_height"])
    if cw == 0 or ch == 0:
        # Fall back to the raw depth grid if there's no colour to align to.
        cw, ch = dw, dh
    depth = fusion.decode_depth(
        frame["depth"], dw, dh,
        int(frame["format"]), float(frame["rawValueToMeters"]),
    )
    Bd = fusion._mat4_from_column_major(frame["normDepthBufferFromNormView"])

    # Build a sampling grid at the *thumbnail's* output resolution in the
    # colour image's UV space, then map each cell through Bd to land in the
    # depth-buffer pixel grid.
    out_w, out_h = _thumb_target_size(cw, ch, size)
    yy, xx = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing="ij")
    u = (xx + 0.5) / out_w        # colour UV (0,0) bottom-left … but we
    v = (yy + 0.5) / out_h        # render top-down for natural display, see flip below
    nv = np.stack([u, v, np.zeros_like(u), np.ones_like(u)], axis=-1)
    nd = nv @ Bd.T
    safe = np.where(np.abs(nd[..., 3]) > 1e-12, nd[..., 3], 1.0)
    u_d = nd[..., 0] / safe
    v_d = nd[..., 1] / safe
    bx = (1.0 - u_d) * dw
    by = v_d * dh
    in_buf = (bx >= 0) & (bx <= dw - 1) & (by >= 0) & (by <= dh - 1)
    bxc = np.clip(bx, 0.0, dw - 1.0 - 1e-3)
    byc = np.clip(by, 0.0, dh - 1.0 - 1e-3)
    bx0 = np.floor(bxc).astype(np.int32); by0 = np.floor(byc).astype(np.int32)
    bx1 = np.minimum(bx0 + 1, dw - 1); by1 = np.minimum(by0 + 1, dh - 1)
    fx = (bxc - bx0).astype(np.float32); fy = (byc - by0).astype(np.float32)
    d00 = depth[by0, bx0]; d10 = depth[by0, bx1]
    d01 = depth[by1, bx0]; d11 = depth[by1, bx1]
    sampled = ((d00 * (1 - fx) + d10 * fx) * (1 - fy)
               + (d01 * (1 - fx) + d11 * fx) * fy)
    sampled = np.where(in_buf, sampled, 0.0)

    valid = (sampled > 0) & np.isfinite(sampled)
    if not valid.any():
        norm = np.zeros_like(sampled, dtype=np.uint8)
    else:
        d_min = float(np.percentile(sampled[valid], 2))
        d_max = float(np.percentile(sampled[valid], 98))
        d_max = max(d_max, d_min + 1e-3)
        norm = np.clip((sampled - d_min) / (d_max - d_min), 0.0, 1.0) * 255.0
        norm = np.where(valid, norm, 0).astype(np.uint8)
    # Sampling grid above had v=0 at the colour image's bottom (matching the
    # GL-convention buffer). Vertical-flip for natural top-down display.
    img = Image.fromarray(norm[::-1, :], mode="L")
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


# 11-point sample of matplotlib's 'turbo' colormap in 0–255 RGB, evenly
# spaced from t=0 (deep blue) to t=1 (deep red). Using a LUT + linear
# interpolation is plenty for thumbnail visualisation and lets us stay
# inside numpy without pulling in matplotlib.
_TURBO_LUT_RGB = [
    [ 48,  18,  59],
    [ 70,  74, 213],
    [ 50, 137, 252],
    [ 26, 195, 232],
    [ 47, 234, 162],
    [124, 250,  90],
    [201, 230,  43],
    [248, 178,  29],
    [243, 113,  19],
    [203,  41,  15],
    [122,   4,   3],
]


def _apply_turbo(x):
    """Map values in [0, 1] (any shape, ndarray) to uint8 RGB along a new
    trailing axis using a piecewise-linear approximation of matplotlib's
    'turbo' colormap."""
    import numpy as np
    lut = np.asarray(_TURBO_LUT_RGB, dtype=np.float32)
    n = lut.shape[0] - 1
    f = np.clip(x, 0.0, 1.0) * n
    i0 = np.floor(f).astype(np.int32)
    i1 = np.minimum(i0 + 1, n)
    t = (f - i0)[..., None]
    rgb = (1 - t) * lut[i0] + t * lut[i1]
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _apply_diverging(x):
    """Map values in [-1, 1] to uint8 RGB. Red for positive, blue for
    negative, near-white at zero. Saturates at the extremes."""
    import numpy as np
    t = np.clip(x, -1.0, 1.0)
    s = np.abs(t)
    fade = 1.0 - 0.85 * s   # white → coloured as |t| grows
    r = np.where(t >= 0, 1.0, fade)
    b = np.where(t <  0, 1.0, fade)
    g = fade
    rgb = np.stack([r, g, b], axis=-1) * 255.0
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _sample_phone_model_on_color_grid(body: bytes, *, out_w: int, out_h: int,
                                       model_raw_arr=None):
    """Sample WebXR phone depth and (optionally) cached model_raw at every
    pixel of a (out_h, out_w) thumbnail grid in NATURAL display orientation
    (yo=0 → top of displayed image). Returns (phone, model) arrays of shape
    (out_h, out_w) — float32, NaN for invalid samples. Pixels are paired:
    phone[i,j] and model[i,j] correspond to the same view ray.

    Phone depth is in metres; model_raw is the Depth-Anything-V2 prediction
    in its native units (≈ metres, but unrescaled). Pass model_raw_arr=None
    to skip the model branch entirely (model returned all-NaN)."""
    import numpy as np
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import fusion  # noqa: E402
    frame = parse_frame(body)
    cw = int(frame["color_width"]); ch = int(frame["color_height"])
    dw = int(frame["width"]);       dh = int(frame["height"])
    if cw == 0 or ch == 0:
        cw, ch = dw, dh
    depth = fusion.decode_depth(
        frame["depth"], dw, dh,
        int(frame["format"]), float(frame["rawValueToMeters"]),
    )
    Bd = fusion._mat4_from_column_major(frame["normDepthBufferFromNormView"])

    yy, xx = np.meshgrid(np.arange(out_h), np.arange(out_w), indexing="ij")
    # Natural display: yo=0 is the top row, but norm-view v=0 is the
    # bottom of the view. Invert v so the sampled pixel at the visual
    # top of the thumb maps to the top of the scene.
    u_v = (xx + 0.5) / out_w
    v_v = 1.0 - (yy + 0.5) / out_h

    # Phone: norm-view (u_v, v_v) → norm-depth-buffer via Bd → pixel.
    nv = np.stack([u_v, v_v, np.zeros_like(u_v), np.ones_like(u_v)], axis=-1)
    nd = nv @ Bd.T
    safe = np.where(np.abs(nd[..., 3]) > 1e-12, nd[..., 3], 1.0)
    u_d = nd[..., 0] / safe
    v_d = nd[..., 1] / safe
    bx = (1.0 - u_d) * dw - 0.5
    by = v_d * dh - 0.5
    phone = _bilinear_sample_2d(depth, bx, by).astype(np.float32)
    phone = np.where((phone > 0) & np.isfinite(phone), phone, np.float32("nan"))

    if model_raw_arr is not None:
        ch_c, cw_c = model_raw_arr.shape
        # model_raw was written by cache_model_raw.py from rgba *without*
        # a vertical flip, so model_raw row 0 = scene-bottom = norm-view
        # v=0. Sample at (u_v · cw_c, v_v · ch_c).
        sx = u_v * cw_c - 0.5
        sy = v_v * ch_c - 0.5
        model = _bilinear_sample_2d(model_raw_arr, sx, sy).astype(np.float32)
        model = np.where(np.isfinite(model) & (model > 1e-3),
                         model, np.float32("nan"))
    else:
        model = np.full_like(phone, np.float32("nan"))
    return phone, model


def _depth_to_turbo_rgb(depth):
    """uint8 RGB the same height/width as `depth`. NaN/zero → black.
    Range comes from the 2nd–98th percentile of the valid pixels so the
    colormap isn't squashed by a handful of outliers."""
    import numpy as np
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.zeros((*depth.shape, 3), dtype=np.uint8)
    d_min = float(np.percentile(depth[valid], 2))
    d_max = float(np.percentile(depth[valid], 98))
    d_max = max(d_max, d_min + 1e-3)
    norm = np.clip((depth - d_min) / (d_max - d_min), 0.0, 1.0)
    rgb = _apply_turbo(norm)
    rgb[~valid] = 0
    return rgb


def _render_phone_color_thumb(body: bytes, *, size: int = 600):
    """Phone depth as turbo-colormapped PNG, in natural display orientation.
    The sampling grid matches `_render_color_thumb` so depth and colour
    line up pixel-for-pixel in the same thumbnail."""
    import io
    from PIL import Image
    frame = parse_frame(body)
    cw = int(frame["color_width"]); ch = int(frame["color_height"])
    dw = int(frame["width"]);       dh = int(frame["height"])
    if cw == 0 or ch == 0:
        cw, ch = dw, dh
    out_w, out_h = _thumb_target_size(cw, ch, size)
    phone, _ = _sample_phone_model_on_color_grid(
        body, out_w=out_w, out_h=out_h, model_raw_arr=None,
    )
    rgb = _depth_to_turbo_rgb(phone)
    img = Image.fromarray(rgb)
    buf = io.BytesIO(); img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _render_model_color_thumb(body: bytes, model_raw_arr, *, size: int = 600):
    """Model_raw rendered with the same orientation + colormap as the
    phone thumb so the user can flick between them and visually compare."""
    import io
    from PIL import Image
    frame = parse_frame(body)
    cw = int(frame["color_width"]); ch = int(frame["color_height"])
    dw = int(frame["width"]);       dh = int(frame["height"])
    if cw == 0 or ch == 0:
        cw, ch = dw, dh
    out_w, out_h = _thumb_target_size(cw, ch, size)
    _, model = _sample_phone_model_on_color_grid(
        body, out_w=out_w, out_h=out_h, model_raw_arr=model_raw_arr,
    )
    rgb = _depth_to_turbo_rgb(model)
    img = Image.fromarray(rgb)
    buf = io.BytesIO(); img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _render_diff_color_thumb(body: bytes, model_raw_arr, *, size: int = 600):
    """Per-pixel signed (model − phone) on the colour grid, rendered with a
    diverging Red-White-Blue colormap. Range is symmetrical around 0 with
    extents at the 95th percentile of |diff| so a few crazy values don't
    flatten everything else. Pixels missing either source render black —
    that gap pattern itself is informative (e.g., model has corners that
    phone-depth doesn't cover)."""
    import io
    import numpy as np
    from PIL import Image
    frame = parse_frame(body)
    cw = int(frame["color_width"]); ch = int(frame["color_height"])
    dw = int(frame["width"]);       dh = int(frame["height"])
    if cw == 0 or ch == 0:
        cw, ch = dw, dh
    out_w, out_h = _thumb_target_size(cw, ch, size)
    phone, model = _sample_phone_model_on_color_grid(
        body, out_w=out_w, out_h=out_h, model_raw_arr=model_raw_arr,
    )
    diff = model - phone
    valid = np.isfinite(diff)
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    if valid.any():
        scale = float(np.percentile(np.abs(diff[valid]), 95))
        scale = max(scale, 1e-3)
        norm = np.clip(diff / scale, -1.0, 1.0)
        # _apply_diverging vectorises over the full grid; we mask to keep
        # the invalid pixels black.
        rgb_full = _apply_diverging(np.where(valid, norm, 0.0))
        rgb[valid] = rgb_full[valid]
    img = Image.fromarray(rgb)
    buf = io.BytesIO(); img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _compute_blend_metres(phone, model, sigma_px: float):
    """Fuse phone-depth (metres, sparse) and model_raw (DA-V2 native units,
    dense) into a single depth map in metres, taking low-frequency content
    from phone and high-frequency detail from model. Approach:

      1. OLS-fit `model_metres = a · model_raw + b` on the overlap of valid
         pixels — same affine the autotune endpoints use.
      2. Hole-aware Gaussian low-pass:  blur(x_with_NaN_zeroed) / blur(mask)
         is a standard trick that gives a sane local mean even when phone
         has missing pixels. Sigma is in pixels of the supplied grid.
      3. blend = low_phone + (model_metres − low_model_metres)

    Pixels where phone has no valid neighbours within the blur kernel
    fall back to plain `model_metres` so the result has no NaN holes
    introduced beyond what model_raw itself can't cover. Returns
    (blend_metres, model_metres, fit_a, fit_b, n_overlap)."""
    import numpy as np
    from scipy.ndimage import gaussian_filter

    overlap = np.isfinite(phone) & np.isfinite(model)
    n_overlap = int(overlap.sum())
    if n_overlap < 100:
        # Without a usable overlap we can't fit the affine; pass through
        # phone as the blend so the page still has something to render.
        return phone.astype(np.float32), phone.astype(np.float32), 1.0, 0.0, n_overlap

    M = model[overlap].astype(np.float64)
    P = phone[overlap].astype(np.float64)
    A = np.stack([M, np.ones_like(M)], axis=-1)
    coeffs, *_ = np.linalg.lstsq(A, P, rcond=None)
    a = float(coeffs[0]); b = float(coeffs[1])
    model_metres = (a * model + b).astype(np.float32)

    def blur_holes(x):
        valid = np.isfinite(x).astype(np.float32)
        x0 = np.where(np.isfinite(x), x, 0.0).astype(np.float32)
        num = gaussian_filter(x0, sigma=sigma_px, mode="reflect")
        den = gaussian_filter(valid, sigma=sigma_px, mode="reflect")
        # Threshold the mask blur so we don't divide by a sliver of valid
        # pixels at a far edge — that produces wild extrapolations.
        return np.where(den > 0.05, num / np.maximum(den, 1e-6), np.nan)

    low_phone = blur_holes(phone)
    low_model = blur_holes(model_metres)
    detail = model_metres - low_model
    blend = low_phone + detail
    blend = np.where(np.isfinite(blend), blend, model_metres)
    return blend.astype(np.float32), model_metres, a, b, n_overlap


def _render_blend_color_thumb(body: bytes, model_raw_arr, *, size: int = 600,
                               sigma_frac: float = 0.03):
    """Detail-injected blend rendered with the same turbo colormap as the
    other depth thumbs. `sigma_frac` is the Gaussian sigma expressed as a
    fraction of the thumbnail's diagonal (so the blur scales sensibly
    across thumb resolutions instead of being baked-in pixels)."""
    import io
    import math
    from PIL import Image
    frame = parse_frame(body)
    cw = int(frame["color_width"]); ch = int(frame["color_height"])
    dw = int(frame["width"]);       dh = int(frame["height"])
    if cw == 0 or ch == 0:
        cw, ch = dw, dh
    out_w, out_h = _thumb_target_size(cw, ch, size)
    phone, model = _sample_phone_model_on_color_grid(
        body, out_w=out_w, out_h=out_h, model_raw_arr=model_raw_arr,
    )
    diag = math.hypot(out_w, out_h)
    sigma_px = max(1.0, float(sigma_frac) * diag)
    blend, *_ = _compute_blend_metres(phone, model, sigma_px)
    rgb = _depth_to_turbo_rgb(blend)
    img = Image.fromarray(rgb)
    buf = io.BytesIO(); img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _load_model_raw_array(session_id: str, idx: int):
    """Read the cached float16 model_raw prediction for one frame, plus
    its (cw, ch) from index.json. Returns (arr, cw, ch) or None if the
    cache is missing for this frame."""
    import numpy as np
    meta_path = FRAMES_DIR / session_id / "model_raw" / "index.json"
    raw_path = FRAMES_DIR / session_id / "model_raw" / f"frame_{idx:06d}.f16"
    if not meta_path.exists() or not raw_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    entry = meta.get(str(idx))
    if entry is None:
        return None
    cw_c = int(entry["w"]); ch_c = int(entry["h"])
    arr = (np.frombuffer(raw_path.read_bytes(), dtype=np.float16)
              .astype(np.float32, copy=False).reshape(ch_c, cw_c))
    return arr


def _ranks_with_ties(x):
    """1-based fractional ranks (ties get the average rank of the run)."""
    import numpy as np
    n = x.size
    order = np.argsort(x, kind="quicksort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1)
    sx = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and sx[j + 1] == sx[i]:
            j += 1
        if j > i:
            avg = (i + j + 2) / 2.0
            ranks[order[i:j+1]] = avg
        i = j + 1
    return ranks


def _frame_to_voxel_indices(
    body: bytes,
    voxel_size: float,
    world_min: list,
    shape: list,
    *,
    near: float = 0.05,
    far: float = 8.0,
) -> tuple[list, list, list]:
    """Decode a frame, backproject every valid depth pixel to world coords,
    bin to the supplied voxel grid, return the list of unique flat voxel
    indices the frame "claims" — plus the camera pose and four frustum
    corners (at far_thumb m) so the caller can draw the camera viewport.

    Subsamples the depth pixels to FRAME_VOXELS_PIXEL_CAP at most so the
    response stays small even for high-res refined depth maps."""
    import numpy as np
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import fusion  # noqa: E402

    frame = parse_frame(body)
    dw = int(frame["width"]); dh = int(frame["height"])
    depth = fusion.decode_depth(
        frame["depth"], dw, dh,
        int(frame["format"]), float(frame["rawValueToMeters"]),
    )
    V  = fusion._mat4_from_column_major(frame["viewMatrix"])
    P  = fusion._mat4_from_column_major(frame["projectionMatrix"])
    Bd = fusion._mat4_from_column_major(frame["normDepthBufferFromNormView"])
    Bv = np.linalg.inv(Bd)
    P_inv = np.linalg.inv(P)
    cam = V[:3, 3].astype(np.float64)

    # Subsample stride so npix/stride² ≤ cap.
    stride = max(1, int(np.ceil(np.sqrt(dw * dh / FRAME_VOXELS_PIXEL_CAP))))
    bx_g, by_g = np.meshgrid(
        np.arange(0, dw, stride, dtype=np.float64),
        np.arange(0, dh, stride, dtype=np.float64),
        indexing="xy",
    )
    by_int = by_g.astype(np.int32); bx_int = bx_g.astype(np.int32)
    d = depth[by_int, bx_int]
    valid = (d > near) & (d < far)
    if not valid.any():
        return [], cam.tolist(), _frustum_corners(P_inv, V, near=0.1, far=2.0)
    bx_v = bx_g[valid]; by_v = by_g[valid]; d_v = d[valid]

    # depth-buffer pixel → norm-view UV via Bd⁻¹.
    u_d = 1.0 - (bx_v + 0.5) / dw
    v_d = (by_v + 0.5) / dh
    nd_h = np.stack([u_d, v_d, np.zeros_like(u_d), np.ones_like(u_d)], axis=-1)
    nv_h = nd_h @ Bv.T
    safe = np.where(np.abs(nv_h[..., 3]) > 1e-12, nv_h[..., 3], 1.0)
    u = nv_h[..., 0] / safe
    v = nv_h[..., 1] / safe

    # norm-view UV → view-space ray direction via P⁻¹.
    x_ndc = 2.0 * u - 1.0
    y_ndc = 2.0 * v - 1.0
    clip = np.stack([x_ndc, y_ndc, -np.ones_like(x_ndc), np.ones_like(x_ndc)], axis=-1)
    view4 = clip @ P_inv.T
    view3 = view4[:, :3] / np.where(np.abs(view4[:, 3:4]) < 1e-12, 1.0, view4[:, 3:4])
    rays = view3 / np.linalg.norm(view3, axis=-1, keepdims=True)
    rays_world = rays @ V[:3, :3].T

    pts = cam[None, :] + d_v[:, None] * rays_world
    wmin = np.asarray(world_min, dtype=np.float64)
    Nx, Ny, Nz = int(shape[0]), int(shape[1]), int(shape[2])

    ix = np.floor((pts[:, 0] - wmin[0]) / voxel_size).astype(np.int64)
    iy = np.floor((pts[:, 1] - wmin[1]) / voxel_size).astype(np.int64)
    iz = np.floor((pts[:, 2] - wmin[2]) / voxel_size).astype(np.int64)
    in_grid = (
        (ix >= 0) & (ix < Nx)
        & (iy >= 0) & (iy < Ny)
        & (iz >= 0) & (iz < Nz)
    )
    ix = ix[in_grid]; iy = iy[in_grid]; iz = iz[in_grid]
    flat = (ix * Ny * Nz + iy * Nz + iz).astype(np.int64)
    uniq = np.unique(flat)
    # Pack as [ix, iy, iz] triples for the JS to render.
    iz_o = uniq % Nz
    iy_o = (uniq // Nz) % Ny
    ix_o = uniq // (Ny * Nz)
    triples = np.stack([ix_o, iy_o, iz_o], axis=-1).astype(int).tolist()

    pose = [float(cam[0]), float(cam[1]), float(cam[2])]
    return triples, pose, _frustum_corners(P_inv, V, near=0.1, far=2.0)


def _sample_depth_at(*, session_id: str, pose_dir: str, kind: str,
                     frames: list[int], us: list[float], vs: list[float]) -> list:
    """For each (frame_i, u_i, v_i) triple, return the depth metres at
    that norm-view UV from the chosen depth source. Frames are loaded
    on demand and cached for the duration of this call so a batch of
    points on the same frame doesn't re-decode the bin repeatedly.
    Out-of-bounds samples come back as JSON null."""
    import numpy as np
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import fusion  # noqa: E402

    # Per-frame cache: idx → either decoded phone depth (with Bd) OR
    # the cached model_raw (with cw, ch).
    phone_cache: dict[int, dict] = {}
    model_cache: dict[int, np.ndarray] = {}
    model_meta: dict | None = None
    if kind == "model":
        meta_path = FRAMES_DIR / session_id / "model_raw" / "index.json"
        if not meta_path.exists():
            raise FileNotFoundError("model_raw cache missing — run cache_model_raw.py")
        try:
            model_meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise RuntimeError(f"model_raw/index.json read failed: {e}") from e

    def get_phone(idx: int) -> dict:
        if idx in phone_cache:
            return phone_cache[idx]
        f = FRAMES_DIR / session_id / pose_dir / f"frame_{idx:06d}.bin"
        if not f.exists():
            raise FileNotFoundError(f"missing {pose_dir}/frame_{idx:06d}.bin")
        frame = parse_frame(f.read_bytes())
        dw = int(frame["width"]); dh = int(frame["height"])
        depth = fusion.decode_depth(
            frame["depth"], dw, dh,
            int(frame["format"]), float(frame["rawValueToMeters"]),
        )
        Bd = fusion._mat4_from_column_major(frame["normDepthBufferFromNormView"])
        phone_cache[idx] = {"depth": depth, "Bd": Bd, "dw": dw, "dh": dh}
        return phone_cache[idx]

    def get_model(idx: int) -> tuple[np.ndarray, int, int]:
        if idx in model_cache:
            arr = model_cache[idx]
            return arr, arr.shape[1], arr.shape[0]
        if model_meta is None:
            raise RuntimeError("model_meta missing")
        entry = model_meta.get(str(idx))
        if entry is None:
            raise FileNotFoundError(f"model_raw/index.json has no entry for frame {idx}")
        cw_c = int(entry["w"]); ch_c = int(entry["h"])
        raw_path = FRAMES_DIR / session_id / "model_raw" / f"frame_{idx:06d}.f16"
        if not raw_path.exists():
            raise FileNotFoundError(f"model_raw cache missing for frame {idx}")
        arr = (np.frombuffer(raw_path.read_bytes(), dtype=np.float16)
                 .astype(np.float32, copy=False).reshape(ch_c, cw_c))
        model_cache[idx] = arr
        return arr, cw_c, ch_c

    out: list = []
    for idx, u, v in zip(frames, us, vs):
        if not (0 <= u <= 1 and 0 <= v <= 1):
            out.append(None); continue
        if kind == "model":
            arr, cw_c, ch_c = get_model(idx)
            sx = u * cw_c - 0.5
            sy = v * ch_c - 0.5
            sample = _bilinear_sample_2d(arr, np.asarray([sx]), np.asarray([sy]))
            val = float(sample[0])
            out.append(val if np.isfinite(val) and val > 1e-3 else None)
        else:  # phone
            ph = get_phone(idx)
            depth = ph["depth"]; Bd = ph["Bd"]; dw = ph["dw"]; dh = ph["dh"]
            # norm-view (u, v) → norm-depth-buffer via Bd, then to pixel
            # with the same horizontal flip used by every other phone-
            # depth sampler in this codebase.
            nv = np.array([u, v, 0.0, 1.0], dtype=np.float64)
            nd = Bd @ nv
            w = nd[3] if abs(nd[3]) > 1e-12 else 1.0
            u_d = nd[0] / w
            v_d = nd[1] / w
            bx = (1.0 - u_d) * dw - 0.5
            by = v_d * dh - 0.5
            sample = _bilinear_sample_2d(depth, np.asarray([bx]), np.asarray([by]))
            val = float(sample[0])
            out.append(val if np.isfinite(val) and val > 1e-3 else None)
    return out


def _snap_feature_match(*, ref_body: bytes, tgt_body: bytes,
                         ref_uv: tuple[float, float],
                         init_uv: tuple[float, float],
                         patch_norm: float, radius_norm: float) -> dict:
    """Vectorised normalised-cross-correlation patch match. Returns
    refined (u, v) on the target plus the NCC score and the search
    bounds actually used (in case clamping kicked in)."""
    import numpy as np

    ref_frame = parse_frame(ref_body)
    tgt_frame = parse_frame(tgt_body)
    if ref_frame["color"] is None or tgt_frame["color"] is None:
        raise ValueError("ref or target frame has no colour buffer")
    cw_r = int(ref_frame["color_width"]); ch_r = int(ref_frame["color_height"])
    cw_t = int(tgt_frame["color_width"]); ch_t = int(tgt_frame["color_height"])
    rgba_r = np.frombuffer(ref_frame["color"], dtype=np.uint8).reshape(ch_r, cw_r, 4)
    rgba_t = np.frombuffer(tgt_frame["color"], dtype=np.uint8).reshape(ch_t, cw_t, 4)
    # rgba is GL-bottom-up (row 0 = scene-bottom). For convenience we
    # work in CV-style (row 0 = top); just flip vertically once.
    img_r = np.ascontiguousarray(rgba_r[::-1, :, :3])
    img_t = np.ascontiguousarray(rgba_t[::-1, :, :3])
    gray_r = (0.299*img_r[..., 0] + 0.587*img_r[..., 1] + 0.114*img_r[..., 2]).astype(np.float32)
    gray_t = (0.299*img_t[..., 0] + 0.587*img_t[..., 1] + 0.114*img_t[..., 2]).astype(np.float32)

    # Convert ref UV (norm-view, v=0 bottom) to top-down pixel.
    cx_r = float(ref_uv[0]) * cw_r
    cy_r = (1.0 - float(ref_uv[1])) * ch_r
    cx_t0 = float(init_uv[0]) * cw_t
    cy_t0 = (1.0 - float(init_uv[1])) * ch_t

    hp_r = max(2, int(round(patch_norm * min(cw_r, ch_r) / 2.0)))
    hp_t = hp_r   # same square patch on the target
    sr   = max(2, int(round(radius_norm * min(cw_t, ch_t))))

    cx_r_i = int(round(cx_r)); cy_r_i = int(round(cy_r))
    if (cx_r_i - hp_r < 0 or cx_r_i + hp_r + 1 > cw_r
        or cy_r_i - hp_r < 0 or cy_r_i + hp_r + 1 > ch_r):
        # Reference patch is at the edge — return init, no refinement.
        return {"u": float(init_uv[0]), "v": float(init_uv[1]),
                "score": 0.0, "snapped": False,
                "reason": "ref patch out of bounds"}
    ref_patch = gray_r[cy_r_i - hp_r:cy_r_i + hp_r + 1,
                       cx_r_i - hp_r:cx_r_i + hp_r + 1]
    ref_p = ref_patch - ref_patch.mean()
    ref_p_norm = float(np.sqrt((ref_p * ref_p).sum()))
    if ref_p_norm < 1e-6:
        return {"u": float(init_uv[0]), "v": float(init_uv[1]),
                "score": 0.0, "snapped": False,
                "reason": "ref patch is flat"}

    cx_t_i = int(round(cx_t0)); cy_t_i = int(round(cy_t0))
    x_lo = max(hp_t,             cx_t_i - sr)
    x_hi = min(cw_t - hp_t - 1,  cx_t_i + sr)
    y_lo = max(hp_t,             cy_t_i - sr)
    y_hi = min(ch_t - hp_t - 1,  cy_t_i + sr)
    if x_lo > x_hi or y_lo > y_hi:
        return {"u": float(init_uv[0]), "v": float(init_uv[1]),
                "score": 0.0, "snapped": False,
                "reason": "search window empty (init too close to edge)"}

    # Vectorise: build the (n_y, n_x, 2hp+1, 2hp+1) tensor of candidate
    # patches via stride tricks. With typical sizes (~60×60×24×24 ≈ 80k
    # cells) this is well under 10 ms.
    from numpy.lib.stride_tricks import sliding_window_view
    windows = sliding_window_view(gray_t, (2*hp_t + 1, 2*hp_t + 1))
    # windows shape: (ch_t - 2hp, cw_t - 2hp, 2hp+1, 2hp+1)
    sub = windows[y_lo - hp_t:y_hi - hp_t + 1,
                  x_lo - hp_t:x_hi - hp_t + 1]
    sub = sub.astype(np.float32, copy=False)
    means = sub.mean(axis=(-1, -2), keepdims=True)
    sub_c = sub - means
    num = (sub_c * ref_p[None, None, :, :]).sum(axis=(-1, -2))
    sub_norm = np.sqrt((sub_c * sub_c).sum(axis=(-1, -2)))
    ncc = num / (ref_p_norm * sub_norm + 1e-9)
    best_idx = int(np.nanargmax(ncc))
    n_y, n_x = ncc.shape
    by = best_idx // n_x
    bx = best_idx % n_x
    best_score = float(ncc[by, bx])
    best_cx = x_lo + bx
    best_cy = y_lo + by
    u_out = best_cx / cw_t
    v_out = 1.0 - best_cy / ch_t
    return {"u": float(u_out), "v": float(v_out),
            "score": best_score, "snapped": True,
            "patch_px": int(2 * hp_r + 1), "radius_px": int(sr),
            "ref_size": [cw_r, ch_r], "target_size": [cw_t, ch_t]}


def _ray_for_uv(frame: dict, u: float, v: float):
    """Return (origin_world (3,), dir_world (3,) unit) for a ray from
    the frame's camera through norm-view UV (u, v). Mirrors the camera-Z
    convention used everywhere else in the project."""
    import numpy as np
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import fusion  # noqa: E402
    V = fusion._mat4_from_column_major(frame["viewMatrix"])
    P = fusion._mat4_from_column_major(frame["projectionMatrix"])
    P_inv = np.linalg.inv(P)
    cam = V[:3, 3].astype(np.float64)
    x_ndc = 2.0 * float(u) - 1.0
    y_ndc = 2.0 * float(v) - 1.0
    clip = np.array([x_ndc, y_ndc, -1.0, 1.0], dtype=np.float64)
    view4 = P_inv @ clip
    w = view4[3] if abs(view4[3]) > 1e-12 else 1.0
    view3 = view4[:3] / w
    n = np.linalg.norm(view3)
    if n < 1e-12:
        return cam, np.array([0.0, 0.0, -1.0], dtype=np.float64)
    unit_view = view3 / n
    dir_world = V[:3, :3] @ unit_view
    # Re-normalise (V[:3,:3] should be a rotation, so it preserves
    # length, but we hedge against accumulated float drift).
    nd = np.linalg.norm(dir_world)
    if nd > 1e-12:
        dir_world = dir_world / nd
    return cam, dir_world


def _triangulate_rays(rays):
    """LS triangulation: minimise sum_i |proj_perp_to_ray_i (P - O_i)|².
    Solves (sum_i M_i) P = sum_i M_i O_i, where M_i = I - D_i D_iᵀ.
    `rays` is a list of (origin, unit_dir). Returns world point or None
    if the system is rank-deficient (e.g. <2 rays, or all parallel)."""
    import numpy as np
    if len(rays) < 2:
        return None
    A = np.zeros((3, 3), dtype=np.float64)
    b = np.zeros(3, dtype=np.float64)
    for o, d in rays:
        M = np.eye(3) - np.outer(d, d)
        A += M
        b += M @ o
    try:
        # Use solve, which raises on singular; fall back to lstsq.
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        return sol


def _triplet_distances_payload(*, session_id: str, pose_dir: str,
                                features_in: list,
                                sigma_frac: float = 0.03) -> dict:
    """Compute, per feature, an LS-triangulated world point from the
    user's clicked marks, plus per-mark (phone_depth, model_depth,
    blend_depth, cam_distance_to_world_pt). Frames are decoded once and
    cached for the duration of the call. The blend grid for each (frame,
    σ) pair lives in a module-level cache so successive marks on the
    same frame don't recompute it."""
    import math
    import numpy as np
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import fusion  # noqa: E402

    frame_cache: dict[int, dict] = {}
    model_meta_path = FRAMES_DIR / session_id / "model_raw" / "index.json"
    model_meta: dict | None = None
    if model_meta_path.exists():
        try:
            model_meta = json.loads(model_meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            model_meta = None
    model_arr_cache: dict[int, np.ndarray] = {}

    # Blend grid resolution: long-edge ~1000 px keeps Gaussian-filter
    # cost low enough to be unnoticeable per request even for the cold
    # path; the colour-image-aspect grid means UV→pixel sampling is
    # exact (a wider/shorter grid would fold in extra resampling
    # error at the bilinear sample step).
    BLEND_LONG_EDGE = 1000
    sigma_q = round(float(sigma_frac), 3)

    def get_frame(idx: int) -> dict:
        if idx in frame_cache:
            return frame_cache[idx]
        f = FRAMES_DIR / session_id / pose_dir / f"frame_{idx:06d}.bin"
        if not f.exists():
            raise FileNotFoundError(f"missing {pose_dir}/frame_{idx:06d}.bin")
        body = f.read_bytes()
        frame = parse_frame(body)
        dw = int(frame["width"]); dh = int(frame["height"])
        depth = fusion.decode_depth(
            frame["depth"], dw, dh,
            int(frame["format"]), float(frame["rawValueToMeters"]),
        )
        Bd = fusion._mat4_from_column_major(frame["normDepthBufferFromNormView"])
        V  = fusion._mat4_from_column_major(frame["viewMatrix"])
        V_w2c = np.linalg.inv(V)
        frame_cache[idx] = {
            "frame": frame, "body": body, "dw": dw, "dh": dh,
            "phone_depth": depth, "Bd": Bd, "V_w2c": V_w2c,
        }
        return frame_cache[idx]

    def get_model(idx: int):
        if idx in model_arr_cache:
            arr = model_arr_cache[idx]
            return arr, arr.shape[1], arr.shape[0]
        if model_meta is None:
            return None, 0, 0
        entry = model_meta.get(str(idx))
        if entry is None:
            return None, 0, 0
        cw_c = int(entry["w"]); ch_c = int(entry["h"])
        raw_path = FRAMES_DIR / session_id / "model_raw" / f"frame_{idx:06d}.f16"
        if not raw_path.exists():
            return None, 0, 0
        arr = (np.frombuffer(raw_path.read_bytes(), dtype=np.float16)
                 .astype(np.float32, copy=False).reshape(ch_c, cw_c))
        model_arr_cache[idx] = arr
        return arr, cw_c, ch_c

    def get_blend_grid(idx: int):
        """Return (blend_array, out_w, out_h) or (None, 0, 0) if the
        model_raw cache is missing for this frame."""
        arr, _, _ = get_model(idx)
        if arr is None:
            return None, 0, 0
        fr = get_frame(idx)
        cw = int(fr["frame"]["color_width"])
        ch_ = int(fr["frame"]["color_height"])
        if cw == 0 or ch_ == 0:
            cw = fr["dw"]; ch_ = fr["dh"]
        out_w, out_h = _thumb_target_size(cw, ch_, BLEND_LONG_EDGE)
        key = (session_id, idx, sigma_q, out_w, out_h)
        cached = _BLEND_GRID_CACHE.get(key)
        if cached is not None:
            return cached, out_w, out_h
        phone, model = _sample_phone_model_on_color_grid(
            fr["body"], out_w=out_w, out_h=out_h, model_raw_arr=arr,
        )
        sigma_px = max(1.0, float(sigma_frac) * math.hypot(out_w, out_h))
        blend, *_ = _compute_blend_metres(phone, model, sigma_px)
        if len(_BLEND_GRID_CACHE) >= _BLEND_GRID_CACHE_MAX:
            _BLEND_GRID_CACHE.pop(next(iter(_BLEND_GRID_CACHE)))
        _BLEND_GRID_CACHE[key] = blend
        return blend, out_w, out_h

    out_features = []
    for f in features_in:
        marks = f.get("marks", []) if isinstance(f, dict) else []
        rays = []
        ray_marks = []   # parallel — same indices to map back
        for m in marks:
            try:
                idx = int(m["frame"])
                u   = float(m["u"]); v = float(m["v"])
            except (KeyError, TypeError, ValueError):
                continue
            try:
                fr = get_frame(idx)
            except FileNotFoundError:
                continue
            o, d = _ray_for_uv(fr["frame"], u, v)
            rays.append((o, d))
            ray_marks.append((idx, u, v))
        world = _triangulate_rays(rays)
        per_mark_out = []
        for (idx, u, v) in ray_marks:
            fr = get_frame(idx)
            # phone depth at (u, v)
            Bd = fr["Bd"]; dw = fr["dw"]; dh = fr["dh"]
            nv = np.array([u, v, 0.0, 1.0], dtype=np.float64)
            nd = Bd @ nv
            wd = nd[3] if abs(nd[3]) > 1e-12 else 1.0
            u_d = nd[0] / wd; v_d = nd[1] / wd
            bx = (1.0 - u_d) * dw - 0.5
            by = v_d * dh - 0.5
            ph = _bilinear_sample_2d(fr["phone_depth"],
                                       np.asarray([bx]), np.asarray([by]))
            phone_d = float(ph[0])
            phone_d = phone_d if (np.isfinite(phone_d) and phone_d > 1e-3) else None
            # model depth at (u, v)
            arr, cw_c, ch_c = get_model(idx)
            if arr is None:
                model_d = None
            else:
                sx = u * cw_c - 0.5; sy = v * ch_c - 0.5
                ms = _bilinear_sample_2d(arr, np.asarray([sx]), np.asarray([sy]))
                m_val = float(ms[0])
                model_d = m_val if (np.isfinite(m_val) and m_val > 1e-3) else None
            # blend depth at (u, v). The blend grid was built in NATURAL
            # display orientation (yo=0 is the top row, norm-view v=1),
            # so to sample at norm-view (u, v) we flip v back to a row
            # index: yo = (1 − v) · out_h.
            blend_grid, b_w, b_h = get_blend_grid(idx)
            if blend_grid is None:
                blend_d = None
            else:
                bx = u * b_w - 0.5
                by = (1.0 - v) * b_h - 0.5
                bs = _bilinear_sample_2d(blend_grid,
                                          np.asarray([bx]), np.asarray([by]))
                b_val = float(bs[0])
                blend_d = b_val if (np.isfinite(b_val) and b_val > 1e-3) else None
            # cam-Z distance from this frame's camera to triangulated P.
            cam_d = None
            if world is not None:
                Ph = np.array([world[0], world[1], world[2], 1.0])
                p_view = fr["V_w2c"] @ Ph
                cam_z = -float(p_view[2])
                if np.isfinite(cam_z) and cam_z > 0:
                    cam_d = cam_z
            per_mark_out.append({
                "frame": idx, "u": u, "v": v,
                "phone_depth": phone_d,
                "model_depth": model_d,
                "blend_depth": blend_d,
                "cam_distance": cam_d,
            })
        out_features.append({
            "world": ([float(world[0]), float(world[1]), float(world[2])]
                      if world is not None else None),
            "marks": per_mark_out,
            "n_rays": len(rays),
        })
    return {
        "session": session_id,
        "pose_dir": pose_dir,
        "features": out_features,
    }


def _build_frame_rays_and_raw(session_id: str, pose_dir: str, idx: int,
                               model_meta: dict, stride: int):
    """Decode one frame's geometry + the cached model_raw at a stride
    suitable for voxel-overlap optimisation. Returns
    (origin_xyz: (3,) float64, dirs: (N, 3) float64, raw: (N,) float64)
    where each ray is the camera-Z-perpendicular direction (so refined
    depth d satisfies world_pt = origin + d · ray, matching the spec
    in fusion.frame_to_world_points). NaN / non-positive raw entries
    are filtered out so the optimiser doesn't waste evals on them."""
    import numpy as np
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import fusion  # noqa: E402

    f_path = FRAMES_DIR / session_id / pose_dir / f"frame_{idx:06d}.bin"
    if not f_path.exists():
        raise FileNotFoundError(f"missing {pose_dir}/frame_{idx:06d}.bin")
    raw_path = FRAMES_DIR / session_id / "model_raw" / f"frame_{idx:06d}.f16"
    if not raw_path.exists():
        raise FileNotFoundError(f"model_raw cache missing for frame {idx}")
    entry = model_meta.get(str(idx))
    if entry is None:
        raise FileNotFoundError(f"model_raw/index.json has no entry for frame {idx}")
    cw_c = int(entry["w"]); ch_c = int(entry["h"])
    pred_full = (np.frombuffer(raw_path.read_bytes(), dtype=np.float16)
                   .astype(np.float32, copy=False).reshape(ch_c, cw_c))

    frame = parse_frame(f_path.read_bytes())
    V  = fusion._mat4_from_column_major(frame["viewMatrix"])
    P  = fusion._mat4_from_column_major(frame["projectionMatrix"])
    P_inv = np.linalg.inv(P)
    cam = V[:3, 3].astype(np.float64)

    # Sample the colour grid at the same stride convention as the live
    # pixel-cloud endpoint: (cx + 0.5)/cw, (cy + 0.5)/ch.
    gxs = np.arange(0, cw_c, stride, dtype=np.float64)
    gys = np.arange(0, ch_c, stride, dtype=np.float64)
    cx, cy = np.meshgrid(gxs, gys, indexing="xy")
    raw = pred_full[cy.astype(np.int32), cx.astype(np.int32)]
    u = (cx + 0.5) / cw_c
    v = (cy + 0.5) / ch_c
    x_ndc = 2.0 * u - 1.0
    y_ndc = 2.0 * v - 1.0
    clip = np.stack([x_ndc, y_ndc, -np.ones_like(x_ndc),
                     np.ones_like(x_ndc)], axis=-1)
    view4 = clip @ P_inv.T
    safe_w = np.where(np.abs(view4[..., 3:4]) > 1e-12, view4[..., 3:4], 1.0)
    view3 = view4[..., :3] / safe_w
    near_z = view3[..., 2:3]
    near_z_safe = np.where(np.abs(near_z) > 1e-6, near_z, -1e-6)
    view_dirs = view3 / -near_z_safe
    rays_world = view_dirs @ V[:3, :3].T   # (gh, gw, 3)

    raw_flat = raw.reshape(-1).astype(np.float64)
    rays_flat = rays_world.reshape(-1, 3).astype(np.float64)
    mask = np.isfinite(raw_flat) & (raw_flat > 1e-3)
    return cam, rays_flat[mask], raw_flat[mask]


def _depth_from_affine(raw, a, b, fit_space):
    """Apply (a, b) in the chosen space to a raw model-depth array.
    Returns refined depth (same shape) clipped at 0.0 for negatives /
    nonpositive denominators. Called once per optimiser eval per frame."""
    import numpy as np
    if fit_space == "disparity":
        denom = a + b * raw
        d = np.where(denom > 1e-3, raw / np.where(denom > 1e-3, denom, 1.0), 0.0)
    else:
        d = a * raw + b
    return np.where(d > 0, d, 0.0)


# Voxel-hash bit layout: each component shifted into a 21-bit slot of
# an int64. Indices are biased by VOXEL_BIAS so negative cells encode
# without sign-extension surprise. VOXEL_BIAS = 2^20 = 1048576 covers
# a world span of ±~52 km at 5 cm cells — generous.
_VOXEL_BIAS = 1 << 20
_VOXEL_MASK = (1 << 21) - 1


def _voxel_hashes(positions, voxel_size: float):
    """Hash a (N, 3) array of world positions to int64 voxel-cell ids
    via simple bit packing. Out-of-range cells (very rare) get clipped
    at the bias boundary."""
    import numpy as np
    ix = np.floor(positions[:, 0] / voxel_size).astype(np.int64) + _VOXEL_BIAS
    iy = np.floor(positions[:, 1] / voxel_size).astype(np.int64) + _VOXEL_BIAS
    iz = np.floor(positions[:, 2] / voxel_size).astype(np.int64) + _VOXEL_BIAS
    np.clip(ix, 0, _VOXEL_MASK, out=ix)
    np.clip(iy, 0, _VOXEL_MASK, out=iy)
    np.clip(iz, 0, _VOXEL_MASK, out=iz)
    return (iz << 42) | (iy << 21) | ix


def _pairwise_overlap(hash_sets):
    """Sum of |V_i ∩ V_j| over all pairs (i, j). Each entry of
    hash_sets is a sorted unique int64 array."""
    import numpy as np
    total = 0
    n = len(hash_sets)
    for i in range(n):
        for j in range(i + 1, n):
            total += int(np.intersect1d(hash_sets[i], hash_sets[j],
                                         assume_unique=True).size)
    return total


def _autotune_voxel_overlap(*, session_id: str, pose_dir: str,
                             frames: list[int], fit_space: str,
                             voxel_sizes: list[float],
                             init_map: dict, stride: int,
                             model_meta: dict) -> dict:
    """Coarse-to-fine voxel-overlap maximisation. Each frame gets its
    own 2D affine `(a, b)`; with N frames we optimise 2N parameters by
    Powell. The objective is the pairwise voxel-overlap sum
    sum_{i<j} |V_i ∩ V_j|, negated for `scipy.optimize.minimize`. We
    run one Powell pass per voxel size in `voxel_sizes`; the warm-up
    coarse stage gets close to the basin and the fine stage refines.

    Bounds are conservative ([0.2, 3.0] for `a`, [-1.5, 1.5] for `b`)
    and Powell respects them via scipy.optimize.minimize since 1.5."""
    import numpy as np
    from scipy.optimize import minimize  # type: ignore

    # Pre-decode each frame once; the optimiser only reapplies (a, b).
    rays_per_frame = []
    raws_per_frame = []
    origin_per_frame = []
    for idx in frames:
        cam, rays, raw = _build_frame_rays_and_raw(
            session_id, pose_dir, idx, model_meta, stride,
        )
        if raw.size < 50:
            raise ValueError(f"frame {idx}: only {raw.size} pixels with valid model_raw "
                             f"after stride={stride}; pick a smaller stride")
        rays_per_frame.append(rays)
        raws_per_frame.append(raw)
        origin_per_frame.append(cam)

    n_frames = len(frames)

    def project(idx_frame, a, b):
        d = _depth_from_affine(raws_per_frame[idx_frame], a, b, fit_space)
        return origin_per_frame[idx_frame] + d[:, None] * rays_per_frame[idx_frame]

    eval_count = [0]

    def make_objective(voxel_size):
        def f(params):
            eval_count[0] += 1
            hash_arrays = []
            for i in range(n_frames):
                a = float(params[2*i]); b = float(params[2*i + 1])
                pos = project(i, a, b)
                hashes = _voxel_hashes(pos, voxel_size)
                hash_arrays.append(np.unique(hashes))
            return -_pairwise_overlap(hash_arrays)
        return f

    # Initial point: per-frame init from caller (or identity).
    x0 = np.empty(2 * n_frames, dtype=np.float64)
    for i, idx in enumerate(frames):
        a0, b0 = init_map.get(idx, (1.0, 0.0))
        x0[2*i]     = float(a0)
        x0[2*i + 1] = float(b0)

    bounds = [(0.2, 3.0), (-1.5, 1.5)] * n_frames
    stages_out = []
    x = x0
    for vs in voxel_sizes:
        eval_count[0] = 0
        # Per-stage budget: small for the coarse pass (objective is
        # plateau-laden but the basin is wide), more generous for fine.
        maxiter = 200 if vs < 0.10 else 120
        res = minimize(
            make_objective(vs), x,
            method="Powell",
            bounds=bounds,
            options={"xtol": 1e-3, "ftol": 1e-3, "maxiter": maxiter,
                     "disp": False},
        )
        x = res.x
        stages_out.append({
            "voxel_size": float(vs),
            "overlap": int(-res.fun) if np.isfinite(res.fun) else 0,
            "n_evals": int(eval_count[0]),
        })

    per_frame: dict[str, dict] = {}
    for i, idx in enumerate(frames):
        per_frame[str(idx)] = {
            "a": float(x[2*i]),
            "b": float(x[2*i + 1]),
            "fit_space": fit_space,
        }
    return {
        "frames": frames,
        "fit_space": fit_space,
        "voxel_sizes": voxel_sizes,
        "stride": stride,
        "stages": stages_out,
        "per_frame": per_frame,
        "n_frames": n_frames,
    }


def _truncated_chamfer_pair(pts_a, pts_b, threshold: float) -> float:
    """Symmetric truncated mean Chamfer-squared distance between two
    point clouds. For each a in A: nearest neighbour in B; clamp at
    `threshold`. Same for B → A. Average each direction (so the score
    is independent of point counts) and add. Pixels with no real
    correspondence in the other cloud max out at threshold², which
    keeps partial-overlap regions from dominating the loss."""
    import numpy as np
    from scipy.spatial import cKDTree  # type: ignore
    tree_a = cKDTree(pts_a)
    tree_b = cKDTree(pts_b)
    # Bound the upper search radius via `distance_upper_bound`; misses
    # come back as `inf`, which we then clip to the threshold. Faster
    # than letting the tree explore arbitrarily far.
    da, _ = tree_b.query(pts_a, k=1, distance_upper_bound=threshold * 1.05)
    db, _ = tree_a.query(pts_b, k=1, distance_upper_bound=threshold * 1.05)
    da = np.minimum(np.where(np.isfinite(da), da, threshold), threshold)
    db = np.minimum(np.where(np.isfinite(db), db, threshold), threshold)
    return float((da * da).mean() + (db * db).mean())


def _autotune_chamfer(*, session_id: str, pose_dir: str,
                      frames: list[int], fit_space: str,
                      thresholds: list[float], init_map: dict,
                      stride: int, model_meta: dict) -> dict:
    """Pairwise truncated-Chamfer minimisation across the selected
    frames. The objective is much smoother than voxel-overlap (no
    flat plateaus), so a single Powell pass per threshold typically
    converges; we run one pass per threshold in `thresholds`,
    annealing from a wide acceptance window down to a tight one."""
    import numpy as np
    from scipy.optimize import minimize  # type: ignore

    rays_per_frame = []
    raws_per_frame = []
    origin_per_frame = []
    for idx in frames:
        cam, rays, raw = _build_frame_rays_and_raw(
            session_id, pose_dir, idx, model_meta, stride,
        )
        if raw.size < 50:
            raise ValueError(
                f"frame {idx}: only {raw.size} valid model_raw pixels at "
                f"stride={stride}; pick a smaller stride"
            )
        rays_per_frame.append(rays)
        raws_per_frame.append(raw)
        origin_per_frame.append(cam)

    n_frames = len(frames)

    def project(i, a, b):
        d = _depth_from_affine(raws_per_frame[i], a, b, fit_space)
        return origin_per_frame[i] + d[:, None] * rays_per_frame[i]

    eval_count = [0]

    def make_objective(threshold: float):
        def f(params):
            eval_count[0] += 1
            clouds = []
            for i in range(n_frames):
                a = float(params[2*i]); b = float(params[2*i + 1])
                clouds.append(project(i, a, b))
            total = 0.0
            for i in range(n_frames):
                for j in range(i + 1, n_frames):
                    total += _truncated_chamfer_pair(clouds[i], clouds[j], threshold)
            return total
        return f

    x0 = np.empty(2 * n_frames, dtype=np.float64)
    for i, idx in enumerate(frames):
        a0, b0 = init_map.get(idx, (1.0, 0.0))
        x0[2*i]     = float(a0)
        x0[2*i + 1] = float(b0)

    bounds = [(0.2, 3.0), (-1.5, 1.5)] * n_frames
    stages_out = []
    x = x0
    for thr in thresholds:
        eval_count[0] = 0
        # Powell's defaults run away on the smooth Chamfer landscape
        # (we observed 2000+ evals per stage at small thresholds, taking
        # ~30s each). Cap both maxiter (line searches) and maxfev
        # (function calls) hard so a single click returns in a couple
        # of seconds — there is no benefit chasing 1e-6 precision when
        # the bare Chamfer objective has known degenerate basins anyway.
        res = minimize(
            make_objective(thr), x,
            method="Powell",
            bounds=bounds,
            options={"xtol": 5e-3, "ftol": 5e-4, "maxiter": 40,
                     "maxfev": 250, "disp": False},
        )
        x = res.x
        stages_out.append({
            "threshold_m": float(thr),
            "loss": float(res.fun) if np.isfinite(res.fun) else None,
            "n_evals": int(eval_count[0]),
        })

    per_frame: dict[str, dict] = {}
    for i, idx in enumerate(frames):
        per_frame[str(idx)] = {
            "a": float(x[2*i]),
            "b": float(x[2*i + 1]),
            "fit_space": fit_space,
        }
    return {
        "frames": frames,
        "fit_space": fit_space,
        "thresholds": thresholds,
        "stride": stride,
        "stages": stages_out,
        "per_frame": per_frame,
        "n_frames": n_frames,
    }


def _features_meta_for_pose_dir(pose_dir: str) -> str:
    """Map a pose-dir name to the matching features_meta sidecar.
    `frames` → features_meta.json,
    `frames_aligned` → features_meta_aligned.json,
    `frames_feature_ba_aligned` → features_meta_feature_ba_aligned.json.
    Mirrors depth_refine.py's `suffix = pose_dir.removeprefix('frames')`."""
    suffix = pose_dir.removeprefix("frames")
    if not suffix:
        return "features_meta.json"
    return f"features_meta{suffix}.json"


def _parse_int_csv(s: str) -> set[int]:
    """Parse a `,`-separated list of integers; ignore empty / non-int parts.
    Returns a *set* (auto-tune handles each frame once even if listed twice)."""
    out: set[int] = set()
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.add(int(part))
        except ValueError:
            continue
    return out


def _find_common_features(meta: dict, frame_set: set[int]) -> list[dict]:
    """Walk every voxel.features in `meta` and return those with at least
    one observation in *every* index from `frame_set`. Returned shape:
    [{ 'world': [x,y,z], 'obs': {frame_idx: [u, v]} }]. The features_meta
    schema (per feature_ray_reconstruct.py) places features under a
    voxel-aggregation tree but each feature carries its own world point
    + obs list — voxelisation is purely spatial bookkeeping that we can
    flatten away."""
    out: list[dict] = []
    for v in meta.get("voxels", []):
        for feat in v.get("features", []):
            obs_by_frame: dict[int, list[float]] = {}
            for ob in feat.get("obs", []):
                fi = int(ob["frame"])
                obs_by_frame[fi] = [float(ob["u"]), float(ob["v"])]
            if frame_set.issubset(obs_by_frame.keys()):
                out.append({
                    "world": [float(c) for c in feat["world"]],
                    "obs": obs_by_frame,
                })
    return out


def _bilinear_sample_2d(img, x_arr, y_arr):
    """Bilinearly sample `img` (H, W) at fractional pixel coords. Out-of-
    bounds reads return NaN. Lifted from depth_refine._bilinear_sample —
    inlined here so serve.py doesn't need to import depth_refine (which
    drags in PIL/torch on `--help`)."""
    import numpy as np
    h, w = img.shape
    inside = (x_arr >= 0) & (x_arr <= w - 1) & (y_arr >= 0) & (y_arr <= h - 1)
    xc = np.clip(x_arr, 0, w - 1 - 1e-3)
    yc = np.clip(y_arr, 0, h - 1 - 1e-3)
    x0 = np.floor(xc).astype(np.int32); y0 = np.floor(yc).astype(np.int32)
    x1 = np.minimum(x0 + 1, w - 1);     y1 = np.minimum(y0 + 1, h - 1)
    fx = (xc - x0).astype(np.float32);  fy = (yc - y0).astype(np.float32)
    s = ((img[y0, x0] * (1 - fx) + img[y0, x1] * fx) * (1 - fy)
         + (img[y1, x0] * (1 - fx) + img[y1, x1] * fx) * fy)
    return np.where(inside, s, np.nan)


def _autotune_one_frame(session_id: str, pose_dir: str, idx: int,
                        *, common_features: list, model_meta: dict,
                        fit_space: str) -> dict:
    """Solve a per-frame affine `(a, b)` so the cached model_raw depths at
    common-feature pixels best match the BA-triangulated world points'
    camera-Z perpendicular distances.

    Two solver branches:
      * fit_space='depth':     a · M_f + b ≈ target_depth_f      (linear LS)
      * fit_space='disparity': a · (1/M_f) + b ≈ 1/target_depth_f
                               then  d = M / (a + b·M)             (linear LS in 1/d)

    target_depth_f = -(V_w2c · [P_world, 1])[z] = perpendicular distance
    from the camera plane to the BA point in frame `idx`. NaN-sampled
    pixels (feature pixel outside the cached prediction) are dropped.
    Frames with too few common features (<3) get a friendly skip
    rather than a noisy fit."""
    import numpy as np
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import fusion  # noqa: E402

    f_path = FRAMES_DIR / session_id / pose_dir / f"frame_{idx:06d}.bin"
    if not f_path.exists():
        raise FileNotFoundError(f"missing {pose_dir}/frame_{idx:06d}.bin")
    raw_path = FRAMES_DIR / session_id / "model_raw" / f"frame_{idx:06d}.f16"
    if not raw_path.exists():
        raise FileNotFoundError(f"model_raw cache missing for frame {idx}")
    entry = model_meta.get(str(idx))
    if entry is None:
        raise FileNotFoundError(f"model_raw/index.json has no entry for frame {idx}")
    cw_c = int(entry["w"]); ch_c = int(entry["h"])
    pred_full = (np.frombuffer(raw_path.read_bytes(), dtype=np.float16)
                   .astype(np.float32, copy=False).reshape(ch_c, cw_c))

    frame = parse_frame(f_path.read_bytes())
    V = fusion._mat4_from_column_major(frame["viewMatrix"])      # world_from_view
    V_w2c = np.linalg.inv(V)

    Ms = []
    Ts = []
    for feat in common_features:
        u, v = feat["obs"].get(idx, (None, None))
        if u is None:
            continue
        # Sample model prediction at (u·cw, v·ch_) — same colour-grid
        # convention pred_full was written under (cache_model_raw.py
        # passes rgba directly to PIL, so pred_full[row=0] = scene-
        # bottom = norm-view UV v=0).
        sx = float(u) * cw_c - 0.5
        sy = float(v) * ch_c - 0.5
        sample = _bilinear_sample_2d(pred_full,
                                      np.asarray([sx]), np.asarray([sy]))
        m = float(sample[0])
        if not np.isfinite(m) or m <= 1e-3:
            continue
        Pw = np.asarray(feat["world"], dtype=np.float64)
        Ph = np.array([Pw[0], Pw[1], Pw[2], 1.0], dtype=np.float64)
        p_view = V_w2c @ Ph
        target = -float(p_view[2])    # camera-Z perpendicular distance
        if not np.isfinite(target) or target < 0.05 or target > 50.0:
            continue
        Ms.append(m); Ts.append(target)

    n = len(Ms)
    if n < 3:
        return {"a": 1.0, "b": 0.0, "n_features": n,
                 "skipped": "too few common features for this frame"}

    M = np.asarray(Ms, dtype=np.float64)
    T = np.asarray(Ts, dtype=np.float64)
    if fit_space == "disparity":
        A = np.stack([1.0 / M, np.ones_like(M)], axis=-1)   # (n, 2)
        y = 1.0 / T
        sol, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        d_pred = M / np.where(a + b * M > 1e-3, a + b * M, 1.0)
        residual = float(np.median(np.abs(d_pred - T)))
    else:
        A = np.stack([M, np.ones_like(M)], axis=-1)         # (n, 2)
        sol, *_ = np.linalg.lstsq(A, T, rcond=None)
        a, b = float(sol[0]), float(sol[1])
        d_pred = a * M + b
        residual = float(np.median(np.abs(d_pred - T)))
    return {"a": a, "b": b, "n_features": n,
            "residual_m": round(residual, 4),
            "fit_space": fit_space}


def _build_pixel_cloud_payload(
    body: bytes,
    depth_kind: str,
    stride: int,
    near: float,
    far: float,
    model_raw_path: Path | None,
    model_raw_meta: Path | None,
    idx: int,
    sigma_frac: float = 0.03,
) -> dict:
    """Compute one frame's pixel cloud as ray directions + depths + colors,
    so the client can re-apply slider-driven (a, b) without re-fetching.

    For depth_kind='phone' the source grid is the depth buffer (dw, dh)
    and 'depth' is the decoded WebXR depth in metres. For depth_kind='model'
    the source grid is the colour image (cw, ch) and 'depth' is the cached
    raw Depth-Anything-V2 prediction in (approximate) metres — the client
    multiplies by a + b·M (or applies the disparity-space correction) to
    get the refined depth. For depth_kind='blend' the source grid is again
    the colour image and 'depth' is the on-the-fly Gaussian detail-injection
    blend in metres — clients should treat it like phone (a=1, b=0).

    Returns a JSON-serialisable dict with:
        origin           [x, y, z]    camera position in world
        forward          [x, y, z]    camera +forward in world
        frustum_world    8 corners    (4 near + 4 far)
        depth_kind       echoed back
        grid             [w, h]       sampling grid before stride
        stride           effective stride applied
        count            N            number of points
        dirs             3N floats    world-space unit ray dirs
        depths           N  floats    raw depth value (phone meters or model M)
        colors           3N ints      RGB 0–255
        near, far        depth window for client-side filter hint

    The default stride is 4 for model (color grid is high-res) and 1 for
    phone (depth grid is already low-res).
    """
    import numpy as np
    sys.path.insert(0, str(PROJECT_ROOT / "tools"))
    import fusion  # noqa: E402

    frame = parse_frame(body)
    dw = int(frame["width"]); dh = int(frame["height"])
    cw = int(frame["color_width"]); ch_ = int(frame["color_height"])
    V  = fusion._mat4_from_column_major(frame["viewMatrix"])
    P  = fusion._mat4_from_column_major(frame["projectionMatrix"])
    Bd = fusion._mat4_from_column_major(frame["normDepthBufferFromNormView"])
    Bv = np.linalg.inv(Bd)
    P_inv = np.linalg.inv(P)
    cam = V[:3, 3].astype(np.float64)
    forward = (-V[:3, 2]).astype(np.float64)

    if depth_kind == "phone":
        if stride <= 0:
            stride = 1
        depth_arr = fusion.decode_depth(
            frame["depth"], dw, dh,
            int(frame["format"]), float(frame["rawValueToMeters"]),
        )
        gw, gh = dw, dh

        # Subsampled (gx, gy) grid on the depth buffer.
        gxs = np.arange(0, gw, stride, dtype=np.float64)
        gys = np.arange(0, gh, stride, dtype=np.float64)
        bx, by = np.meshgrid(gxs, gys, indexing="xy")
        d = depth_arr[by.astype(np.int32), bx.astype(np.int32)]

        # depth-buffer pixel center → norm-view UV via Bd⁻¹ (matches the
        # Bd convention in fusion.py: u_d horizontally flipped, v_d direct).
        u_d = 1.0 - (bx + 0.5) / gw
        v_d = (by + 0.5) / gh
        nd_h = np.stack([u_d, v_d, np.zeros_like(u_d), np.ones_like(u_d)],
                        axis=-1)
        nv_h = nd_h @ Bv.T
        safe = np.where(np.abs(nv_h[..., 3]) > 1e-12, nv_h[..., 3], 1.0)
        u = nv_h[..., 0] / safe
        v = nv_h[..., 1] / safe
    elif depth_kind in ("model", "blend"):
        if stride <= 0:
            stride = 4
        if model_raw_path is None or not model_raw_path.exists():
            raise FileNotFoundError(
                f"model_raw cache missing for frame {idx}: "
                f"run tools/cache_model_raw.py --session <id>"
            )
        # Look up (cw_cached, ch_cached) — the cache might have been built
        # against an older capture session with a different colour resolution
        # than the current frame.bin. We trust the cache's dims.
        if model_raw_meta is None or not model_raw_meta.exists():
            raise FileNotFoundError(
                f"model_raw/index.json missing — re-run cache_model_raw.py"
            )
        meta = json.loads(model_raw_meta.read_text())
        entry = meta.get(str(idx))
        if entry is None:
            raise FileNotFoundError(
                f"model_raw cache has no entry for frame {idx}"
            )
        cw_c = int(entry["w"]); ch_c = int(entry["h"])
        model_arr = np.frombuffer(
            model_raw_path.read_bytes(), dtype=np.float16,
        ).astype(np.float32, copy=False).reshape(ch_c, cw_c)

        if depth_kind == "model":
            depth_arr = model_arr
        else:
            # Build the blend on the colour-image grid (in NATURAL display
            # orientation) and flip rows so the resulting array follows
            # the same GL-style convention as model_arr — row 0 = scene
            # bottom = norm-view v=0, matching the (cy+0.5)/gh sampling
            # below. Sigma is a fraction of the grid diagonal.
            import math
            phone_grid, model_grid = _sample_phone_model_on_color_grid(
                body, out_w=cw_c, out_h=ch_c, model_raw_arr=model_arr,
            )
            sigma_px = max(1.0, float(sigma_frac) * math.hypot(cw_c, ch_c))
            blend_natural, *_ = _compute_blend_metres(
                phone_grid, model_grid, sigma_px,
            )
            depth_arr = np.ascontiguousarray(blend_natural[::-1, :])
        gw, gh = cw_c, ch_c

        gxs = np.arange(0, gw, stride, dtype=np.float64)
        gys = np.arange(0, gh, stride, dtype=np.float64)
        cx, cy = np.meshgrid(gxs, gys, indexing="xy")
        d = depth_arr[cy.astype(np.int32), cx.astype(np.int32)]

        # Colour-image pixel → norm-view UV directly (no Bd needed since the
        # colour image *is* the view image): u = (cx+0.5)/cw, v = (cy+0.5)/ch.
        u = (cx + 0.5) / gw
        v = (cy + 0.5) / gh
    else:
        raise ValueError(f"bad depth_kind {depth_kind!r}")

    # norm-view UV → view-space ray. We do NOT normalise to a unit ray —
    # the canonical convention in this codebase (fusion.frame_to_world_points)
    # treats `depth` as camera-Z (perpendicular) distance, not radial. A
    # unit-ray formula `cam + d · unit_ray` puts off-axis pixels at the
    # wrong distance (≈cos(θ) too close at corners, where θ is the angle
    # from the optical axis). Instead we scale `view3` so view3.z = −1,
    # giving rays such that  view_pt = view3_scaled · depth  has
    # view_pt.z = −depth, which is exactly what the W3C depth spec
    # promises. The cost: dirs are no longer unit vectors but length
    # 1/|cos θ|, which is fine because the client just multiplies by
    # depth. Without this fix phone-depth pixels project visibly far
    # (especially at the edges of the FOV).
    x_ndc = 2.0 * u - 1.0
    y_ndc = 2.0 * v - 1.0
    clip = np.stack([x_ndc, y_ndc, -np.ones_like(x_ndc),
                     np.ones_like(x_ndc)], axis=-1)
    view4 = clip @ P_inv.T
    safe_w = np.where(np.abs(view4[..., 3:4]) > 1e-12, view4[..., 3:4], 1.0)
    view3 = view4[..., :3] / safe_w
    near_z = view3[..., 2:3]
    near_z_safe = np.where(np.abs(near_z) > 1e-6, near_z, -1e-6)
    view_dirs = view3 / -near_z_safe   # view3.z normalised to -1
    rays_world = view_dirs @ V[:3, :3].T   # (gh', gw', 3)

    # Sample colour image at norm-view UV (u, v). The rgba memory is in
    # WebGL convention (origin bottom-left, row 0 = bottom of view), so
    # row index = v · ch_ — NOT (1 − v) · ch_. Mirrors the canonical
    # sampling in fusion.frame_to_world_points (color_img[cy, cx]
    # with cy = Vv · ch_). Getting this wrong vertically flips the
    # rendered colours about the image centre, which presents as "wall
    # pixels appear at the floor in 3D".
    if frame["color"] is not None and cw > 0 and ch_ > 0:
        rgba = np.frombuffer(frame["color"], dtype=np.uint8).reshape(ch_, cw, 4)
        sx = np.clip((u * cw).astype(np.int32), 0, cw - 1)
        sy = np.clip((v * ch_).astype(np.int32), 0, ch_ - 1)
        rgb = rgba[sy, sx, :3]
    else:
        rgb = np.full((d.shape[0], d.shape[1], 3), 128, dtype=np.uint8)

    # Drop pixels with degenerate depth so the JSON stays small. For phone
    # we drop d≤0 (zeros = unobserved); for model we keep the raw signal
    # since the client may want to inspect even out-of-range predictions.
    # Blend is already in metres so the same (near, far) window as phone
    # applies — no need to drag invalid faraway points into the cloud.
    if depth_kind in ("phone", "blend"):
        mask = (d > near) & (d < far) & np.isfinite(d)
    else:
        mask = np.isfinite(d) & (d > 1e-3)
    flat_mask = mask.reshape(-1)
    flat_rays = rays_world.reshape(-1, 3)[flat_mask]
    flat_depths = d.reshape(-1)[flat_mask].astype(np.float32)
    flat_rgb = rgb.reshape(-1, 3)[flat_mask]

    # Round positions to 4 decimal places (sub-mm) to keep JSON compact.
    return {
        "idx": idx,
        "depth_kind": depth_kind,
        "grid": [int(gw), int(gh)],
        "stride": int(stride),
        "near": float(near),
        "far": float(far),
        "origin": [float(cam[0]), float(cam[1]), float(cam[2])],
        "forward": [float(forward[0]), float(forward[1]), float(forward[2])],
        "frustum_world": _frustum_corners(P_inv, V, near=0.1, far=2.0),
        "count": int(flat_depths.size),
        "dirs": _round_floats(flat_rays.astype(np.float32).reshape(-1).tolist(), 5),
        "depths": _round_floats(flat_depths.tolist(), 4),
        "colors": flat_rgb.reshape(-1).astype(int).tolist(),
    }


def _round_floats(seq: list, ndigits: int) -> list:
    """Cheaply truncate float precision in a list before JSON-encoding,
    so the wire payload doesn't carry meaningless trailing digits.
    Plain `round(x, n)` keeps the float type — no string formatting cost."""
    return [round(float(x), ndigits) for x in seq]


def _frustum_corners(P_inv, V, near: float, far: float) -> list:
    """Compute 8 frustum-edge corners in world coords (4 at near, 4 at far)
    plus the camera origin, for the JS to draw a wireframe pyramid."""
    import numpy as np
    corners = []
    for d_clip in (-1.0, 1.0):  # -1 = near, +1 = far in clip z (after /w)
        for u, v in ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)):
            clip = np.array([2*u - 1, 2*v - 1, d_clip, 1.0], dtype=np.float64)
            view = P_inv @ clip
            view3 = view[:3] / (view[3] if abs(view[3]) > 1e-12 else 1.0)
            ray = view3 / max(np.linalg.norm(view3), 1e-12)
            ray_w = V[:3, :3] @ ray
            t = near if d_clip < 0 else far
            world = V[:3, 3] + t * ray_w
            corners.append([float(world[0]), float(world[1]), float(world[2])])
    return corners


# --------------------------------------------------------------------------

def build_ssl_context(cert: Path, key: Path) -> ssl.SSLContext:
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.minimum_version = ssl.TLSVersion.TLSv1_2
    ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))
    try:
        ctx.set_alpn_protocols(["http/1.1"])
    except (AttributeError, NotImplementedError):
        pass
    return ctx


def run_server(port: int, use_https: bool, cert_dir: Path, voxel_size_m: float) -> None:
    if not WEB_DIR.is_dir():
        print(f"ERROR: {WEB_DIR} not found.", file=sys.stderr)
        sys.exit(1)
    init_voxel_room(voxel_size_m)
    os.chdir(WEB_DIR)

    ip = primary_ipv4()
    host = hostname_local()

    httpd = http.server.ThreadingHTTPServer(("0.0.0.0", port), RoomgameHandler)

    def say(s: str = "") -> None:
        sys.stderr.write(s + "\n")
        sys.stderr.flush()

    scheme = "https" if use_https else "http"
    phone_url = f"{scheme}://{ip}:{port}/"
    say(f"\nServing {WEB_DIR} over {scheme.upper()}")
    say(f"  On this Mac:             {scheme}://localhost:{port}/")
    if host:
        say(f"  On the same Wi-Fi:       {scheme}://{host}:{port}/")
    say(f"  On the same Wi-Fi (IP):  {phone_url}")
    say(f"  Phone landing:           {phone_url}")
    say(f"  Laptop game/viewer:      {scheme}://localhost:{port}/game.html")
    say(f"  Laptop cube viewer:      {scheme}://localhost:{port}/cubeview.html")

    say("\nScan this QR code on your phone (opens the landing page):\n")
    print_qr_to_stderr(phone_url)

    if use_https:
        sans = ["localhost", "127.0.0.1", ip]
        if host:
            sans.append(host)
        sans = sorted(set(sans))
        cert, key = ensure_cert(cert_dir, sans)
        ctx = build_ssl_context(cert, key)
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True)
        say("\nHTTPS uses a self-signed cert. Chrome will show a warning — tap")
        say("'Advanced → Proceed'. WebXR refuses non-secure origins on the phone,")
        say("so HTTPS (or a tunnel, or adb reverse) is required.")
    else:
        say("\nHTTP works on this Mac (localhost is a secure context). For the")
        say("phone scanner, use one of: --https, `cloudflared tunnel --url`,")
        say("`ngrok http <port>`, or `adb reverse tcp:<port> tcp:<port>`.")

    say("\nCtrl-C to stop.\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--https", action="store_true", help="serve HTTPS with a self-signed cert")
    ap.add_argument("--port", type=int, default=None,
                    help="port to listen on (default: 8080 for HTTP, 8443 for HTTPS)")
    ap.add_argument("--cert-dir", default=str(DEFAULT_CERT_DIR),
                    help="where to store the generated cert/key (HTTPS only)")
    ap.add_argument("--voxel-size", type=float, default=0.02,
                    help="voxel edge length in metres (default: 0.02 = 2 cm)")
    args = ap.parse_args()
    port = args.port if args.port is not None else (8443 if args.https else 8080)
    cert_dir = Path(args.cert_dir).expanduser()
    run_server(port=port, use_https=args.https, cert_dir=cert_dir, voxel_size_m=args.voxel_size)


if __name__ == "__main__":
    main()
