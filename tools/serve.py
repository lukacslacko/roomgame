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
frame_count = 0


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
        path = self.path.rstrip("/") or "/"
        if path == "/frame":
            self._handle_frame()
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
        path = self.path.rstrip("/") or "/"
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

    def _handle_frame(self) -> None:
        global frame_count
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

        # Persist the raw body for offline replay (tools/replay.py).
        try:
            FRAMES_DIR.mkdir(parents=True, exist_ok=True)
            (FRAMES_DIR / f"frame_{frame_count:06d}.bin").write_bytes(body)
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
                # ingest_frame returns the number of points written.
                n_written = voxel_room.ingest_frame(frame)
                stats = voxel_room.stats()
                ingest_summary = f" → wrote {n_written}pts, total={stats['voxels']}vox in {stats['chunks']}ch"
            except Exception as e:  # noqa: BLE001
                self._send_text(500, f"{type(e).__name__}: {e}\n")
                return

        color_summary = ""
        if frame["color"] is not None:
            color_summary = f" color={frame['color_width']}x{frame['color_height']}"
        else:
            color_summary = " color=none"

        sys.stderr.write(
            f"[{ip}] FRAME #{frame_count} {frame['width']}x{frame['height']} "
            f"fmt={frame['format']} rawToM={frame['rawValueToMeters']:.6f} "
            f"pose=({pose_translation[0]:+.2f},{pose_translation[1]:+.2f},{pose_translation[2]:+.2f})"
            f"{depth_summary}{color_summary}{ingest_summary}\n"
        )
        self._send_json(200, {"ok": True, "frame": frame_count})


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
    say(f"  Phone scanner:           {phone_url}scan.html")
    say(f"  Laptop game/viewer:      {scheme}://localhost:{port}/game.html")

    say("\nScan this QR code on your phone (open scan.html):\n")
    print_qr_to_stderr(phone_url + "scan.html")

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
