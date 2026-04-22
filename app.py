"""
Bambu camera → Obico failure detection → telemetry overlay → MJPEG stream.

Single container, single process.  Publishes detection state via MQTT
with Home Assistant auto-discovery.

Endpoints:
  GET  /stream   — MJPEG stream for Home Assistant
  GET  /snapshot — single JPEG frame
  GET  /health   — liveness check
  POST /p/       — upload an image for ad-hoc detection
  GET  /p/?img=  — fetch a URL for ad-hoc detection

MQTT topics (published to HA broker):
  bambu_obico/<serial>/state      — detection state JSON
  bambu_obico/<serial>/available  — online/offline
  homeassistant/...               — auto-discovery configs
"""

import json, logging, os, socket, ssl, struct, threading, time
from io import BytesIO

import cv2
import httpx
import numpy as np
import onnxruntime as ort
import paho.mqtt.client as mqtt
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, ImageDraw, ImageFont

# ── config ────────────────────────────────────────────────────────────────────

BAMBU_HOST = os.environ["BAMBU_HOST"]
BAMBU_CODE = os.environ["BAMBU_CODE"]
BAMBU_SERIAL = os.environ["BAMBU_SERIAL"]
BAMBU_CAM_PORT = int(os.environ.get("BAMBU_CAM_PORT", "6000"))
BAMBU_MQTT_PORT = int(os.environ.get("BAMBU_MQTT_PORT", "8883"))

# MQTT broker (e.g. eclipse-mosquitto)
MQTT_HOST = os.environ.get("MQTT_HOST", "")
MQTT_PORT = int(os.environ.get("MQTT_PORT", "1883"))
MQTT_USER = os.environ.get("MQTT_USER", "")
MQTT_PASS = os.environ.get("MQTT_PASS", "")

DETECTION_THRESH = float(os.environ.get("DETECTION_THRESHOLD", "0.20"))
ALERT_THRESH = float(os.environ.get("ALERT_THRESHOLD", "0.40"))
NMS_THRESH = float(os.environ.get("NMS_THRESHOLD", "0.45"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "85"))
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/model/model.onnx")
NAMES_PATH = os.environ.get("NAMES_PATH", "/app/model/names")

MQTT_TOPIC_PREFIX = f"bambu_obico/{BAMBU_SERIAL}"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("bambu-obico")

# ── ONNX model ────────────────────────────────────────────────────────────────

log.info("Loading ONNX model from %s", MODEL_PATH)
_session = ort.InferenceSession(MODEL_PATH)
_inp = _session.get_inputs()[0]
_, _, _model_h, _model_w = _inp.shape
with open(NAMES_PATH) as f:
    _names = [l.strip() for l in f if l.strip()]
log.info("Model ready — input %s, labels %s", _inp.shape, _names)


def detect(image_bytes: bytes) -> list[dict]:
    img = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    h, w = img.shape[:2]

    blob = cv2.resize(img, (_model_w, _model_h))
    blob = cv2.cvtColor(blob, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    blob = blob.transpose(2, 0, 1)[np.newaxis]

    outputs = _session.run(None, {_inp.name: blob})
    boxes = np.array(outputs[0]).squeeze(0)
    confs = np.array(outputs[1]).squeeze(0)

    dets = []
    for i in range(boxes.shape[0]):
        for c in range(confs.shape[1]):
            conf = float(confs[i, c])
            if conf < DETECTION_THRESH:
                continue
            cx, cy, bw, bh = boxes[i, c]
            dets.append({
                "label": _names[c] if c < len(_names) else f"class_{c}",
                "confidence": round(conf, 3),
                "box": [
                    round(float(cx) * w, 1), round(float(cy) * h, 1),
                    round(float(bw) * w, 1), round(float(bh) * h, 1),
                ],
            })
    dets.sort(key=lambda d: d["confidence"], reverse=True)
    return _nms(dets, NMS_THRESH)


def _iou(a, b):
    ax1, ay1, ax2, ay2 = a[0]-a[2]/2, a[1]-a[3]/2, a[0]+a[2]/2, a[1]+a[3]/2
    bx1, by1, bx2, by2 = b[0]-b[2]/2, b[1]-b[3]/2, b[0]+b[2]/2, b[1]+b[3]/2
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    union = a[2]*a[3] + b[2]*b[3] - ix*iy
    return (ix * iy / union) if union > 0 else 0


def _nms(dets, thresh):
    keep = []
    for d in dets:
        if all(_iou(d["box"], k["box"]) < thresh for k in keep):
            keep.append(d)
    return keep


# ── Bambu MQTT (telemetry from printer) ───────────────────────────────────────

_state: dict = {}
_state_lock = threading.Lock()

_MQTT_KEYS = (
    "nozzle_temper", "nozzle_target_temper", "bed_temper", "bed_target_temper",
    "chamber_temper", "mc_percent", "mc_remaining_time", "layer_num",
    "total_layer_num", "gcode_state", "spd_lvl", "subtask_name",
    "gcode_file", "wifi_signal",
)
SPEED_LVL = {1: "Silent", 2: "Standard", 3: "Sport", 4: "Ludicrous"}
STATE_COLOR = {
    "RUNNING": (76, 217, 100), "FINISH": (76, 217, 100),
    "PAUSE": (255, 204, 0), "PREPARE": (255, 149, 0),
    "FAILED": (255, 69, 58), "IDLE": (152, 152, 157),
}
ACCENT = (0, 180, 255)


def _bambu_mqtt_thread():
    report = f"device/{BAMBU_SERIAL}/report"
    request = f"device/{BAMBU_SERIAL}/request"
    pushall = json.dumps({"pushing": {"sequence_id": "1", "command": "pushall"}})

    while True:
        try:
            c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                            client_id=f"obico-telemetry-{os.getpid()}")
            c.username_pw_set("bblp", BAMBU_CODE)
            c.tls_set(cert_reqs=ssl.CERT_NONE)
            c.tls_insecure_set(True)

            def on_connect(cl, *_a):
                cl.subscribe(report)
                cl.publish(request, pushall)

            def on_message(_cl, _u, msg):
                try:
                    p = json.loads(msg.payload).get("print", {})
                except Exception:
                    return
                if p:
                    with _state_lock:
                        for k in _MQTT_KEYS:
                            if k in p:
                                _state[k] = p[k]

            c.on_connect = on_connect
            c.on_message = on_message
            c.connect(BAMBU_HOST, BAMBU_MQTT_PORT, keepalive=60)
            c.loop_forever()
        except Exception as e:
            log.warning("bambu-mqtt: %s — retrying in 5s", e)
            time.sleep(5)


# ── MQTT (publish detections to Home Assistant) ────────────────────────────

_mqtt_pub: mqtt.Client | None = None
_mqtt_pub_lock = threading.Lock()


def _mqtt_publish(topic: str, payload: str, retain: bool = False):
    with _mqtt_pub_lock:
        if _mqtt_pub and _mqtt_pub.is_connected():
            _mqtt_pub.publish(topic, payload, retain=retain)


def _publish_ha_discovery():
    device = {
        "identifiers": [f"bambu_obico_{BAMBU_SERIAL}"],
        "name": "Bambu Obico",
        "manufacturer": "Bambu Lab",
        "model": "P1P",
        "sw_version": "bambu-obico 1.0",
    }
    serial_short = BAMBU_SERIAL[-6:]
    avail = {
        "availability_topic": f"{MQTT_TOPIC_PREFIX}/available",
        "payload_available": "online",
        "payload_not_available": "offline",
    }

    # Binary sensor: failure detected (on/off at ALERT_THRESH)
    _mqtt_publish(
        f"homeassistant/binary_sensor/bambu_obico_{serial_short}/failure/config",
        json.dumps({
            "name": "Print Failure",
            "unique_id": f"bambu_obico_{BAMBU_SERIAL}_failure",
            "state_topic": f"{MQTT_TOPIC_PREFIX}/state",
            "value_template": "{{ value_json.failure }}",
            "payload_on": "ON",
            "payload_off": "OFF",
            "device_class": "problem",
            "device": device,
            "icon": "mdi:printer-3d-nozzle-alert",
            **avail,
        }),
        retain=True,
    )

    # Sensor: max confidence %
    _mqtt_publish(
        f"homeassistant/sensor/bambu_obico_{serial_short}/confidence/config",
        json.dumps({
            "name": "Failure Confidence",
            "unique_id": f"bambu_obico_{BAMBU_SERIAL}_confidence",
            "state_topic": f"{MQTT_TOPIC_PREFIX}/state",
            "value_template": "{{ value_json.max_confidence }}",
            "unit_of_measurement": "%",
            "device": device,
            "icon": "mdi:percent-circle",
            **avail,
        }),
        retain=True,
    )

    # Sensor: detection count
    _mqtt_publish(
        f"homeassistant/sensor/bambu_obico_{serial_short}/count/config",
        json.dumps({
            "name": "Failure Count",
            "unique_id": f"bambu_obico_{BAMBU_SERIAL}_count",
            "state_topic": f"{MQTT_TOPIC_PREFIX}/state",
            "value_template": "{{ value_json.count }}",
            "device": device,
            "icon": "mdi:counter",
            **avail,
        }),
        retain=True,
    )

    log.info("MQTT auto-discovery published")


def _mqtt_pub_thread():
    global _mqtt_pub

    if not MQTT_HOST:
        log.info("MQTT_HOST not set — MQTT publishing disabled")
        return

    while True:
        try:
            c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2,
                            client_id=f"bambu-obico-{os.getpid()}")
            if MQTT_USER:
                c.username_pw_set(MQTT_USER, MQTT_PASS)
            c.will_set(f"{MQTT_TOPIC_PREFIX}/available", "offline", retain=True)

            def on_connect(cl, *_a):
                cl.publish(f"{MQTT_TOPIC_PREFIX}/available", "online", retain=True)
                _publish_ha_discovery()
                log.info("Connected to MQTT at %s:%s", MQTT_HOST, MQTT_PORT)

            c.on_connect = on_connect

            with _mqtt_pub_lock:
                _mqtt_pub = c

            c.connect(MQTT_HOST, MQTT_PORT, keepalive=60)
            c.loop_forever()
        except Exception as e:
            log.warning("mqtt-pub: %s — retrying in 5s", e)
            with _mqtt_pub_lock:
                _mqtt_pub = None
            time.sleep(5)


def publish_detections(dets: list[dict]):
    max_conf = max((d["confidence"] for d in dets), default=0)
    _mqtt_publish(
        f"{MQTT_TOPIC_PREFIX}/state",
        json.dumps({
            "failure": "ON" if max_conf >= ALERT_THRESH else "OFF",
            "max_confidence": round(max_conf * 100, 1),
            "count": len(dets),
            "detections": dets,
        }),
    )


# ── overlay drawing ───────────────────────────────────────────────────────────

def _font(size):
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"):
        try:
            return ImageFont.truetype(p, size)
        except Exception:
            continue
    return ImageFont.load_default()


_FXL, _FLG, _FMD, _FSM = _font(24), _font(19), _font(16), _font(13)


def _tw(draw, text, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0]


def _fmt_eta(mins):
    try:
        m = int(mins)
    except Exception:
        return "--"
    if m <= 0:
        return "0m"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m"


def draw_overlay(img: Image.Image, detections: list[dict]) -> Image.Image:
    W, H = img.size
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    pad = 10

    with _state_lock:
        s = dict(_state)

    gs = str(s.get("gcode_state", "IDLE")).upper()
    color = STATE_COLOR.get(gs, ACCENT)

    for det in detections:
        cx, cy, bw, bh = det["box"]
        x1, y1 = int(cx - bw / 2), int(cy - bh / 2)
        x2, y2 = int(cx + bw / 2), int(cy + bh / 2)
        box_color = (255, 69, 58, 200)
        d.rectangle((x1, y1, x2, y2), outline=box_color, width=2)
        label = f"{det['label']} {det['confidence']:.0%}"
        lw = _tw(d, label, _FSM)
        d.rounded_rectangle((x1, y1 - 20, x1 + lw + 8, y1), radius=4,
                            fill=(255, 69, 58, 180))
        d.text((x1 + 4, y1 - 18), label, font=_FSM, fill=(255, 255, 255, 255))

    if s:
        name = str(s.get("subtask_name") or s.get("gcode_file") or "—")
        for suffix in (".gcode.3mf", ".3mf"):
            if name.lower().endswith(suffix):
                name = name[:-len(suffix)]
        name = name.replace("_", " ")
        if len(name) > 36:
            name = name[:35] + "…"

        title_w = max(_tw(d, name, _FLG), _tw(d, gs, _FMD) + 18) + pad * 2
        title_h = 58
        d.rounded_rectangle((12, 12, 12 + title_w, 12 + title_h), radius=10,
                            fill=(0, 0, 0, 170))
        d.text((12 + pad, 12 + pad - 2), name, font=_FLG, fill=(255, 255, 255, 255))
        dot_y = 12 + pad + 28
        d.ellipse((12 + pad, dot_y + 3, 12 + pad + 10, dot_y + 13),
                  fill=color + (255,))
        d.text((12 + pad + 16, dot_y), gs, font=_FMD, fill=color + (255,))

    if "layer_num" in s or "mc_remaining_time" in s:
        layer_s = f"Layer  {s.get('layer_num', 0)} / {s.get('total_layer_num', '?')}"
        eta_s = f"ETA  {_fmt_eta(s.get('mc_remaining_time', 0))}"
        box_w = max(_tw(d, layer_s, _FMD), _tw(d, eta_s, _FMD)) + pad * 2
        x0 = W - 12 - box_w
        d.rounded_rectangle((x0, 12, x0 + box_w, 70), radius=10,
                            fill=(0, 0, 0, 170))
        d.text((x0 + pad, 18), layer_s, font=_FMD, fill=(255, 255, 255, 255))
        d.text((x0 + pad, 40), eta_s, font=_FMD, fill=ACCENT + (255,))

    if s:
        bar_h = 74
        by = H - bar_h - 12
        bx0, bx1 = 12, W - 12
        d.rounded_rectangle((bx0, by, bx1, by + bar_h), radius=12,
                            fill=(0, 0, 0, 180))

        chips = []
        if "nozzle_temper" in s:
            v = f"{s['nozzle_temper']:.0f}°"
            tgt = s.get("nozzle_target_temper")
            if tgt:
                v += f" / {tgt:.0f}°"
            chips.append(("Nozzle", v))
        if "bed_temper" in s:
            v = f"{s['bed_temper']:.0f}°"
            tgt = s.get("bed_target_temper")
            if tgt:
                v += f" / {tgt:.0f}°"
            chips.append(("Bed", v))
        if "chamber_temper" in s:
            chips.append(("Chamber", f"{s['chamber_temper']:.0f}°"))
        if "spd_lvl" in s:
            chips.append(("Speed", SPEED_LVL.get(int(s["spd_lvl"]), "?")))

        cx = bx0 + 14
        cy = by + 10
        for lbl, val in chips:
            d.text((cx, cy), lbl, font=_FSM, fill=(170, 170, 175, 255))
            d.text((cx, cy + 14), val, font=_FMD, fill=(255, 255, 255, 255))
            cx += max(_tw(d, lbl, _FSM), _tw(d, val, _FMD)) + 26

        pct = max(0, min(100, int(s.get("mc_percent", 0))))
        pb_x0, pb_x1 = bx0 + 14, bx1 - 14
        pb_y0, pb_y1 = by + bar_h - 22, by + bar_h - 10
        d.rounded_rectangle((pb_x0, pb_y0, pb_x1, pb_y1), radius=6,
                            fill=(60, 60, 65, 220))
        fill_w = int((pb_x1 - pb_x0) * pct / 100)
        if fill_w > 0:
            d.rounded_rectangle(
                (pb_x0, pb_y0, pb_x0 + max(fill_w, 12), pb_y1), radius=6,
                fill=color + (255,))
        pct_s = f"{pct}%"
        d.text((pb_x1 - _tw(d, pct_s, _FMD), pb_y0 - 20), pct_s,
               font=_FMD, fill=color + (255,))

    base = img.convert("RGBA")
    return Image.alpha_composite(base, layer).convert("RGB")


# ── camera + pipeline ─────────────────────────────────────────────────────────

_latest_frame: bytes | None = None
_frame_lock = threading.Lock()
_frame_event = threading.Event()

_AUTH = (struct.pack("<IIII", 0x40, 0x3000, 0, 0)
         + b"bblp".ljust(32, b"\x00")
         + BAMBU_CODE.encode().ljust(32, b"\x00"))


def _tls_ctx():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    ctx.set_ciphers("DEFAULT:@SECLEVEL=0")
    return ctx


def _recv(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("printer closed connection")
        buf += chunk
    return bytes(buf)


def _camera_thread():
    global _latest_frame
    backoff = 1

    while True:
        try:
            raw = socket.create_connection((BAMBU_HOST, BAMBU_CAM_PORT), timeout=15)
            raw.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock = _tls_ctx().wrap_socket(raw, server_hostname=BAMBU_HOST)
            sock.sendall(_AUTH)
            log.info("Camera connected to %s:%s", BAMBU_HOST, BAMBU_CAM_PORT)
            backoff = 1

            while True:
                hdr = _recv(sock, 16)
                length = struct.unpack("<I", hdr[:4])[0]
                if not 0 < length <= 4_000_000:
                    raise ValueError(f"bad frame length {length}")

                jpeg_raw = _recv(sock, length)
                if jpeg_raw[:2] != b"\xff\xd8" or jpeg_raw[-2:] != b"\xff\xd9":
                    continue

                dets = detect(jpeg_raw)
                publish_detections(dets)
                if dets:
                    log.info("Detections: %s", dets)

                img = Image.open(BytesIO(jpeg_raw)).convert("RGB")
                img = draw_overlay(img, dets)

                buf = BytesIO()
                img.save(buf, "JPEG", quality=JPEG_QUALITY)
                frame = buf.getvalue()

                with _frame_lock:
                    _latest_frame = frame
                _frame_event.set()

        except Exception as e:
            log.warning("camera: %s — retry in %ss", e, backoff)
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)


# ── FastAPI ───────────────────────────────────────────────────────────────────

app = FastAPI(title="Bambu Obico")


@app.on_event("startup")
def _startup():
    threading.Thread(target=_bambu_mqtt_thread, daemon=True).start()
    threading.Thread(target=_mqtt_pub_thread, daemon=True).start()
    threading.Thread(target=_camera_thread, daemon=True).start()
    log.info("Pipeline started")


def _mjpeg_gen():
    while True:
        _frame_event.wait(timeout=10)
        _frame_event.clear()
        with _frame_lock:
            frame = _latest_frame
        if frame:
            yield (b"--frame\r\n"
                   b"Content-Type: image/jpeg\r\n"
                   b"Content-Length: " + str(len(frame)).encode() + b"\r\n\r\n"
                   + frame + b"\r\n")


@app.get("/stream")
async def stream():
    return StreamingResponse(
        _mjpeg_gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/snapshot")
async def snapshot():
    with _frame_lock:
        frame = _latest_frame
    if not frame:
        raise HTTPException(503, "No frame yet")
    return StreamingResponse(BytesIO(frame), media_type="image/jpeg")


@app.get("/health")
async def health():
    with _state_lock:
        gs = _state.get("gcode_state", "unknown")
    return {"status": "ok", "printer": gs, "has_frame": _latest_frame is not None}


@app.get("/p/")
async def detect_url(img: str = Query(...)):
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.get(img)
            r.raise_for_status()
    except Exception as e:
        raise HTTPException(400, f"Failed to fetch: {e}")
    return JSONResponse(detect(r.content))


@app.post("/p/")
async def detect_upload(file: UploadFile = File(...)):
    return JSONResponse(detect(await file.read()))