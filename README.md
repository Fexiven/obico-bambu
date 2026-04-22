# bambu-obico

Standalone print failure detection for Bambu Lab printers. Connects to your printer's camera, runs the [Obico](https://github.com/TheSpaghettiDetective/obico-server) ML model on every frame, and serves an MJPEG stream with a live overlay showing print telemetry and detection bounding boxes.

Publishes failure alerts to MQTT with Home Assistant auto-discovery — no cloud, no Obico account needed.

## Quick start

```bash
docker run -d --name bambu-obico -p 3333:3333 \
  -e BAMBU_HOST=<printer-ip> \
  -e BAMBU_CODE=<lan-access-code> \
  -e BAMBU_SERIAL=<serial-number> \
  -e MQTT_HOST=<mqtt-broker> \
  ghcr.io/YOURUSER/bambu-obico:latest
```

Then open `http://localhost:3333/stream` in a browser.

Your LAN access code and serial number are on the printer's screen under Network → LAN Mode.

## Home Assistant

Add the camera:

```yaml
camera:
  - platform: mjpeg
    name: Bambu P1P
    mjpeg_url: http://<host>:3333/stream
    still_image_url: http://<host>:3333/snapshot
```

If `MQTT_HOST` is set and your HA is connected to the same broker, three entities appear automatically:

- **Print Failure** (binary sensor) — on/off
- **Failure Confidence** (sensor) — 0–100%
- **Failure Count** (sensor) — number of detections

See [`homeassistant.yaml`](homeassistant.yaml) for an example notification automation.

## Configuration

All configuration is via environment variables.

| Variable | Default | Description |
|---|---|---|
| `BAMBU_HOST` | | Printer IP address |
| `BAMBU_CODE` | | LAN access code |
| `BAMBU_SERIAL` | | Printer serial number |
| `MQTT_HOST` | | MQTT broker hostname (optional, disables MQTT if unset) |
| `MQTT_PORT` | `1883` | MQTT broker port |
| `MQTT_USER` | | MQTT username |
| `MQTT_PASS` | | MQTT password |
| `DETECTION_THRESHOLD` | `0.20` | Minimum confidence to show a detection |
| `ALERT_THRESHOLD` | `0.40` | Minimum confidence to trigger the failure alert |
| `JPEG_QUALITY` | `85` | MJPEG output quality |

## Endpoints

| Endpoint | Description |
|---|---|
| `GET /stream` | MJPEG video stream |
| `GET /snapshot` | Single JPEG frame |
| `GET /health` | Liveness check |
| `POST /p/` | Upload an image for one-off detection |
| `GET /p/?img=<url>` | Detect from a URL |

## Building

```bash
docker build -t bambu-obico .
```

## License

MIT