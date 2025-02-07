#OBS
import obswebsocket
from obswebsocket import obsws, requests

# Connect to OBS WebSocket
ws = obsws("localhost", 4444, "password")  # Default port is 4444, set your password
ws.connect()

# Switch to a different scene
ws.call(requests.SetCurrentScene("Deepfake Scene"))

ws.disconnect()
