import json
import os
import urllib.parse
import urllib.request
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "0.0.0.0"
PORT = 8001
GEOSERVER_BASE = "http://localhost:8080/geoserver"
ROOT_DIR = Path(__file__).resolve().parent
LAYER_PREFIX = {
    "qc": "Habitats_INRS_Qc_",
    "laval": "Habitats_INRS_Laval_",
}


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/classes":
            self.serve_classes(parsed)
            return
        if parsed.path.startswith("/geoserver/"):
            self.proxy_geoserver(parsed)
            return
        if parsed.path in ("/", ""):
            self.path = "/index.html"
        return super().do_GET()

    def serve_classes(self, parsed):
        params = urllib.parse.parse_qs(parsed.query)
        campus = params.get("campus", ["qc"])[0]
        year = params.get("year", ["2025"])[0]
        prefix = LAYER_PREFIX.get(campus, LAYER_PREFIX["qc"])
        target = (
            f"{GEOSERVER_BASE}/ne/ows?service=WFS&version=1.0.0&request=GetFeature"
            f"&typeName=ne:{prefix}{urllib.parse.quote(str(year))}"
            f"&outputFormat=application/json&propertyName=class,nom_classe,surface_m2"
        )
        try:
            with urllib.request.urlopen(target, timeout=20) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            self.send_json({"classes": [], "error": str(exc)}, status=502)
            return

        seen = {}
        for feature in payload.get("features", []):
            props = feature.get("properties", {})
            class_id = props.get("class")
            if class_id is None:
                continue
            item = seen.setdefault(class_id, {
                "id": class_id,
                "name": props.get("nom_classe") or f"Classe {class_id}",
                "surface_m2": 0,
            })
            try:
                item["surface_m2"] += float(props.get("surface_m2") or 0)
            except (TypeError, ValueError):
                pass

        classes = [seen[key] for key in sorted(seen, key=lambda value: float(value))]
        self.send_json({"classes": classes})

    def proxy_geoserver(self, parsed):
        target = GEOSERVER_BASE + parsed.path[len("/geoserver"):]
        if parsed.query:
            target += "?" + parsed.query
        self.proxy_binary(target)

    def proxy_binary(self, target):
        try:
            with urllib.request.urlopen(target, timeout=30) as response:
                data = response.read()
                content_type = response.headers.get_content_type()
        except Exception as exc:
            self.send_error(502, f"Proxy error: {exc}")
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def send_json(self, payload, status=200):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    os.chdir(ROOT_DIR)
    httpd = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"Serveur disponible sur http://{HOST}:{PORT}")
    httpd.serve_forever()
