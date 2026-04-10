import json
import os
import re
import urllib.parse
import urllib.request
from html import unescape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HOST = "127.0.0.1"
PORT = 8000
GEOSERVER_BASE = "http://localhost:8080/geoserver"
ROOT_DIR = Path(__file__).resolve().parent
AIR_QUALITY_URL = "https://www.iqa.environnement.gouv.qc.ca/contenu/indice.asp?site={site_id}"
AIR_QUALITY_STATIONS = [
    {"site_id": 3006, "name": "Québec - Vieux-Limoilou", "latitude": 46.8211, "longitude": -71.2208},
    {"site_id": 3021, "name": "Québec - École les Primevères", "latitude": 46.77416667, "longitude": -71.36972222},
    {"site_id": 3028, "name": "Québec - Collège St-Charles-Garnier", "latitude": 46.79472222, "longitude": -71.24638889},
    {"site_id": 3052, "name": "Québec - Henri-IV", "latitude": 46.781331, "longitude": -71.308711},
]


class AppHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT_DIR), **kwargs)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)

        if parsed.path == "/api/classes":
            self.serve_classes(parsed)
            return

        if parsed.path == "/api/class-legend-image":
            self.serve_legend_image(parsed)
            return

        if parsed.path == "/api/air-quality":
            self.serve_air_quality()
            return

        if parsed.path.startswith("/geoserver/"):
            self.proxy_geoserver(parsed)
            return

        if parsed.path in ("/", ""):
            self.path = "/index.html"

        return super().do_GET()

    def serve_classes(self, parsed):
        params = urllib.parse.parse_qs(parsed.query)
        year = params.get("year", ["2025"])[0]
        target = (
            f"{GEOSERVER_BASE}/ne/ows?service=WFS&version=1.0.0&request=GetFeature"
            f"&typeName=ne:Habitats_ULaval_{urllib.parse.quote(str(year))}"
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
                "surface_m2": 0
            })
            try:
                item["surface_m2"] += float(props.get("surface_m2") or 0)
            except (TypeError, ValueError):
                pass

        classes = [seen[key] for key in sorted(seen, key=lambda value: float(value))]
        self.send_json({"classes": classes})

    def serve_air_quality(self):
        stations = []
        for station in AIR_QUALITY_STATIONS:
            try:
                payload = self.fetch_air_quality_station(station)
                stations.append(payload)
            except Exception as exc:
                stations.append({
                    "site_id": station["site_id"],
                    "station_name": station.get("name") or f"Station {station['site_id']}",
                    "latitude": station["latitude"],
                    "longitude": station["longitude"],
                    "live_value_available": False,
                    "error": str(exc),
                    "source_url": AIR_QUALITY_URL.format(site_id=urllib.parse.quote(str(station["site_id"]))),
                })

        self.send_json({"stations": stations})

    def fetch_air_quality_station(self, station):
        target = AIR_QUALITY_URL.format(site_id=urllib.parse.quote(str(station["site_id"])))
        with urllib.request.urlopen(target, timeout=20) as response:
            raw = response.read()

        html = raw.decode("cp1252", errors="replace")
        station_name = self.extract_air_match(html, r'<font size="2">([^<]+)</font>')
        pollutant_html = self.extract_air_match(
            html,
            r'<b><a style="text-decoration: none".*?>(.*?)</a></b><br\s*/?><b>',
            flags=re.S
        )
        category = self.extract_air_match(html, r'</a></b><br\s*/?><b>([^<]+)&nbsp;</b>')
        measured_at = self.extract_air_match(
            html,
            r'<td width="100%" valign="top" align="center">\s*([^<]+?)\s*</td>'
        )
        value = self.extract_air_match(html, r'src="\.\./graphique/A(\d+)\.gif"')

        pollutant = re.sub(r"<[^>]+>", "", pollutant_html or "")
        return {
            "site_id": station["site_id"],
            "station_name": self.clean_text(station_name) or station.get("name") or f"Station {station['site_id']}",
            "pollutant": self.clean_text(pollutant),
            "category": self.clean_text(category),
            "measured_at": self.clean_text(measured_at),
            "value": int(value) if value else None,
            "latitude": station["latitude"],
            "longitude": station["longitude"],
            "source_url": target,
            "live_value_available": bool(value),
        }

    def extract_air_match(self, html, pattern, flags=0):
        match = re.search(pattern, html, flags)
        return match.group(1) if match else ""

    def clean_text(self, value):
        return unescape(str(value or "")).replace("\xa0", " ").strip()

    def serve_legend_image(self, parsed):
        params = urllib.parse.parse_qs(parsed.query)
        year = params.get("year", ["2025"])[0]
        target = (
            f"{GEOSERVER_BASE}/ne/wms?service=WMS&version=1.0.0&request=GetLegendGraphic"
            f"&format=image/png&layer=ne:Habitats_ULaval_{urllib.parse.quote(str(year))}"
        )
        self.proxy_binary(target, "image/png")

    def proxy_geoserver(self, parsed):
        target = GEOSERVER_BASE + parsed.path[len("/geoserver"):]
        if parsed.query:
            target += "?" + parsed.query
        self.proxy_binary(target)

    def proxy_binary(self, target, content_type=None):
        try:
            with urllib.request.urlopen(target, timeout=30) as response:
                data = response.read()
                guessed_type = response.headers.get_content_type()
        except Exception as exc:
            self.send_error(502, f"Proxy error: {exc}")
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type or guessed_type or "application/octet-stream")
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
