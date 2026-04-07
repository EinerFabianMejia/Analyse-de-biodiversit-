document.addEventListener("DOMContentLoaded", function () {
  const directGeoserverBase = "http://localhost:8080/geoserver";
  const serverBase = window.location.origin;
  const apiBase = serverBase.startsWith("http") ? serverBase : "";
  const geoserverBase = apiBase ? apiBase + "/geoserver" : directGeoserverBase;
  const defaultClassIds = [0, 1, 2, 3, 4, 5];
  const classColorById = {
    0: "#198f35",
    1: "#e0c8f1",
    2: "#929ca0",
    3: "#4cf643",
    8: "#0d0ced",
    10: "#a2f2b5"
  };
  const fallbackPalette = ["#198f35", "#e0c8f1", "#929ca0", "#4cf643", "#0d0ced", "#a2f2b5"];

  const map = L.map("map").setView([46.781, -71.277], 14);

  L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png").addTo(map);

  L.tileLayer.wms(geoserverBase + "/ne/wms", {
    layers: "ne:SecteurUlaval",
    format: "image/png",
    transparent: true
  }).addTo(map);

  L.Control.geocoder().addTo(map);

  const markers = L.layerGroup().addTo(map);
  const slider = document.getElementById("yearSlider");
  const classListEl = document.getElementById("classList");
  const classLegendContent = document.getElementById("classLegendContent");

  let currentYear = 2025;
  let classifLayer = null;
  let interval = null;
  let allObservations = [];
  let loadedIds = new Set();
  let classMetadata = [];

  window.toggleGroup = function (id) {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.display = el.style.display === "block" ? "none" : "block";
  };

  function escapeHtml(value) {
    return String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function normalizeClassEntry(entry, index) {
    const rawId = entry.id ?? entry.class ?? index;
    const id = Number(rawId);
    const safeId = Number.isNaN(id) ? index : id;
    const surfaceM2 = Number(entry.surface_m2 ?? 0);
    return {
      id: safeId,
      name: entry.name || entry.nom_classe || `Classe ${safeId}`,
      color: entry.color || classColorById[safeId] || fallbackPalette[index % fallbackPalette.length],
      surface_m2: Number.isNaN(surfaceM2) ? 0 : surfaceM2
    };
  }

  function formatArea(surfaceM2) {
    if (!surfaceM2) return "0 ha";
    const hectares = surfaceM2 / 10000;
    return hectares.toLocaleString("fr-CA", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }) + " ha";
  }

  function getClassName(classId) {
    const item = classMetadata.find((entry) => Number(entry.id) === Number(classId));
    return item ? item.name : `Classe ${classId}`;
  }

  function createLayer(year) {
    return L.tileLayer.wms(geoserverBase + "/ne/wms", {
      layers: "ne:Habitats_ULaval_" + year,
      format: "image/png",
      transparent: true
    });
  }

  function renderClassFilters() {
    if (!classMetadata.length) {
      classListEl.innerHTML = '<div class="loading-text">Aucune classe disponible.</div>';
      return;
    }

    const checkedValues = new Set(
      Array.from(document.querySelectorAll(".classCheck:checked")).map((cb) => cb.value)
    );

    classListEl.innerHTML = classMetadata
      .map((item) => {
        const shouldCheck = checkedValues.size === 0 || checkedValues.has(String(item.id));
        return `
          <label class="class-option">
            <input type="checkbox" class="classCheck" value="${escapeHtml(item.id)}" ${shouldCheck ? "checked" : ""}>
            <span class="class-option-swatch" style="background:${escapeHtml(item.color)}"></span>
            <span class="class-option-body">
              <span class="class-option-area">${escapeHtml(formatArea(item.surface_m2))}</span>
              <span class="class-option-name">${escapeHtml(item.name)}</span>
            </span>
          </label>
        `;
      })
      .join("");

    document.querySelectorAll(".classCheck").forEach((cb) => {
      cb.addEventListener("change", updateClassification);
    });
  }

  function renderClassLegend(year) {
    if (!classMetadata.length) {
      classLegendContent.innerHTML = '<div class="legend-hint">Aucune l?gende disponible.</div>';
      return;
    }

    const itemsHtml = classMetadata
      .map(
        (item) => `
          <div class="legend-item">
            <span class="legend-swatch" style="background:${escapeHtml(item.color)}"></span>
            <span>${escapeHtml(item.name)}</span>
          </div>
        `
      )
      .join("");

    classLegendContent.innerHTML = `<div class="legend-list">${itemsHtml}</div>`;
  }

  async function fetchClassesFromServer(year) {
    if (!apiBase) {
      throw new Error("server.py non disponible");
    }

    const res = await fetch(apiBase + "/api/classes?year=" + encodeURIComponent(year));
    if (!res.ok) {
      throw new Error("R?ponse serveur invalide");
    }

    return res.json();
  }

  async function loadClassMetadata(year) {
    classListEl.innerHTML = '<div class="loading-text">Chargement des classes...</div>';
    classLegendContent.innerHTML = '<div class="loading-text">Chargement de la l?gende...</div>';

    try {
      const payload = await fetchClassesFromServer(year);
      const source = Array.isArray(payload.classes) ? payload.classes : [];
      classMetadata = source
        .map(normalizeClassEntry)
        .sort((a, b) => Number(a.id) - Number(b.id));

      if (!classMetadata.length) {
        throw new Error("Aucune classe r?cup?r?e");
      }
    } catch (error) {
      console.error("Erreur chargement des classes:", error);
      classMetadata = defaultClassIds.map((id, index) =>
        normalizeClassEntry({ id, name: `Classe ${id}` }, index)
      );
      classLegendContent.innerHTML =
        '<div class="legend-hint">Impossible de charger les noms automatiquement.</div>';
    }

    renderClassFilters();
    renderClassLegend(year);
  }

  map.on("click", function (e) {
    if (!classifLayer) return;

    const point = map.latLngToContainerPoint(e.latlng, map.getZoom());
    const size = map.getSize();

    const url =
      geoserverBase +
      "/ne/wms?" +
      "SERVICE=WMS&VERSION=1.1.1&REQUEST=GetFeatureInfo" +
      "&LAYERS=ne:Habitats_ULaval_" + currentYear +
      "&QUERY_LAYERS=ne:Habitats_ULaval_" + currentYear +
      "&STYLES=" +
      "&BBOX=" + map.getBounds().toBBoxString() +
      "&FEATURE_COUNT=1" +
      "&HEIGHT=" + size.y +
      "&WIDTH=" + size.x +
      "&INFO_FORMAT=application/json" +
      "&SRS=EPSG:4326" +
      "&X=" + Math.floor(point.x) +
      "&Y=" + Math.floor(point.y);

    fetch(url)
      .then((res) => res.json())
      .then((data) => {
        if (!data.features || data.features.length === 0) {
          alert("Aucune donn?e ici");
          return;
        }

        const props = data.features[0].properties || {};
        const nom = props.nom_classe || getClassName(props.class) || "Inconnu";
        const surface = props.surface_m2 ? (props.surface_m2 / 10000).toFixed(2) + " ha" : "Non disponible";

        L.popup()
          .setLatLng(e.latlng)
          .setContent(`
            <b>Classe :</b> ${escapeHtml(nom)}<br>
            <b>Superficie :</b> ${escapeHtml(surface)}
          `)
          .openOn(map);
      })
      .catch((err) => {
        console.error("Erreur GetFeatureInfo:", err);
      });
  });

  function updateClassification() {
    const selected = [];

    document.querySelectorAll(".classCheck").forEach((cb) => {
      if (cb.checked) selected.push(cb.value);
    });

    if (selected.length === 0) {
      if (classifLayer && map.hasLayer(classifLayer)) {
        map.removeLayer(classifLayer);
      }
      return;
    }

    if (!classifLayer) return;

    if (!map.hasLayer(classifLayer)) {
      classifLayer.addTo(map);
    }

    classifLayer.setParams({
      CQL_FILTER: "class IN (" + selected.join(",") + ")"
    });
  }

  async function changeYear(year) {
    currentYear = Number(year);

    if (classifLayer && map.hasLayer(classifLayer)) {
      map.removeLayer(classifLayer);
    }

    classifLayer = createLayer(currentYear);
    classifLayer.addTo(map);

    document.getElementById("yearLabel").innerText = currentYear;

    await loadClassMetadata(currentYear);
    updateClassification();
  }

  slider.addEventListener("input", function () {
    changeYear(this.value);
  });

  async function getAvailableYears() {
    const url = geoserverBase + "/ne/wms?service=WMS&request=GetCapabilities";
    const res = await fetch(url);
    const text = await res.text();

    const parser = new DOMParser();
    const xml = parser.parseFromString(text, "text/xml");
    const layers = xml.getElementsByTagName("Layer");
    const years = [];

    for (let i = 0; i < layers.length; i++) {
      const name = layers[i].getElementsByTagName("Name")[0];
      if (!name) continue;

      const layerName = name.textContent;
      if (!layerName.includes("Habitats_ULaval_")) continue;

      const parsedYear = Number(layerName.split("_").pop());
      if (!Number.isNaN(parsedYear)) {
        years.push(parsedYear);
      }
    }

    return years.sort((a, b) => a - b);
  }

  async function setupYearSlider() {
    try {
      const years = await getAvailableYears();
      if (years.length) {
        slider.min = Math.min(...years);
        slider.max = Math.max(...years);
        slider.value = slider.max;
      }
    } catch (error) {
      console.error("Impossible de r?cup?rer les ann?es:", error);
    }

    await changeYear(slider.value);
  }

  window.playAnimation = function () {
    if (interval) return;

    interval = setInterval(() => {
      let year = parseInt(slider.value, 10);
      year++;

      if (year > Number(slider.max)) year = Number(slider.min);

      slider.value = year;
      changeYear(year);
    }, 1500);
  };

  window.stopAnimation = function () {
    clearInterval(interval);
    interval = null;
  };

  const taxonGroups = {
    Mammiferes: 40151,
    Oiseaux: 3,
    Reptiles: 26036,
    Amphibiens: 20978,
    Poissons: 47178,
    Insectes: 47157,
    Arachnides: 47119,
    Mollusques: 47115,
    Plantes: 47126,
    Champignons: 47170,
    Protozoaires: 47686,
    Chromistes: 48222,
    AutresAnimaux: 1
  };

  const groupColors = {
    Mammiferes: "#dee51e",
    Oiseaux: "#72ff13",
    Reptiles: "#ff179e",
    Amphibiens: "#d43aff",
    Poissons: "#60aaff",
    Insectes: "#ff9036",
    Arachnides: "#00fbf3",
    Mollusques: "#6D4C41",
    Plantes: "#4CAF50",
    Champignons: "#ff2929",
    Protozoaires: "#AB47BC",
    Chromistes: "#26A69A",
    AutresAnimaux: "#E53935"
  };

  async function loadAllProjectObservations() {
    const panel = document.getElementById("infoPanel");
    panel.innerHTML = "T?l?chargement des observations...";

    markers.clearLayers();
    allObservations = [];
    loadedIds.clear();

    let page = 1;
    let hasMore = true;

    while (hasMore) {
      const url =
        "https://api.inaturalist.org/v1/observations?" +
        "project_id=universite-laval-reseau-biodiversite-au-campus" +
        "&quality_grade=research,needs_id" +
        "&per_page=200&page=" + page;

      const res = await fetch(url);
      const data = await res.json();

      if (!data.results) break;

      if (data.results.length < 200) {
        hasMore = false;
      } else {
        page++;
      }

      data.results.forEach((obs) => {
        if (!obs.geojson || !obs.geojson.coordinates) return;
        if (loadedIds.has(obs.id)) return;

        loadedIds.add(obs.id);
        allObservations.push(obs);
      });
    }

    renderObservations();

    let researchCount = 0;
    let needsIdCount = 0;

    allObservations.forEach((obs) => {
      if (obs.quality_grade === "research") researchCount++;
      if (obs.quality_grade === "needs_id") needsIdCount++;
    });

    const total = researchCount + needsIdCount;
    const researchPercent = total ? ((researchCount / total) * 100).toFixed(1) : "0.0";
    const needsIdPercent = total ? ((needsIdCount / total) * 100).toFixed(1) : "0.0";

    panel.innerHTML = `
      ${allObservations.length} observations<br><br>
      <b>Statistiques :</b><br>
      Research : ${researchPercent}%<br>
      Needs ID : ${needsIdPercent}%
    `;
  }

  window.loadAllProjectObservations = loadAllProjectObservations;

  function renderObservations() {
    markers.clearLayers();

    const selected = Array.from(document.querySelectorAll(".obsType:checked")).map((cb) => cb.value);
    if (selected.length === 0) return;

    allObservations.forEach((obs) => {
      if (!obs.geojson || !obs.geojson.coordinates) return;

      const ancestorIds = obs.taxon?.ancestor_ids || [];
      let groupFound = null;

      for (const group in taxonGroups) {
        if (ancestorIds.includes(taxonGroups[group])) {
          groupFound = group;
          break;
        }
      }

      if (!groupFound || !selected.includes(groupFound)) return;

      const [lng, lat] = obs.geojson.coordinates;
      const color = groupColors[groupFound];

      const marker = L.circleMarker([lat, lng], {
        radius: 3,
        color: color,
        fillOpacity: 1
      });

      marker.bindPopup(`
        <strong>${escapeHtml(obs.taxon?.preferred_common_name || "Nom inconnu")}</strong><br>
        Date: ${escapeHtml(obs.observed_on || "Inconnue")}<br>
        <a href="${escapeHtml(obs.uri || "#")}" target="_blank" rel="noreferrer">Voir</a>
      `);

      markers.addLayer(marker);
    });
  }

  document.querySelectorAll(".obsType").forEach((cb) => cb.addEventListener("change", renderObservations));

  loadAllProjectObservations();

  window.goHome = function () {
    map.setView([46.781, -71.277], 14);
    markers.clearLayers();

    if (typeof airLayer !== "undefined" && airLayer && typeof airLayer.clearLayers === "function") {
      airLayer.clearLayers();
    }

    if (classifLayer && map.hasLayer(classifLayer)) {
      map.removeLayer(classifLayer);
    }
  };

  setupYearSlider();

  if (typeof loadAirData === "function") {
    loadAirData();
  }
});
