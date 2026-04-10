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

  const map = L.map("map", { maxZoom: 22 }).setView([46.781, -71.277], 15);
  const baseLayers = {
    "Carte": L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
      attribution: "&copy; OpenStreetMap &copy; CARTO"
    }),
    "Street": L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors"
    }),
    "Satellite": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      attribution: "Tiles &copy; Esri"
    }),
    "Topo": L.tileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenTopoMap contributors"
    })
  };
  baseLayers["Carte"].addTo(map);
  const secteurLayer = L.tileLayer.wms(geoserverBase + "/ne/wms", {
    layers: "ne:SecteurUlaval",
    format: "image/png",
    transparent: true
  }).addTo(map);
  const baseLayerRefs = {
    carte: baseLayers["Carte"],
    street: baseLayers["Street"],
    satellite: baseLayers["Satellite"],
    topo: baseLayers["Topo"]
  };
  const overlayLayerRefs = {
    footprint: secteurLayer,
    classification: null
  };
  let layerControl = null;
  L.control.scale({ imperial: false, position: "bottomleft" }).addTo(map);

  const markers = L.layerGroup().addTo(map);
  const slider = document.getElementById("yearSlider");
  const classListEl = document.getElementById("classList");
  const airQualityListEl = document.getElementById("airQualityList");
  const classLegendContent = document.getElementById("classLegendContent");
  const infoPanel = document.getElementById("infoPanel");
  const classLegendPanel = document.getElementById("classLegendPanel");
  const addressSearch = document.getElementById("addressSearch");
  const languageSelect = document.getElementById("languageSelect");
  const searchSuggestions = document.getElementById("searchSuggestions");
  const chartModal = document.getElementById("chartModal");
  const chartModalSubtitle = document.getElementById("chartModalSubtitle");
  const classAreaCanvas = document.getElementById("classAreaChart");

  let currentYear = 2025;
  let classifLayer = null;
  let interval = null;
  let observationsRefreshInterval = null;
  let allObservations = [];
  let loadedIds = new Set();
  let classMetadata = [];
  let airQualityData = [];
  let airQualityLoaded = false;
  let airQualityVisible = false;
  let airQualityLayer = null;
  let airStationMarkers = L.layerGroup();
  let searchMarker = null;
  let searchAbortController = null;
  let searchDebounce = null;
  let classAreaChart = null;
  let currentLanguage = localStorage.getItem("platformLanguage") || "fr";

  const translations = {
    fr: {
      app_title: "Biodiversite",
      language: "Langue",
      home: "Accueil",
      classification: "Classification",
      loading_classes: "Chargement des classes...",
      temporal_tracking: "Suivi temporel",
      active_year: "Annee active",
      surface_chart: "Graphique superficies",
      export_png: "Exporter carte PNG",
      monthly_report: "Generer rapport mensuel",
      observations: "Observations",
      load_all_species: "Charger toutes especes",
      remove_observations: "Retirer observations",
      air_quality: "Qualite de l'air",
      loading_air_quality: "Chargement de la qualite de l'air...",
      air_quality_source: "Source gouvernementale",
      air_quality_refresh: "Mise a jour",
      air_quality_campus_estimate: "Estimation campus",
      air_quality_stations: "Stations officielles",
      air_quality_interpolation_note: "Interpolation IDW a partir des stations officielles du gouvernement du Quebec autour du campus.",
      air_quality_unavailable: "Impossible de charger la qualite de l'air.",
      air_quality_live_note: "Seules les stations avec une valeur IQA en direct sont utilisees dans l'interpolation.",
      air_quality_overlay_on: "Couche air visible",
      air_quality_overlay_off: "Couche air masquee",
      station: "Station",
      pollutant: "Polluant",
      live_value_missing: "Valeur IQA en direct non disponible",
      air_index: "Indice IQA",
      components: "Composantes",
      search_address: "Chercher adresse",
      select_class_hint: "Selectionnez une classe pour voir la superficie",
      class_legend: "Legende des classes",
      loading_legend: "Chargement de la legende...",
      surface_evolution: "Evolution des superficies",
      loading: "Chargement...",
      close: "Fermer",
      no_classes: "Aucune classe disponible.",
      no_legend: "Aucune legende disponible.",
      loading_observations: "Telechargement des observations...",
      observations_removed: "Observations retirees de la carte.",
      year: "Annee",
      statistics: "Statistiques",
      total_observations: "observations",
      realtime_observations: "Observations iNaturalist temps reel",
      classification_year: "Annee classification",
      search_no_result: "Aucune suggestion trouvee.",
      result: "Resultat",
      address: "Adresse",
      loading_chart: "Chargement du graphique...",
      chart_unavailable: "Impossible de charger le graphique.",
      chart_surface_subtitle: "Evolution des superficies par classe",
      area_ha: "Superficie (ha)",
      class_label: "Classe",
      area_label: "Superficie",
      unknown: "Inconnue",
      type: "Type",
      quality: "Qualite",
      date: "Date",
      french_name: "Nom francais",
      view_source: "Voir l'observation source",
      report_title: "Rapport mensuel biodiversite",
      month: "Mois",
      observation_types: "Types d'observations affiches",
      observation_srs: "Systeme de reference des observations",
      classification_srs: "Systeme de reference de la classification carte",
      observation_stats: "Statistiques observations",
      total: "Total",
      summary: "Resume",
      report_summary_text: "Ce rapport resume l'etat courant des observations iNaturalist filtrees en Research et Needs ID, ainsi que la superficie des classes de la couche ULaval pour {year}.",
      scale: "Echelle",
      scale_unavailable: "Echelle non disponible",
      exported_on: "Exporte le",
      biodiversity_map_title: "Carte de biodiversite ULaval",
      classification_observations: "Classification {year} | Observations iNaturalist",
      platform_footer: "Plateforme biodiversite campus ULaval",
      source_footer: "Source : GeoServer + iNaturalist",
      base_map: "Carte",
      street_map: "Street",
      satellite_map: "Satellite",
      topo_map: "Topo",
      footprint: "Emprise",
      map_classification: "Classification",
      none: "Aucun",
      not_available: "Non disponible",
      export_error: "Impossible d'exporter l'image de la carte.",
      research_label: "Recherche",
      needs_id_label: "A identifier",
      observation_default_name: "Observation",
      air_category_good: "Bon",
      air_category_acceptable: "Acceptable",
      air_category_poor: "Mauvais"
    },
    en: {
      app_title: "Biodiversity",
      language: "Language",
      home: "Home",
      classification: "Classification",
      loading_classes: "Loading classes...",
      temporal_tracking: "Temporal tracking",
      active_year: "Active year",
      surface_chart: "Area chart",
      export_png: "Export PNG map",
      monthly_report: "Generate monthly report",
      observations: "Observations",
      load_all_species: "Load all species",
      remove_observations: "Remove observations",
      air_quality: "Air quality",
      loading_air_quality: "Loading air quality...",
      air_quality_source: "Government source",
      air_quality_refresh: "Updated",
      air_quality_campus_estimate: "Campus estimate",
      air_quality_stations: "Official stations",
      air_quality_interpolation_note: "IDW interpolation from official Government of Quebec stations surrounding the campus.",
      air_quality_unavailable: "Unable to load air quality.",
      air_quality_live_note: "Only stations with a live IQA value are used in the interpolation.",
      air_quality_overlay_on: "Air layer visible",
      air_quality_overlay_off: "Air layer hidden",
      station: "Station",
      pollutant: "Pollutant",
      live_value_missing: "Live IQA value unavailable",
      air_index: "AQI index",
      components: "Components",
      search_address: "Search address",
      select_class_hint: "Select a class to view its area",
      class_legend: "Class legend",
      loading_legend: "Loading legend...",
      surface_evolution: "Area evolution",
      loading: "Loading...",
      close: "Close",
      no_classes: "No class available.",
      no_legend: "No legend available.",
      loading_observations: "Downloading observations...",
      observations_removed: "Observations removed from the map.",
      year: "Year",
      statistics: "Statistics",
      total_observations: "observations",
      realtime_observations: "Real-time iNaturalist observations",
      classification_year: "Classification year",
      search_no_result: "No suggestion found.",
      result: "Result",
      address: "Address",
      loading_chart: "Loading chart...",
      chart_unavailable: "Unable to load chart.",
      chart_surface_subtitle: "Area evolution by class",
      area_ha: "Area (ha)",
      class_label: "Class",
      area_label: "Area",
      unknown: "Unknown",
      type: "Type",
      quality: "Quality",
      date: "Date",
      french_name: "French name",
      view_source: "View source observation",
      report_title: "Monthly biodiversity report",
      month: "Month",
      observation_types: "Displayed observation types",
      observation_srs: "Observation reference system",
      classification_srs: "Map classification reference system",
      observation_stats: "Observation statistics",
      total: "Total",
      summary: "Summary",
      report_summary_text: "This report summarizes the current state of iNaturalist observations filtered as Research and Needs ID, as well as the area of ULaval layer classes for {year}.",
      scale: "Scale",
      scale_unavailable: "Scale not available",
      exported_on: "Exported on",
      biodiversity_map_title: "ULaval biodiversity map",
      classification_observations: "Classification {year} | iNaturalist observations",
      platform_footer: "ULaval campus biodiversity platform",
      source_footer: "Source: GeoServer + iNaturalist",
      base_map: "Map",
      street_map: "Street",
      satellite_map: "Satellite",
      topo_map: "Topo",
      footprint: "Campus boundary",
      map_classification: "Classification",
      none: "None",
      not_available: "Not available",
      export_error: "Unable to export the map image.",
      research_label: "Research",
      needs_id_label: "Needs ID",
      observation_default_name: "Observation",
      air_category_good: "Good",
      air_category_acceptable: "Acceptable",
      air_category_poor: "Poor"
    }
  };

  const classNameTranslations = {
    Gazon: { fr: "Gazon", en: "Lawn" },
    SurfacesImpermeables: { fr: "SurfacesImpermeables", en: "ImperviousSurfaces" },
    Infrastructures: { fr: "Infrastructures", en: "Infrastructure" },
    BoiseUrbain: { fr: "BoiseUrbain", en: "UrbanWoodland" },
    SurfaceMinerales: { fr: "SurfaceMinerales", en: "MineralSurfaces" },
    JardinAgricultureUrbaine: { fr: "JardinAgricultureUrbaine", en: "UrbanAgricultureGarden" }
  };

  const groupLabels = {
    Mammiferes: { fr: "Mammiferes", en: "Mammals" },
    Oiseaux: { fr: "Oiseaux", en: "Birds" },
    Reptiles: { fr: "Reptiles", en: "Reptiles" },
    Amphibiens: { fr: "Amphibiens", en: "Amphibians" },
    Poissons: { fr: "Poissons", en: "Fish" },
    Insectes: { fr: "Insectes", en: "Insects" },
    Arachnides: { fr: "Arachnides", en: "Arachnids" },
    Mollusques: { fr: "Mollusques", en: "Molluscs" },
    Plantes: { fr: "Plantes", en: "Plants" },
    Champignons: { fr: "Champignons", en: "Fungi" },
    Protozoaires: { fr: "Protozoaires", en: "Protozoa" },
    Chromistes: { fr: "Chromistes", en: "Chromists" },
    AutresAnimaux: { fr: "Autres animaux", en: "Other animals" }
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
    Mammiferes: "#ef4444",
    Oiseaux: "#f59e0b",
    Reptiles: "#22c55e",
    Amphibiens: "#14b8a6",
    Poissons: "#3b82f6",
    Insectes: "#8b5cf6",
    Arachnides: "#ec4899",
    Mollusques: "#a16207",
    Plantes: "#16a34a",
    Champignons: "#78716c",
    Protozoaires: "#06b6d4",
    Chromistes: "#0f766e",
    AutresAnimaux: "#dc2626"
  };

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

  function t(key, vars = {}) {
    let text = translations[currentLanguage]?.[key] || translations.fr[key] || key;
    Object.entries(vars).forEach(([name, value]) => {
      text = text.replaceAll(`{${name}}`, value);
    });
    return text;
  }

  function translateClassName(name) {
    return classNameTranslations[name]?.[currentLanguage] || name;
  }

  function translateGroupName(name) {
    return groupLabels[name]?.[currentLanguage] || name;
  }

  function updateStaticTranslations() {
    document.documentElement.lang = currentLanguage;
    document.title = t("biodiversity_map_title");
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      node.textContent = t(node.dataset.i18n);
    });
    if (addressSearch) {
      addressSearch.placeholder = t("search_address");
      addressSearch.setAttribute("aria-label", t("search_address"));
    }
    document.querySelectorAll("[data-group-label]").forEach((node) => {
      node.textContent = translateGroupName(node.dataset.groupLabel);
    });
    if (languageSelect) languageSelect.value = currentLanguage;
  }

  function formatArea(surfaceM2) {
    if (!surfaceM2) return "0 ha";
    return (Number(surfaceM2) / 10000).toLocaleString(currentLanguage === "en" ? "en-CA" : "fr-CA", {
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }) + " ha";
  }

  function getLocalizedBaseLayers() {
    return {
      [t("base_map")]: baseLayerRefs.carte,
      [t("street_map")]: baseLayerRefs.street,
      [t("satellite_map")]: baseLayerRefs.satellite,
      [t("topo_map")]: baseLayerRefs.topo
    };
  }

  function getLocalizedOverlayLayers() {
    const localized = {
      [t("footprint")]: overlayLayerRefs.footprint
    };
    if (overlayLayerRefs.classification) {
      localized[t("map_classification")] = overlayLayerRefs.classification;
    }
    return localized;
  }

  function refreshLayerControl() {
    if (layerControl) map.removeControl(layerControl);
    layerControl = L.control.layers(getLocalizedBaseLayers(), getLocalizedOverlayLayers(), {
      collapsed: true,
      position: "topleft"
    }).addTo(map);
  }

  function getAirQualityColor(value) {
    if (value == null || Number.isNaN(Number(value))) return "#94a3b8";
    if (Number(value) <= 25) return "#22c55e";
    if (Number(value) <= 50) return "#f59e0b";
    return "#ef4444";
  }

  function getAirCategoryLabel(value) {
    if (value == null || Number.isNaN(Number(value))) return t("unknown");
    if (Number(value) <= 25) return t("air_category_good");
    if (Number(value) <= 50) return t("air_category_acceptable");
    return t("air_category_poor");
  }

  function computeCampusAirEstimate(stations) {
    const targetLat = 46.781;
    const targetLng = -71.277;
    let numerator = 0;
    let denominator = 0;

    stations.forEach((station) => {
      const value = Number(station.value);
      const lat = Number(station.latitude);
      const lng = Number(station.longitude);
      if (Number.isNaN(value) || Number.isNaN(lat) || Number.isNaN(lng)) return;
      const distance = Math.hypot(lat - targetLat, lng - targetLng);
      if (distance === 0) {
        numerator = value;
        denominator = 1;
        return;
      }
      const weight = 1 / Math.pow(distance, 2);
      numerator += value * weight;
      denominator += weight;
    });

    if (!denominator) return null;
    return Number((numerator / denominator).toFixed(1));
  }

  function formatAirDate(value) {
    if (!value) return t("unknown");
    const date = new Date(Number(value));
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toLocaleString(currentLanguage === "en" ? "en-CA" : "fr-CA", {
      dateStyle: "medium",
      timeStyle: "short"
    });
  }

  function formatAirComponents(station) {
    if (station.pollutant) return station.pollutant;
    const components = [
      station.pm25 != null ? `PM2.5: ${station.pm25}` : null,
      station.o3 != null ? `O3: ${station.o3}` : null,
      station.no2 != null ? `NO2: ${station.no2}` : null,
      station.so2 != null ? `SO2: ${station.so2}` : null,
      station.co != null ? `CO: ${station.co}` : null
    ].filter(Boolean);

    return components.length ? components.join(" | ") : t("unknown");
  }

  function renderAirQualityPanel() {
    if (!airQualityListEl) return;
    if (!airQualityData.length) {
      airQualityListEl.innerHTML = `<div class="loading-text">${escapeHtml(t("air_quality_unavailable"))}</div>`;
      return;
    }

    const validStations = airQualityData.filter((station) => station.value != null && !station.error);
    const campusEstimate = computeCampusAirEstimate(validStations);
    const measuredAt = formatAirDate(validStations.find((station) => station.measured_at)?.measured_at);

    airQualityListEl.innerHTML = `
      <div class="air-quality-card">
        <div class="air-quality-head">
          <span class="air-quality-source">${escapeHtml(t("air_quality_source"))}</span>
          <span class="air-quality-time">${escapeHtml(t("air_quality_refresh"))} : ${escapeHtml(measuredAt)}</span>
        </div>
        <div class="air-campus-badge" style="background:${escapeHtml(getAirQualityColor(campusEstimate))}">
          ${escapeHtml(t("air_quality_campus_estimate"))} : ${escapeHtml(campusEstimate != null ? String(campusEstimate) : t("unknown"))}
        </div>
        <div class="air-quality-note">${escapeHtml(t("air_quality_interpolation_note"))}</div>
        <div class="air-quality-note">${escapeHtml(t("air_quality_live_note"))}</div>
        <div class="air-station-title">${escapeHtml(t("air_quality_stations"))}</div>
        <div class="air-station-list">
          ${airQualityData.map((station) => `
            <div class="air-station-item">
              <span class="air-station-dot" style="background:${escapeHtml(getAirQualityColor(station.value))}"></span>
              <div class="air-station-body">
                <div class="air-station-name">${escapeHtml(station.station_name || `${t("station")} ${station.site_id}`)}</div>
                <div class="air-station-meta">
                  ${station.value != null
                    ? `${escapeHtml(t("air_index"))} : ${escapeHtml(String(station.value))} | ${escapeHtml(station.category || getAirCategoryLabel(station.value))}<br>${escapeHtml(t("pollutant"))} : ${escapeHtml(formatAirComponents(station))}`
                    : `${escapeHtml(t("live_value_missing"))}`
                  }
                </div>
              </div>
            </div>
          `).join("")}
        </div>
      </div>
    `;
  }

  function clearAirQualityLayers() {
    if (airQualityLayer && map.hasLayer(airQualityLayer)) {
      map.removeLayer(airQualityLayer);
    }
    airQualityLayer = null;
    airStationMarkers.clearLayers();
    if (map.hasLayer(airStationMarkers)) {
      map.removeLayer(airStationMarkers);
    }
  }

  function renderAirQualityLayers() {
    clearAirQualityLayers();
    const validStations = airQualityData.filter((station) => station.value != null && !station.error);
    if (!validStations.length) return;

    const idwPoints = [];
    validStations.forEach((station) => {
      const lat = Number(station.latitude);
      const lng = Number(station.longitude);
      const value = Number(station.value);
      if (Number.isNaN(lat) || Number.isNaN(lng) || Number.isNaN(value)) return;

      idwPoints.push([lat, lng, value]);

      const marker = L.circleMarker([lat, lng], {
        radius: 9,
        color: "#ffffff",
        weight: 2,
        fillColor: getAirQualityColor(value),
        fillOpacity: 0.95
      });
      marker.bindPopup(`
        <div class="popup-card">
          <h3>${escapeHtml(station.station_name || `${t("station")} ${station.site_id}`)}</h3>
          ${station.value != null ? `<p><strong>${escapeHtml(t("air_index"))} :</strong> ${escapeHtml(String(value))}</p>` : `<p><strong>${escapeHtml(t("air_index"))} :</strong> ${escapeHtml(t("live_value_missing"))}</p>`}
          <p><strong>${escapeHtml(t("quality"))} :</strong> ${escapeHtml(station.category || getAirCategoryLabel(value))}</p>
          <p><strong>${escapeHtml(t("pollutant"))} :</strong> ${escapeHtml(formatAirComponents(station))}</p>
          <p><strong>${escapeHtml(t("date"))} :</strong> ${escapeHtml(formatAirDate(station.measured_at))}</p>
          <a class="popup-link" href="${escapeHtml(station.source_url || "#")}" target="_blank" rel="noreferrer">${escapeHtml(t("view_source"))}</a>
        </div>
      `);
      airStationMarkers.addLayer(marker);
    });

    if (typeof L.idw === "function" && idwPoints.length >= 2) {
      airQualityLayer = L.idw(idwPoints, {
        opacity: 0.28,
        cellSize: 18,
        exp: 2,
        max: 100
      });
      airQualityLayer.addTo(map);
    }

    airStationMarkers.addTo(map);
  }

  async function loadAirQuality() {
    if (!airQualityListEl) return;
    airQualityListEl.innerHTML = `<div class="loading-text">${escapeHtml(t("loading_air_quality"))}</div>`;

    try {
      const response = await fetch(apiBase + "/api/air-quality");
      if (!response.ok) throw new Error("Reponse air invalide");
      const payload = await response.json();
      airQualityData = Array.isArray(payload.stations) ? payload.stations : [];
      airQualityLoaded = true;
      renderAirQualityPanel();
      if (airQualityVisible) renderAirQualityLayers();
    } catch (error) {
      console.error("Erreur qualite de l'air:", error);
      airQualityData = [];
      airQualityLoaded = false;
      airQualityListEl.innerHTML = `<div class="loading-text">${escapeHtml(t("air_quality_unavailable"))}</div>`;
      clearAirQualityLayers();
    }
  }

  function getObservationStats() {
    let researchCount = 0;
    let needsIdCount = 0;
    allObservations.forEach((obs) => {
      if (obs.quality_grade === "research") researchCount++;
      if (obs.quality_grade === "needs_id") needsIdCount++;
    });

    const total = researchCount + needsIdCount;
    return {
      total: allObservations.length,
      researchCount,
      needsIdCount,
      researchPercent: total ? ((researchCount / total) * 100).toFixed(1) : "0.0",
      needsIdPercent: total ? ((needsIdCount / total) * 100).toFixed(1) : "0.0"
    };
  }

  function renderObservationsInfoPanel() {
    const stats = getObservationStats();

    infoPanel.innerHTML = `
      <b>${escapeHtml(t("realtime_observations"))} :</b> ${escapeHtml(stats.total)}<br><br>
      <b>${escapeHtml(t("classification_year"))} :</b> ${escapeHtml(currentYear || "-")}<br><br>
      <b>${escapeHtml(t("statistics"))} :</b><br>
      ${escapeHtml(t("research_label"))} : ${escapeHtml(stats.researchPercent)}%<br>
      ${escapeHtml(t("needs_id_label"))} : ${escapeHtml(stats.needsIdPercent)}%
    `;
  }

  function downloadTextFile(filename, content, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  }

  function cloneForExport(element) {
    const clone = element.cloneNode(true);
    clone.removeAttribute("id");
    clone.style.position = "static";
    clone.style.left = "auto";
    clone.style.right = "auto";
    clone.style.top = "auto";
    clone.style.bottom = "auto";
    clone.style.width = "100%";
    clone.style.maxWidth = "none";
    clone.style.margin = "0";
    clone.style.boxSizing = "border-box";
    clone.style.background = "linear-gradient(180deg, rgba(255,255,255,0.98), rgba(247,250,252,0.96))";
    clone.style.border = "1px solid rgba(15,23,42,0.08)";
    clone.style.borderRadius = "18px";
    clone.style.boxShadow = "0 12px 28px rgba(15,23,42,0.1)";
    return clone;
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

  function destroyClassChart() {
    if (classAreaChart) {
      classAreaChart.destroy();
      classAreaChart = null;
    }
  }


  function createLayer(year) {
    return L.tileLayer.wms(geoserverBase + "/ne/wms", {
      layers: "ne:Habitats_ULaval_" + year,
      format: "image/png",
      transparent: true
    });
  }

  async function fetchClassesFromServer(year) {
    if (!apiBase) throw new Error("server.py non disponible");
    const res = await fetch(apiBase + "/api/classes?year=" + encodeURIComponent(year));
    if (!res.ok) throw new Error("Reponse serveur invalide");
    return res.json();
  }

  function renderClassFilters() {
    if (!classMetadata.length) {
      classListEl.innerHTML = `<div class="loading-text">${escapeHtml(t("no_classes"))}</div>`;
      return;
    }

    const checkedValues = new Set(Array.from(document.querySelectorAll(".classCheck:checked")).map((cb) => cb.value));

    classListEl.innerHTML = classMetadata.map((item) => {
      const shouldCheck = checkedValues.size === 0 || checkedValues.has(String(item.id));
      return `
        <label class="class-option">
          <input type="checkbox" class="classCheck" value="${escapeHtml(item.id)}" ${shouldCheck ? "checked" : ""}>
          <span class="class-option-swatch" style="background:${escapeHtml(item.color)}"></span>
          <span class="class-option-body">
            <span class="class-option-area">${escapeHtml(formatArea(item.surface_m2))}</span>
            <span class="class-option-name">${escapeHtml(translateClassName(item.name))}</span>
          </span>
        </label>
      `;
    }).join("");

    document.querySelectorAll(".classCheck").forEach((cb) => cb.addEventListener("change", updateClassification));
  }

  function renderClassLegend() {
    if (!classMetadata.length) {
      classLegendContent.innerHTML = `<div class="legend-hint">${escapeHtml(t("no_legend"))}</div>`;
      return;
    }

    classLegendContent.innerHTML = `<div class="legend-list">${classMetadata.map((item) => `
      <div class="legend-item">
        <span class="legend-swatch" style="background:${escapeHtml(item.color)}"></span>
        <span>${escapeHtml(translateClassName(item.name))}</span>
      </div>
    `).join("")}</div>`;
  }

  async function loadClassMetadata(year) {
    classListEl.innerHTML = `<div class="loading-text">${escapeHtml(t("loading_classes"))}</div>`;
    classLegendContent.innerHTML = `<div class="loading-text">${escapeHtml(t("loading_legend"))}</div>`;

    try {
      const payload = await fetchClassesFromServer(year);
      const source = Array.isArray(payload.classes) ? payload.classes : [];
      classMetadata = source.map(normalizeClassEntry).sort((a, b) => Number(a.id) - Number(b.id));
      if (!classMetadata.length) throw new Error("Aucune classe recuperee");
    } catch (error) {
      console.error("Erreur chargement des classes:", error);
      classMetadata = defaultClassIds.map((id, index) => normalizeClassEntry({ id, name: `Classe ${id}` }, index));
    }

    renderClassFilters();
    renderClassLegend();
  }

  function updateClassification() {
    const selected = [];
    document.querySelectorAll(".classCheck").forEach((cb) => {
      if (cb.checked) selected.push(cb.value);
    });

    if (selected.length === 0) {
      if (classifLayer && map.hasLayer(classifLayer)) map.removeLayer(classifLayer);
      return;
    }

    if (!classifLayer) return;
    if (!map.hasLayer(classifLayer)) classifLayer.addTo(map);
    classifLayer.setParams({ CQL_FILTER: "class IN (" + selected.join(",") + ")" });
  }

  async function getAvailableYears() {
    const res = await fetch(geoserverBase + "/ne/wms?service=WMS&request=GetCapabilities");
    const text = await res.text();
    const xml = new DOMParser().parseFromString(text, "text/xml");
    const layers = xml.getElementsByTagName("Name");
    const years = [];

    for (let i = 0; i < layers.length; i++) {
      const name = layers[i].textContent;
      if (!name.includes("Habitats_ULaval_")) continue;
      const year = Number(name.split("_").pop());
      if (!Number.isNaN(year)) years.push(year);
    }

    return years.sort((a, b) => a - b);
  }

  async function changeYear(year) {
    currentYear = Number(year);
    if (classifLayer && map.hasLayer(classifLayer)) map.removeLayer(classifLayer);
    classifLayer = createLayer(currentYear).addTo(map);
    overlayLayerRefs.classification = classifLayer;
    refreshLayerControl();
    document.getElementById("yearLabel").innerText = currentYear;
    await loadClassMetadata(currentYear);
    updateClassification();
    renderObservationsInfoPanel();
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
      console.error("Impossible de recuperer les annees:", error);
    }
    await changeYear(slider.value);
  }

  window.playAnimation = function () {
    if (interval) return;
    interval = setInterval(() => {
      let year = parseInt(slider.value, 10) + 1;
      if (year > Number(slider.max)) year = Number(slider.min);
      slider.value = year;
      changeYear(year);
    }, 1500);
  };

  window.stopAnimation = function () {
    clearInterval(interval);
    interval = null;
  };

  function applyObservationTypeColors() {
    document.querySelectorAll(".obsType").forEach((checkbox) => {
      const label = checkbox.closest("label");
      if (!label) return;
      const text = label.textContent.replace(/\s+/g, " ").trim();
      label.className = "obs-type-label";
      label.innerHTML = `
        <input type="checkbox" class="obsType" value="${escapeHtml(checkbox.value)}" ${checkbox.checked ? "checked" : ""}>
        <span class="obs-type-dot" style="background:${escapeHtml(groupColors[checkbox.value] || "#0a73d9")}"></span>
        <span class="obs-type-name">${escapeHtml(translateGroupName(checkbox.value) || text)}</span>
      `;
    });

    document.querySelectorAll(".obsType").forEach((cb) => cb.addEventListener("change", renderObservations));
  }

  async function loadAllProjectObservations(forceReload = true) {
    infoPanel.innerHTML = t("loading_observations");
    markers.clearLayers();
    const nextObservations = [];
    const nextLoadedIds = new Set();

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
      if (data.results.length < 200) hasMore = false;
      else page++;

      data.results.forEach((obs) => {
        if (!obs.geojson || !obs.geojson.coordinates) return;
        if (nextLoadedIds.has(obs.id)) return;
        nextLoadedIds.add(obs.id);
        nextObservations.push(obs);
      });
    }

    allObservations = nextObservations;
    loadedIds = nextLoadedIds;
    renderObservations();

    renderObservationsInfoPanel();
  }

  window.loadAllProjectObservations = function () {
    document.querySelectorAll(".obsType").forEach((cb) => {
      cb.checked = true;
    });
    return loadAllProjectObservations(true);
  };

  window.clearAllObservations = function () {
    markers.clearLayers();
    allObservations = [];
    loadedIds.clear();
    document.querySelectorAll(".obsType").forEach((cb) => {
      cb.checked = false;
    });
    infoPanel.innerHTML = t("observations_removed");
  };

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
      const markerColor = groupColors[groupFound] || "#0a73d9";
      const marker = L.circleMarker([lat, lng], {
        radius: 5,
        color: markerColor,
        fillColor: markerColor,
        fillOpacity: 0.95,
        weight: 1.5
      });

      const observationName =
        obs.taxon?.preferred_common_name ||
        obs.taxon?.names?.find((item) => item.locale === "fr")?.name ||
        obs.species_guess ||
        obs.taxon?.name ||
        t("observation_default_name");
      const frenchName =
        obs.taxon?.names?.find((item) => item.locale === "fr")?.name ||
        obs.taxon?.preferred_common_name ||
        "";
      const quality =
        obs.quality_grade === "research"
          ? t("research_label")
          : obs.quality_grade === "needs_id"
            ? t("needs_id_label")
            : (obs.quality_grade || t("unknown"));
      const imageUrl =
        obs.photos?.[0]?.url?.replace("square", "medium") ||
        obs.observation_photos?.[0]?.photo?.url?.replace("square", "medium") ||
        "";
      const sourceUrl = obs.uri || obs.html_url || "#";

      marker.bindPopup(`
        <div class="popup-card">
          ${imageUrl ? `<img class="popup-img" src="${escapeHtml(imageUrl)}" alt="${escapeHtml(observationName)}">` : ""}
          <h3><span class="popup-type-dot" style="background:${escapeHtml(markerColor)}"></span>${escapeHtml(observationName)}</h3>
          ${frenchName && frenchName !== observationName ? `<p><strong>${escapeHtml(t("french_name"))} :</strong> ${escapeHtml(frenchName)}</p>` : ""}
          <p><strong>${escapeHtml(t("type"))} :</strong> ${escapeHtml(translateGroupName(groupFound))}</p>
          <p><strong>${escapeHtml(t("quality"))} :</strong> ${escapeHtml(quality)}</p>
          <p><strong>${escapeHtml(t("date"))} :</strong> ${escapeHtml(obs.observed_on || t("unknown"))}</p>
          <a class="popup-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer">${escapeHtml(t("view_source"))}</a>
        </div>
      `);

      markers.addLayer(marker);
    });
  }

  window.openObservationsPanel = function () {
    const panel = document.getElementById("obsList");
    if (!panel) return;
    const isOpen = panel.style.display === "block";
    panel.style.display = isOpen ? "none" : "block";
    if (!isOpen) {
      document.querySelectorAll(".obsType").forEach((cb) => {
        cb.checked = true;
      });
      loadAllProjectObservations(true).catch((error) => console.error(error));
    }
  };

  window.toggleAirQualityPanel = async function () {
    if (!airQualityListEl) return;
    const isOpen = airQualityListEl.style.display === "block";
    airQualityListEl.style.display = isOpen ? "none" : "block";
    airQualityVisible = !isOpen;

    if (airQualityVisible) {
      if (!airQualityLoaded) {
        await loadAirQuality();
      } else {
        renderAirQualityPanel();
        renderAirQualityLayers();
      }
    } else {
      clearAirQualityLayers();
    }
  };

  window.exportMapImage = async function () {
    try {
      const exportedAt = new Date().toLocaleString(currentLanguage === "en" ? "en-CA" : "fr-CA", {
        dateStyle: "medium",
        timeStyle: "short"
      });
      const mapCanvas = await html2canvas(document.getElementById("map"), {
        useCORS: true,
        backgroundColor: null
      });

      const exportWrapper = document.createElement("div");
      exportWrapper.style.position = "fixed";
      exportWrapper.style.left = "-10000px";
      exportWrapper.style.top = "0";
      exportWrapper.style.width = "1320px";
      exportWrapper.style.padding = "30px";
      exportWrapper.style.background = "linear-gradient(180deg, #f8fafc, #eef3f8)";
      exportWrapper.style.fontFamily = '"Inter", "Segoe UI", sans-serif';
      exportWrapper.style.color = "#1f2937";
      exportWrapper.style.zIndex = "99999";

      const title = document.createElement("div");
      title.style.display = "flex";
      title.style.alignItems = "flex-start";
      title.style.justifyContent = "space-between";
      title.style.gap = "20px";
      title.style.marginBottom = "18px";
      title.style.padding = "18px 22px";
      title.style.borderRadius = "22px";
      title.style.background = "linear-gradient(135deg, #ffffff, #f6f8fb)";
      title.style.boxShadow = "0 16px 34px rgba(15,23,42,0.1)";
      title.style.border = "1px solid rgba(15,23,42,0.08)";
      title.innerHTML = `
        <div>
          <div style="font-size:31px;font-weight:800;color:#7c250b;line-height:1.08;">${escapeHtml(t("biodiversity_map_title"))}</div>
          <div style="margin-top:7px;font-size:15px;color:#475569;">${escapeHtml(t("classification_observations", { year: currentYear }))}</div>
        </div>
        <div style="display:grid;gap:8px;justify-items:end;">
          <div style="padding:7px 12px;border-radius:999px;background:#eef2f8;color:#17345f;font-size:13px;font-weight:700;">${escapeHtml(t("year"))} ${escapeHtml(currentYear)}</div>
          <div style="font-size:13px;color:#64748b;">${escapeHtml(t("exported_on"))} ${escapeHtml(exportedAt)}</div>
        </div>
      `;

      const mapImage = document.createElement("img");
      mapImage.src = mapCanvas.toDataURL("image/png");
      mapImage.style.display = "block";
      mapImage.style.width = "100%";
      mapImage.style.borderRadius = "18px";
      mapImage.style.boxShadow = "0 18px 40px rgba(15,23,42,0.18)";
      mapImage.style.border = "1px solid rgba(15,23,42,0.08)";

      const mapCard = document.createElement("div");
      mapCard.style.background = "#ffffff";
      mapCard.style.borderRadius = "22px";
      mapCard.style.padding = "14px";
      mapCard.style.boxShadow = "0 16px 38px rgba(15,23,42,0.12)";
      mapCard.style.border = "1px solid rgba(15,23,42,0.08)";
      mapCard.appendChild(mapImage);

      const metaGrid = document.createElement("div");
      metaGrid.style.display = "grid";
      metaGrid.style.gridTemplateColumns = "minmax(0, 0.95fr) minmax(0, 1.05fr)";
      metaGrid.style.gap = "16px";
      metaGrid.style.marginTop = "18px";

      const infoClone = cloneForExport(infoPanel);
      const legendClone = cloneForExport(classLegendPanel);

      const scaleCard = document.createElement("div");
      scaleCard.style.background = "rgba(255,255,255,0.98)";
      scaleCard.style.borderRadius = "18px";
      scaleCard.style.padding = "14px 16px";
      scaleCard.style.boxShadow = "0 12px 28px rgba(15,23,42,0.1)";
      scaleCard.style.border = "1px solid rgba(15,23,42,0.08)";
      scaleCard.innerHTML = `<div style="font-size:13px;font-weight:700;color:#7c250b;margin-bottom:10px;">${escapeHtml(t("scale"))}</div>`;

      const liveScale = document.querySelector(".leaflet-control-scale");
      if (liveScale) {
        const scaleClone = liveScale.cloneNode(true);
        scaleClone.style.position = "static";
        scaleClone.style.display = "inline-block";
        scaleClone.style.margin = "0";
        scaleCard.appendChild(scaleClone);
      } else {
        const noScale = document.createElement("div");
        noScale.textContent = t("scale_unavailable");
        noScale.style.fontSize = "12px";
        noScale.style.color = "#64748b";
        scaleCard.appendChild(noScale);
      }

      const leftColumn = document.createElement("div");
      leftColumn.style.display = "grid";
      leftColumn.style.gap = "16px";
      leftColumn.appendChild(infoClone);
      leftColumn.appendChild(scaleCard);

      metaGrid.appendChild(leftColumn);
      metaGrid.appendChild(legendClone);

      const footer = document.createElement("div");
      footer.style.marginTop = "16px";
      footer.style.padding = "12px 16px";
      footer.style.borderRadius = "16px";
      footer.style.background = "rgba(255,255,255,0.82)";
      footer.style.border = "1px solid rgba(15,23,42,0.06)";
      footer.style.fontSize = "12px";
      footer.style.color = "#64748b";
      footer.style.display = "flex";
      footer.style.justifyContent = "space-between";
      footer.style.gap = "16px";
      footer.innerHTML = `
        <span>${escapeHtml(t("platform_footer"))}</span>
        <span>${escapeHtml(t("source_footer"))}</span>
      `;

      exportWrapper.appendChild(title);
      exportWrapper.appendChild(mapCard);
      exportWrapper.appendChild(metaGrid);
      exportWrapper.appendChild(footer);
      document.body.appendChild(exportWrapper);

      const finalCanvas = await html2canvas(exportWrapper, {
        useCORS: true,
        backgroundColor: "#f8fafc"
      });

      exportWrapper.remove();

      finalCanvas.toBlob((blob) => {
        if (!blob) return;
        const url = URL.createObjectURL(blob);
        const link = document.createElement("a");
        link.href = url;
        link.download = `${currentLanguage === "en" ? "biodiversity_map" : "carte_biodiversite"}_${currentYear}.png`;
        document.body.appendChild(link);
        link.click();
        link.remove();
        URL.revokeObjectURL(url);
      }, "image/png");
    } catch (error) {
      console.error(error);
      infoPanel.innerHTML = escapeHtml(t("export_error"));
    }
  };

  window.generateMonthlyReport = function () {
    const now = new Date();
    const monthLabel = now.toLocaleString(currentLanguage === "en" ? "en-CA" : "fr-CA", { month: "long", year: "numeric" });
    const stats = getObservationStats();
    const selectedObservationTypes = Array.from(document.querySelectorAll(".obsType:checked"))
      .map((cb) => cb.value)
      .map((name) => translateGroupName(name))
      .join(", ") || t("none");

    const classRows = classMetadata.map((item) => `
      <tr>
        <td>${escapeHtml(translateClassName(item.name))}</td>
        <td>${escapeHtml(formatArea(item.surface_m2))}</td>
      </tr>
    `).join("");

    const report = `<!DOCTYPE html>
<html lang="${escapeHtml(currentLanguage)}">
<head>
  <meta charset="UTF-8">
  <title>${escapeHtml(t("report_title"))} ${escapeHtml(monthLabel)}</title>
  <style>
    body { font-family: Segoe UI, sans-serif; margin: 32px; color: #1f2937; }
    h1, h2 { color: #7c250b; }
    .meta, .card { margin-bottom: 24px; }
    .card { padding: 18px; border: 1px solid #ead7cf; border-radius: 14px; background: #fffaf8; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 10px; border-bottom: 1px solid #ead7cf; text-align: left; }
    th { color: #7c250b; }
  </style>
</head>
<body>
  <h1>${escapeHtml(t("report_title"))}</h1>
  <div class="meta">
    <strong>${escapeHtml(t("month"))} :</strong> ${escapeHtml(monthLabel)}<br>
    <strong>${escapeHtml(t("year"))} :</strong> ${escapeHtml(currentYear)}<br>
    <strong>${escapeHtml(t("observation_types"))} :</strong> ${escapeHtml(selectedObservationTypes)}<br>
    <strong>${escapeHtml(t("observation_srs"))} :</strong> EPSG:4326<br>
    <strong>${escapeHtml(t("classification_srs"))} :</strong> EPSG:4326
  </div>

  <div class="card">
    <h2>${escapeHtml(t("observation_stats"))}</h2>
    <p><strong>${escapeHtml(t("total"))} :</strong> ${escapeHtml(stats.total)}</p>
    <p><strong>Research :</strong> ${escapeHtml(stats.researchCount)} (${escapeHtml(stats.researchPercent)}%)</p>
    <p><strong>Needs ID :</strong> ${escapeHtml(stats.needsIdCount)} (${escapeHtml(stats.needsIdPercent)}%)</p>
  </div>

  <div class="card">
    <h2>${escapeHtml(t("surface_evolution"))}</h2>
    <table>
      <thead>
        <tr>
          <th>${escapeHtml(t("class_label"))}</th>
          <th>${escapeHtml(t("area_label"))}</th>
        </tr>
      </thead>
      <tbody>
        ${classRows}
      </tbody>
    </table>
  </div>

  <div class="card">
    <h2>${escapeHtml(t("summary"))}</h2>
    <p>${escapeHtml(t("report_summary_text", { year: currentYear }))}</p>
  </div>
</body>
</html>`;

    const filename = `rapport_mensuel_biodiversite_${now.getFullYear()}_${String(now.getMonth() + 1).padStart(2, "0")}.html`;
    downloadTextFile(filename, report, "text/html;charset=utf-8");
  };

  function startObservationsAutoRefresh() {
    if (observationsRefreshInterval) clearInterval(observationsRefreshInterval);
    observationsRefreshInterval = setInterval(() => {
      loadAllProjectObservations(true).catch((error) => console.error(error));
    }, 24 * 60 * 60 * 1000);
  }

  function hideSearchSuggestions() {
    if (!searchSuggestions) return;
    searchSuggestions.style.display = "none";
    searchSuggestions.innerHTML = "";
  }

  function showSearchSuggestions(items) {
    if (!searchSuggestions) return;
    if (!items.length) {
      searchSuggestions.innerHTML = `<div class="search-empty">${escapeHtml(t("search_no_result"))}</div>`;
      searchSuggestions.style.display = "block";
      return;
    }

    searchSuggestions.innerHTML = items.map((item) => `
      <button type="button" class="search-suggestion" data-lat="${escapeHtml(item.lat)}" data-lon="${escapeHtml(item.lon)}" data-label="${escapeHtml(item.display_name || item.name || "")}">
        <span class="search-suggestion-title">${escapeHtml(item.name || item.display_name || t("result"))}</span>
        <span class="search-suggestion-subtitle">${escapeHtml(item.display_name || "")}</span>
      </button>
    `).join("");
    searchSuggestions.style.display = "block";

    searchSuggestions.querySelectorAll(".search-suggestion").forEach((button) => {
      button.addEventListener("click", function () {
        const lat = Number(this.dataset.lat);
        const lon = Number(this.dataset.lon);
        const label = this.dataset.label || t("address");
        if (Number.isNaN(lat) || Number.isNaN(lon)) return;
        map.setView([lat, lon], 18);
        if (searchMarker) map.removeLayer(searchMarker);
        searchMarker = L.marker([lat, lon]).addTo(map).bindPopup(label).openPopup();
        addressSearch.value = label;
        hideSearchSuggestions();
      });
    });
  }

  async function fetchAddressSuggestions(query) {
    if (!query || query.trim().length < 3) {
      hideSearchSuggestions();
      return;
    }
    if (searchAbortController) searchAbortController.abort();
    searchAbortController = new AbortController();
    try {
      const response = await fetch(
        "https://nominatim.openstreetmap.org/search?format=jsonv2&limit=5&countrycodes=ca&q=" + encodeURIComponent(query),
        {
          signal: searchAbortController.signal,
          headers: { Accept: "application/json" }
        }
      );
      const results = await response.json();
      showSearchSuggestions(results);
    } catch (error) {
      if (error.name !== "AbortError") console.error("Erreur suggestions adresse:", error);
    }
  }

  window.closeClassChart = function () {
    chartModal.classList.remove("is-open");
  };

  window.openClassChart = async function () {
    chartModal.classList.add("is-open");
    chartModalSubtitle.innerText = t("loading_chart");
    destroyClassChart();

    try {
      const years = await getAvailableYears();
      const series = await Promise.all(
        years.map(async (year) => {
          const payload = await fetchClassesFromServer(year);
          return { year, classes: (payload.classes || []).map(normalizeClassEntry) };
        })
      );

      const classMap = new Map();
      series.forEach(({ classes }) => {
        classes.forEach((item) => {
          if (!classMap.has(item.id)) classMap.set(item.id, { name: item.name, color: item.color });
        });
      });

      const datasets = Array.from(classMap.entries()).map(([classId, meta]) => ({
        label: translateClassName(meta.name),
        data: years.map((year) => {
          const yearData = series.find((entry) => entry.year === year);
          const found = yearData?.classes.find((item) => Number(item.id) === Number(classId));
          return found ? Number((found.surface_m2 / 10000).toFixed(2)) : 0;
        }),
        borderColor: meta.color,
        backgroundColor: meta.color,
        tension: 0.28,
        fill: false
      }));

      classAreaChart = new Chart(classAreaCanvas.getContext("2d"), {
        type: "line",
        data: { labels: years, datasets },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: { position: "bottom" },
            tooltip: {
              callbacks: {
                label: function (context) {
                  return context.dataset.label + ": " + context.parsed.y.toLocaleString(currentLanguage === "en" ? "en-CA" : "fr-CA") + " ha";
                }
              }
            }
          },
          scales: {
            y: { title: { display: true, text: t("area_ha") } },
            x: { title: { display: true, text: t("year") } }
          }
        }
      });

      chartModalSubtitle.innerText = t("chart_surface_subtitle");
    } catch (error) {
      console.error(error);
      chartModalSubtitle.innerText = t("chart_unavailable");
    }
  };

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
        if (!data.features || !data.features.length) return;
        const props = data.features[0].properties || {};
        const surface = props.surface_m2 ? formatArea(props.surface_m2) : t("not_available");
        L.popup().setLatLng(e.latlng).setContent(`
          <b>${escapeHtml(t("class_label"))} :</b> ${escapeHtml(translateClassName(props.nom_classe || String(props.class || t("unknown"))))}<br>
          <b>${escapeHtml(t("area_label"))} :</b> ${escapeHtml(surface)}
        `).openOn(map);
      })
      .catch((err) => console.error("Erreur GetFeatureInfo:", err));
  });

  window.goHome = function () {
    map.setView([46.781, -71.277], 15);
    markers.clearLayers();
    hideSearchSuggestions();
    if (addressSearch) {
      addressSearch.value = "";
    }
    if (searchMarker && map.hasLayer(searchMarker)) {
      map.removeLayer(searchMarker);
      searchMarker = null;
    }
    map.closePopup();
    if (classifLayer && map.hasLayer(classifLayer)) map.removeLayer(classifLayer);
  };

  slider.addEventListener("input", function () {
    changeYear(this.value);
  });

  if (addressSearch) {
    addressSearch.addEventListener("input", function () {
      clearTimeout(searchDebounce);
      const value = this.value;
      searchDebounce = setTimeout(() => {
        fetchAddressSuggestions(value);
      }, 250);
    });

    addressSearch.addEventListener("focus", function () {
      if (this.value.trim().length >= 3) fetchAddressSuggestions(this.value);
    });
  }

  if (languageSelect) {
    languageSelect.value = currentLanguage;
    languageSelect.addEventListener("change", function () {
      currentLanguage = this.value;
      localStorage.setItem("platformLanguage", currentLanguage);
      updateStaticTranslations();
      refreshLayerControl();
      applyObservationTypeColors();
      renderClassFilters();
      renderClassLegend();
      renderObservationsInfoPanel();
      if (airQualityLoaded) {
        renderAirQualityPanel();
        if (airQualityVisible) renderAirQualityLayers();
      }
    });
  }

  document.addEventListener("click", function (event) {
    if (!event.target.closest("#searchPanel")) hideSearchSuggestions();
  });

  chartModal.addEventListener("click", function (event) {
    if (event.target === chartModal) closeClassChart();
  });

  updateStaticTranslations();
  refreshLayerControl();
  applyObservationTypeColors();
  setupYearSlider();
  loadAllProjectObservations(true);
  startObservationsAutoRefresh();

});
