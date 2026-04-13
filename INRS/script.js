document.addEventListener("DOMContentLoaded", function () {
  const directGeoserverBase = "http://localhost:8080/geoserver";
  const serverBase = window.location.origin;
  const apiBase = serverBase.startsWith("http") ? serverBase : "";
  const geoserverBase = apiBase ? apiBase + "/geoserver" : directGeoserverBase;
  let currentLanguage = localStorage.getItem("inrsLanguage") || "fr";

  const campusConfig = {
    qc: {
      key: "qc",
      label: { fr: "INRS Quebec", en: "INRS Quebec" },
      center: [46.79643680007479, -71.30277395255203],
      zoom: 17,
      secteur: "ne:INRS_emprise_Qc",
      layerPrefix: "Habitats_INRS_Qc_",
      inaturalist: {
        projectId: "inrs-reseau-biodiversite-sur-le-campus",
        lat: 46.79643680007479,
        lng: -71.30277395255203,
        radius: 0.3
      }
    },
    laval: {
      key: "laval",
      label: { fr: "INRS Laval", en: "INRS Laval" },
      center: [45.54543960384367, -73.7202473476199],
      zoom: 17,
      secteur: "ne:INRS_emprise_Laval",
      layerPrefix: "Habitats_INRS_Laval_",
      inaturalist: {
        projectId: "inrs-reseau-biodiversite-sur-le-campus",
        lat: 45.54543960384367,
        lng: -73.7202473476199,
        radius: 0.3
      }
    }
  };

  const translations = {
    fr: {
      app_title: "Biodiversite INRS",
      app_subtitle: "Deux campus, une interface claire pour suivre les habitats et les observations.",
      language: "Langue",
      home: "Accueil",
      campus_menu: "Campus INRS",
      campus_qc: "INRS Quebec",
      campus_qc_address: "2605 Bd du Parc-Technologique",
      campus_laval: "INRS Laval",
      campus_laval_address: "531 Boul. des Prairies",
      habitat_classification: "Classification habitat",
      loading_classes: "Chargement des classes...",
      temporal_tracking: "Suivi temporel",
      active_year: "Annee active",
      surface_chart: "Graphique superficies",
      export_png: "Exporter carte PNG",
      monthly_report: "Generer rapport mensuel",
      observations: "Observations iNaturalist",
      load_observations: "Charger observations",
      remove_observations: "Retirer observations",
      select_campus_hint: "Selectionnez un campus pour afficher la classification et les observations.",
      class_legend: "Legende des classes",
      loading_legend: "Chargement de la legende...",
      loading: "Chargement...",
      close: "Fermer",
      no_legend: "Aucune legende disponible.",
      no_classes: "Aucune classe disponible.",
      realtime_observations: "Observations iNaturalist temps reel",
      campus: "Campus",
      classification_year: "Annee classification",
      statistics: "Statistiques",
      observations_removed: "Observations retirees de la carte.",
      french_name: "Nom francais",
      type: "Type",
      quality: "Qualite",
      date: "Date",
      unknown: "Inconnue",
      source_link: "Voir l'observation source",
      export_error: "Impossible d'exporter l'image de la carte.",
      export_title: "Carte de biodiversite INRS",
      export_subtitle: "{campus} | Classification {year} | Observations iNaturalist",
      exported_on: "Exporte le",
      scale: "Echelle",
      scale_unavailable: "Echelle non disponible",
      platform_footer: "Plateforme biodiversite campus INRS",
      source_footer: "Source : GeoServer + iNaturalist",
      report_title: "Rapport mensuel biodiversite INRS",
      month: "Mois",
      observation_types: "Types d'observations affiches",
      observation_srs: "Systeme de reference des observations",
      classification_srs: "Systeme de reference de la classification carte",
      observation_stats: "Statistiques observations",
      total: "Total",
      class_label: "Classe",
      area_label: "Superficie",
      summary: "Resume",
      report_summary_text: "Ce rapport resume l'etat courant des observations iNaturalist filtrees en Research et Needs ID, ainsi que la superficie des classes de la classification pour {campus} en {year}.",
      chart_loading: "Chargement du graphique...",
      chart_error: "Impossible de charger le graphique.",
      chart_subtitle: "{campus} - evolution des superficies par classe",
      area_axis: "Superficie (ha)",
      year_axis: "Annee",
      base_map: "Carte",
      street_map: "Street",
      satellite_map: "Satellite",
      topo_map: "Topo",
      footprint: "Emprise",
      map_classification: "Classification",
      research_label: "Research",
      needs_id_label: "ID manquant"
    },
    en: {
      app_title: "INRS Biodiversity",
      app_subtitle: "Two campuses, one clear interface to monitor habitats and observations.",
      language: "Language",
      home: "Home",
      campus_menu: "INRS Campuses",
      campus_qc: "INRS Quebec",
      campus_qc_address: "2605 Parc-Technologique Blvd",
      campus_laval: "INRS Laval",
      campus_laval_address: "531 des Prairies Blvd",
      habitat_classification: "Habitat classification",
      loading_classes: "Loading classes...",
      temporal_tracking: "Temporal tracking",
      active_year: "Active year",
      surface_chart: "Area chart",
      export_png: "Export PNG map",
      monthly_report: "Generate monthly report",
      observations: "iNaturalist observations",
      load_observations: "Load observations",
      remove_observations: "Remove observations",
      select_campus_hint: "Select a campus to display classification and observations.",
      class_legend: "Class legend",
      loading_legend: "Loading legend...",
      loading: "Loading...",
      close: "Close",
      no_legend: "No legend available.",
      no_classes: "No class available.",
      realtime_observations: "Real-time iNaturalist observations",
      campus: "Campus",
      classification_year: "Classification year",
      statistics: "Statistics",
      observations_removed: "Observations removed from the map.",
      french_name: "French name",
      type: "Type",
      quality: "Quality",
      date: "Date",
      unknown: "Unknown",
      source_link: "View source observation",
      export_error: "Unable to export the map image.",
      export_title: "INRS biodiversity map",
      export_subtitle: "{campus} | Classification {year} | iNaturalist observations",
      exported_on: "Exported on",
      scale: "Scale",
      scale_unavailable: "Scale unavailable",
      platform_footer: "INRS campus biodiversity platform",
      source_footer: "Source: GeoServer + iNaturalist",
      report_title: "INRS monthly biodiversity report",
      month: "Month",
      observation_types: "Displayed observation types",
      observation_srs: "Observation reference system",
      classification_srs: "Map classification reference system",
      observation_stats: "Observation statistics",
      total: "Total",
      class_label: "Class",
      area_label: "Area",
      summary: "Summary",
      report_summary_text: "This report summarizes the current state of iNaturalist observations filtered as Research and Needs ID, as well as the class areas of the classification for {campus} in {year}.",
      chart_loading: "Loading chart...",
      chart_error: "Unable to load the chart.",
      chart_subtitle: "{campus} - class area evolution",
      area_axis: "Area (ha)",
      year_axis: "Year",
      base_map: "Map",
      street_map: "Street",
      satellite_map: "Satellite",
      topo_map: "Topo",
      footprint: "Footprint",
      map_classification: "Classification",
      research_label: "Research",
      needs_id_label: "Needs ID"
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

  const classColorById = {
    0: "#0aa000",
    1: "#d9d9d9",
    2: "#f08a3c",
    3: "#8be85f",
    8: "#f7b6c8",
    10: "#d67cff"
  };
  const fallbackPalette = ["#0aa000", "#d9d9d9", "#f08a3c", "#8be85f", "#f7b6c8", "#d67cff"];

  const map = L.map("map", { maxZoom: 22 }).setView(campusConfig.qc.center, campusConfig.qc.zoom);
  const baseLayers = {
    Carte: L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
      attribution: "&copy; OpenStreetMap &copy; CARTO"
    }),
    Street: L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors"
    }),
    Satellite: L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      attribution: "Tiles &copy; Esri"
    }),
    Topo: L.tileLayer("https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenTopoMap contributors"
    })
  };
  baseLayers.Carte.addTo(map);
  const baseLayerRefs = {
    carte: baseLayers.Carte,
    street: baseLayers.Street,
    satellite: baseLayers.Satellite,
    topo: baseLayers.Topo
  };
  const overlayLayerRefs = {
    footprint: null,
    classification: null
  };
  let layerControl = null;
  L.control.scale({ imperial: false, position: "bottomleft" }).addTo(map);

  const markers = L.layerGroup().addTo(map);
  const slider = document.getElementById("yearSlider");
  const classListEl = document.getElementById("classList");
  const classLegendContent = document.getElementById("classLegendContent");
  const classLegendPanel = document.getElementById("classLegendPanel");
  const infoPanel = document.getElementById("infoPanel");
  const languageSelect = document.getElementById("languageSelect");
  const chartModal = document.getElementById("chartModal");
  const chartModalSubtitle = document.getElementById("chartModalSubtitle");
  const classAreaCanvas = document.getElementById("classAreaChart");

  let currentCampus = "qc";
  let currentYear = Number(slider.value);
  let classifLayer = null;
  let secteurLayer = null;
  let campusMarker = null;
  let interval = null;
  let observationsRefreshInterval = null;
  let classMetadata = [];
  let allObservations = [];
  let loadedIds = new Set();
  const campusBoundaries = {};
  let classAreaChart = null;

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

  function campusLabel(name) {
    return campusConfig[name]?.label?.[currentLanguage] || campusConfig[name]?.label?.fr || name;
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
    return {
      [t("footprint")]: overlayLayerRefs.footprint,
      [t("map_classification")]: overlayLayerRefs.classification
    };
  }

  function refreshLayerControl() {
    if (layerControl) map.removeControl(layerControl);
    layerControl = L.control.layers(getLocalizedBaseLayers(), getLocalizedOverlayLayers(), {
      collapsed: true
    }).addTo(map);
  }

  function applyTranslations() {
    document.documentElement.lang = currentLanguage;
    document.title = t("app_title");
    document.querySelectorAll("[data-i18n]").forEach((node) => {
      node.textContent = t(node.dataset.i18n);
    });
    document.querySelectorAll("[data-group-label]").forEach((node) => {
      node.textContent = translateGroupName(node.dataset.groupLabel);
    });
    if (languageSelect) languageSelect.value = currentLanguage;
    renderClassFilters();
    renderClassLegend();
    renderObservationsInfoPanel();
    applyObservationTypeColors();
    refreshLayerControl();
  }

  function normalizeClassEntry(entry, index) {
    const rawId = entry.id ?? entry.class ?? index;
    const classId = Number(rawId);
    const safeId = Number.isNaN(classId) ? index : classId;
    const surfaceM2 = Number(entry.surface_m2 ?? 0);
    return {
      id: safeId,
      name: entry.name || entry.nom_classe || `Classe ${safeId}`,
      color: entry.color || classColorById[safeId] || fallbackPalette[index % fallbackPalette.length],
      surface_m2: Number.isNaN(surfaceM2) ? 0 : surfaceM2
    };
  }

  function formatArea(surfaceM2) {
    if (!surfaceM2) return "0 m²";
    return Number(surfaceM2).toLocaleString(currentLanguage === "en" ? "en-CA" : "fr-CA", {
      minimumFractionDigits: 0,
      maximumFractionDigits: 0
    }) + " m²";
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

  function renderObservationsInfoPanel() {
    const researchCount = allObservations.filter((obs) => obs.quality_grade === "research").length;
    const needsIdCount = allObservations.filter((obs) => obs.quality_grade === "needs_id").length;
    const total = researchCount + needsIdCount;
    const researchPercent = total ? ((researchCount / total) * 100).toFixed(1) : "0.0";
    const needsIdPercent = total ? ((needsIdCount / total) * 100).toFixed(1) : "0.0";
    infoPanel.innerHTML = `
      <b>${escapeHtml(t("realtime_observations"))} :</b> ${escapeHtml(allObservations.length)}<br><br>
      <b>${escapeHtml(t("campus"))} :</b> ${escapeHtml(campusLabel(currentCampus))}<br>
      <b>${escapeHtml(t("classification_year"))} :</b> ${escapeHtml(currentYear || "-")}<br><br>
      <b>${escapeHtml(t("statistics"))} :</b><br>
      ${escapeHtml(t("research_label"))} : ${escapeHtml(researchPercent)}%<br>
      ${escapeHtml(t("needs_id_label"))} : ${escapeHtml(needsIdPercent)}%
    `;
  }

  function wait(ms) {
    return new Promise((resolve) => window.setTimeout(resolve, ms));
  }

  async function stabilizeMapForExport() {
    map.invalidateSize(true);
    await wait(180);
    map.invalidateSize(true);
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    await wait(120);
  }

  function getSelectedObservationTypes() {
    return Array.from(document.querySelectorAll(".obsType:checked")).map((cb) => cb.value);
  }

  function getObservationGroup(obs) {
    const ancestorIds = obs.taxon?.ancestor_ids || [];
    for (const group in taxonGroups) {
      if (ancestorIds.includes(taxonGroups[group])) {
        return group;
      }
    }
    return null;
  }

  async function buildExportMapImage() {
    const exportMapHost = document.createElement("div");
    exportMapHost.style.position = "fixed";
    exportMapHost.style.left = "-10000px";
    exportMapHost.style.top = "0";
    exportMapHost.style.width = "1180px";
    exportMapHost.style.height = "700px";
    exportMapHost.style.background = "#ffffff";
    exportMapHost.style.padding = "0";
    exportMapHost.style.overflow = "hidden";

    const exportMapEl = document.createElement("div");
    exportMapEl.style.width = "100%";
    exportMapEl.style.height = "100%";
    exportMapHost.appendChild(exportMapEl);
    document.body.appendChild(exportMapHost);

    const exportMap = L.map(exportMapEl, {
      zoomControl: false,
      attributionControl: true,
      preferCanvas: true
    });

    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
      attribution: "&copy; OpenStreetMap &copy; CARTO"
    }).addTo(exportMap);

    const campus = campusLabel(currentCampus);
    L.tileLayer.wms(geoserverBase + "/ne/wms", {
      layers: campus.secteur,
      format: "image/png",
      transparent: true
    }).addTo(exportMap);

    if (currentYear) {
      L.tileLayer.wms(geoserverBase + "/ne/wms", {
        layers: "ne:" + campus.layerPrefix + currentYear,
        format: "image/png",
        transparent: true
      }).addTo(exportMap);
    }

    const selected = new Set(getSelectedObservationTypes());
    allObservations.forEach((obs) => {
      if (!obs.geojson || !obs.geojson.coordinates) return;
      const groupFound = getObservationGroup(obs);
      if (!groupFound || !selected.has(groupFound)) return;

      const [lng, lat] = obs.geojson.coordinates;
      const markerColor = groupColors[groupFound] || "#0a73d9";
      L.circleMarker([lat, lng], {
        radius: 5,
        color: markerColor,
        fillColor: markerColor,
        fillOpacity: 0.95,
        weight: 1.5
      }).addTo(exportMap);
    });

    exportMap.fitBounds(map.getBounds(), { animate: false, padding: [0, 0] });
    exportMap.invalidateSize(true);
    await wait(1000);
    exportMap.invalidateSize(true);
    await new Promise((resolve) => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    await wait(400);

    const mapCanvas = await html2canvas(exportMapEl, {
      useCORS: true,
      backgroundColor: null,
      logging: false,
      scale: 1
    });

    exportMap.remove();
    exportMapHost.remove();
    return mapCanvas;
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

  function createLayer(year) {
    const prefix = campusConfig[currentCampus].layerPrefix;
    return L.tileLayer.wms(geoserverBase + "/ne/wms", {
      layers: "ne:" + prefix + year,
      format: "image/png",
      transparent: true
    });
  }

  async function fetchClasses(year) {
    const res = await fetch(apiBase + "/api/classes?campus=" + encodeURIComponent(currentCampus) + "&year=" + encodeURIComponent(year));
    if (!res.ok) throw new Error("Reponse serveur invalide");
    return res.json();
  }

  function destroyClassChart() {
    if (classAreaChart) {
      classAreaChart.destroy();
      classAreaChart = null;
    }
  }

  async function getCampusBoundary(campusKey) {
    if (campusBoundaries[campusKey]) {
      return campusBoundaries[campusKey];
    }

    const layerName = campusConfig[campusKey].secteur;
    const url =
      geoserverBase +
      "/ne/ows?service=WFS&version=1.0.0&request=GetFeature&typeName=" +
      encodeURIComponent(layerName) +
      "&outputFormat=application/json";

    const res = await fetch(url);
    if (!res.ok) {
      throw new Error("Impossible de charger les emprises du campus.");
    }

    const featureCollection = await res.json();
    campusBoundaries[campusKey] = featureCollection;
    return featureCollection;
  }

  function fitCampusBoundary(boundaryCollection, fallbackCenter, fallbackZoom) {
    try {
      const bounds = L.geoJSON(boundaryCollection).getBounds();
      if (bounds.isValid()) {
        map.fitBounds(bounds.pad(0.08));
        return;
      }
    } catch (error) {
      console.error("Erreur zoom emprise campus:", error);
    }

    map.setView(fallbackCenter, fallbackZoom);
  }

  function isObservationInsideCampus(obs, boundaryCollection) {
    if (!obs.geojson || !obs.geojson.coordinates || !boundaryCollection?.features?.length) {
      return false;
    }

    const point = turf.point(obs.geojson.coordinates);
    return boundaryCollection.features.some((feature) => {
      try {
        return turf.booleanPointInPolygon(point, feature);
      } catch (error) {
        console.error("Erreur verification emprise:", error);
        return false;
      }
    });
  }

  async function loadClassMetadata(year) {
    classListEl.innerHTML = `<div class="loading-text">${escapeHtml(t("loading_classes"))}</div>`;
    classLegendContent.innerHTML = `<div class="loading-text">${escapeHtml(t("loading_legend"))}</div>`;
    try {
      const payload = await fetchClasses(year);
      classMetadata = (payload.classes || []).map(normalizeClassEntry).sort((a, b) => Number(a.id) - Number(b.id));
    } catch (error) {
      console.error(error);
      classMetadata = [];
    }
    renderClassFilters();
    renderClassLegend();
  }

  function updateClassification() {
    if (!classifLayer) return;
    const selected = [];
    document.querySelectorAll(".classCheck").forEach((cb) => {
      if (cb.checked) selected.push(cb.value);
    });

    if (selected.length === 0) {
      if (map.hasLayer(classifLayer)) map.removeLayer(classifLayer);
      return;
    }

    if (!map.hasLayer(classifLayer)) classifLayer.addTo(map);
    classifLayer.setParams({
      CQL_FILTER: "class IN (" + selected.join(",") + ")"
    });
  }

  async function getAvailableYears() {
    const res = await fetch(geoserverBase + "/ne/wms?service=WMS&request=GetCapabilities");
    const text = await res.text();
    const xml = new DOMParser().parseFromString(text, "text/xml");
    const names = xml.getElementsByTagName("Name");
    const prefix = campusConfig[currentCampus].layerPrefix;
    const years = [];

    for (let i = 0; i < names.length; i++) {
      const name = names[i].textContent;
      if (!name.includes(prefix)) continue;
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
    document.getElementById("yearLabel").innerText = currentYear;
    await loadClassMetadata(currentYear);
    updateClassification();
    refreshLayerControl();
    renderObservationsInfoPanel();
  }

  async function setupYearSlider() {
    const years = await getAvailableYears();
    if (!years.length) return;
    slider.min = Math.min(...years);
    slider.max = Math.max(...years);
    slider.value = slider.max;
    await changeYear(slider.value);
  }

  slider.addEventListener("input", function () {
    changeYear(this.value);
  });

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

  function applyObservationTypeColors() {
    document.querySelectorAll(".obsType").forEach((checkbox) => {
      const label = checkbox.closest("label");
      if (!label) return;
      const text = translateGroupName(checkbox.value);
      label.className = "obs-type-label";
        label.innerHTML = `
          <input type="checkbox" class="obsType" value="${escapeHtml(checkbox.value)}" ${checkbox.checked ? "checked" : ""}>
          <span class="obs-type-dot" style="color:${escapeHtml(groupColors[checkbox.value] || "#0a73d9")}">●</span>
          <span class="obs-type-name">${escapeHtml(text)}</span>
        `;
      });

    document.querySelectorAll(".obsType").forEach((cb) => cb.addEventListener("change", renderObservations));
  }

  async function loadAllProjectObservations(forceReload = true) {
    markers.clearLayers();
    if (forceReload) {
      allObservations = [];
      loadedIds.clear();
    }
    const campus = campusConfig[currentCampus];
    const boundaryCollection = await getCampusBoundary(currentCampus);

    let page = 1;
    let hasMore = true;
    while (hasMore) {
      const url =
        "https://api.inaturalist.org/v1/observations?" +
        "project_id=" + campus.inaturalist.projectId +
        "&quality_grade=research,needs_id" +
        "&per_page=200&page=" + page;

      const res = await fetch(url);
      const data = await res.json();
      if (!data.results) break;
      if (data.results.length < 200) hasMore = false; else page++;

      data.results.forEach((obs) => {
        if (!obs.geojson || !obs.geojson.coordinates) return;
        if (loadedIds.has(obs.id)) return;
        if (!isObservationInsideCampus(obs, boundaryCollection)) return;
        loadedIds.add(obs.id);
        allObservations.push(obs);
      });
    }

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
    infoPanel.innerHTML = escapeHtml(t("observations_removed"));
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
        t("observations");
      const frenchName =
        obs.taxon?.names?.find((item) => item.locale === "fr")?.name ||
        obs.taxon?.preferred_common_name ||
        "";
      const quality =
        obs.quality_grade === "research"
          ? "Research"
          : obs.quality_grade === "needs_id"
            ? "Needs ID"
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
          <a class="popup-link" href="${escapeHtml(sourceUrl)}" target="_blank" rel="noreferrer">${escapeHtml(t("source_link"))}</a>
        </div>
      `);
      markers.addLayer(marker);
    });
  }

  applyObservationTypeColors();

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

  window.exportMapImage = async function () {
    try {
      const exportedAt = new Date().toLocaleString(currentLanguage === "en" ? "en-CA" : "fr-CA", {
        dateStyle: "medium",
        timeStyle: "short"
      });
      await stabilizeMapForExport();
      const mapCanvas = await buildExportMapImage();

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
            <div style="font-size:31px;font-weight:800;color:#0d3d7d;line-height:1.08;">${escapeHtml(t("export_title"))}</div>
            <div style="margin-top:7px;font-size:15px;color:#475569;">${escapeHtml(t("export_subtitle", { campus: campusLabel(currentCampus), year: String(currentYear) }))}</div>
          </div>
          <div style="display:grid;gap:8px;justify-items:end;">
            <div style="padding:7px 12px;border-radius:999px;background:#eef2f8;color:#17345f;font-size:13px;font-weight:700;">${escapeHtml(t("classification_year"))} ${escapeHtml(currentYear)}</div>
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
      scaleCard.innerHTML = `<div style="font-size:13px;font-weight:700;color:#0d3d7d;margin-bottom:10px;">${escapeHtml(t("scale"))}</div>`;

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
        link.download = `carte_biodiversite_inrs_${currentCampus}_${currentYear}.png`;
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
    const campus = campusConfig[currentCampus];
    let researchCount = 0;
    let needsIdCount = 0;

    allObservations.forEach((obs) => {
      if (obs.quality_grade === "research") researchCount++;
      if (obs.quality_grade === "needs_id") needsIdCount++;
    });

    const total = researchCount + needsIdCount;
    const researchPercent = total ? ((researchCount / total) * 100).toFixed(1) : "0.0";
    const needsIdPercent = total ? ((needsIdCount / total) * 100).toFixed(1) : "0.0";
    const monthLabel = now.toLocaleString(currentLanguage === "en" ? "en-CA" : "fr-CA", { month: "long", year: "numeric" });
    const selectedObservationTypes = Array.from(document.querySelectorAll(".obsType:checked"))
      .map((cb) => cb.value)
      .map((name) => translateGroupName(name))
      .join(", ") || t("no_classes");

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
    h1, h2 { color: #0d3d7d; }
    .meta, .card { margin-bottom: 24px; }
    .card { padding: 18px; border: 1px solid #d5e1f3; border-radius: 14px; background: #f8fbff; }
    table { width: 100%; border-collapse: collapse; }
    th, td { padding: 10px; border-bottom: 1px solid #d5e1f3; text-align: left; }
    th { color: #0d3d7d; }
  </style>
</head>
<body>
  <h1>${escapeHtml(t("report_title"))}</h1>
  <div class="meta">
    <strong>${escapeHtml(t("month"))} :</strong> ${escapeHtml(monthLabel)}<br>
    <strong>${escapeHtml(t("campus"))} :</strong> ${escapeHtml(campus)}<br>
    <strong>${escapeHtml(t("classification_year"))} :</strong> ${escapeHtml(currentYear)}<br>
    <strong>${escapeHtml(t("observation_types"))} :</strong> ${escapeHtml(selectedObservationTypes)}<br>
    <strong>${escapeHtml(t("observation_srs"))} :</strong> EPSG:4326<br>
    <strong>${escapeHtml(t("classification_srs"))} :</strong> EPSG:4326
  </div>

  <div class="card">
    <h2>${escapeHtml(t("observation_stats"))}</h2>
    <p><strong>${escapeHtml(t("realtime_observations"))} :</strong> ${escapeHtml(allObservations.length)}</p>
    <p><strong>${escapeHtml(t("research_label"))} :</strong> ${escapeHtml(researchCount)} (${escapeHtml(researchPercent)}%)</p>
    <p><strong>${escapeHtml(t("needs_id_label"))} :</strong> ${escapeHtml(needsIdCount)} (${escapeHtml(needsIdPercent)}%)</p>
  </div>

  <div class="card">
    <h2>${escapeHtml(t("habitat_classification"))}</h2>
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
    <p>${escapeHtml(t("report_summary_text", { campus, year: String(currentYear) }))}</p>
  </div>
</body>
</html>`;

    const filename = `rapport_mensuel_biodiversite_inrs_${currentCampus}_${now.getFullYear()}_${String(now.getMonth() + 1).padStart(2, "0")}.html`;
    downloadTextFile(filename, report, "text/html;charset=utf-8");
  };

  function startObservationsAutoRefresh() {
      if (observationsRefreshInterval) {
        clearInterval(observationsRefreshInterval);
      }

      observationsRefreshInterval = setInterval(() => {
      loadAllProjectObservations(true).catch((error) => console.error(error));
      }, 24 * 60 * 60 * 1000);
    }

  window.loadCampus = async function (name) {
    currentCampus = name;
    const campus = campusConfig[name];
    const boundaryCollection = await getCampusBoundary(name);
    fitCampusBoundary(boundaryCollection, campus.center, campus.zoom);

    if (campusMarker) {
      map.removeLayer(campusMarker);
      campusMarker = null;
    }

    if (secteurLayer && map.hasLayer(secteurLayer)) map.removeLayer(secteurLayer);
    secteurLayer = L.tileLayer.wms(geoserverBase + "/ne/wms", {
      layers: campus.secteur,
      format: "image/png",
      transparent: true
    }).addTo(map);
    overlayLayerRefs.footprint = secteurLayer;

    await setupYearSlider();
    document.querySelectorAll(".obsType").forEach((cb) => {
      cb.checked = true;
    });
    await loadAllProjectObservations(true);
    startObservationsAutoRefresh();
    refreshLayerControl();
  };

  map.on("click", function (e) {
    if (!classifLayer) return;
    const point = map.latLngToContainerPoint(e.latlng, map.getZoom());
    const size = map.getSize();
    const url =
      geoserverBase +
      "/ne/wms?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetFeatureInfo" +
      "&LAYERS=ne:" + campusConfig[currentCampus].layerPrefix + currentYear +
      "&QUERY_LAYERS=ne:" + campusConfig[currentCampus].layerPrefix + currentYear +
      "&STYLES=&BBOX=" + map.getBounds().toBBoxString() +
      "&FEATURE_COUNT=1&HEIGHT=" + size.y + "&WIDTH=" + size.x +
      "&INFO_FORMAT=application/json&SRS=EPSG:4326&X=" + Math.floor(point.x) + "&Y=" + Math.floor(point.y);

    fetch(url)
      .then((res) => res.json())
      .then((data) => {
        if (!data.features || !data.features.length) return;
        const props = data.features[0].properties || {};
        L.popup().setLatLng(e.latlng).setContent(`
          <b>${escapeHtml(t("class_label"))} :</b> ${escapeHtml(translateClassName(props.nom_classe || props.class || t("unknown")))}<br>
          <b>${escapeHtml(t("area_label"))} :</b> ${escapeHtml(formatArea(props.surface_m2 || 0))}
        `).openOn(map);
      })
      .catch((error) => console.error(error));
  });

  window.goHome = function () {
    loadCampus(currentCampus);
  };

  window.closeClassChart = function () {
    chartModal.classList.remove("is-open");
  };

  window.openClassChart = async function () {
    chartModal.classList.add("is-open");
    chartModalSubtitle.innerText = t("chart_loading");
    destroyClassChart();

    try {
      const years = await getAvailableYears();
      const series = await Promise.all(
        years.map(async (year) => {
          const payload = await fetchClasses(year);
          return {
            year,
            classes: (payload.classes || []).map(normalizeClassEntry)
          };
        })
      );

      const classMap = new Map();
      series.forEach(({ classes }) => {
        classes.forEach((item) => {
          if (!classMap.has(item.id)) {
            classMap.set(item.id, { name: item.name, color: item.color });
          }
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
        data: {
          labels: years,
          datasets
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          plugins: {
            legend: {
              position: "bottom"
            },
            tooltip: {
              callbacks: {
                label: function (context) {
                  return context.dataset.label + ": " + context.parsed.y.toLocaleString(currentLanguage === "en" ? "en-CA" : "fr-CA") + " ha";
                }
              }
            }
          },
          scales: {
            y: {
              title: {
                display: true,
                text: t("area_axis")
              }
            },
            x: {
              title: {
                display: true,
                text: t("year_axis")
              }
            }
          }
        }
      });

      chartModalSubtitle.innerText = t("chart_subtitle", { campus: campusLabel(currentCampus) });
    } catch (error) {
      console.error(error);
      chartModalSubtitle.innerText = t("chart_error");
    }
  };

  chartModal.addEventListener("click", function (event) {
    if (event.target === chartModal) {
      closeClassChart();
    }
  });

  if (languageSelect) {
    languageSelect.value = currentLanguage;
    languageSelect.addEventListener("change", function () {
      currentLanguage = this.value || "fr";
      localStorage.setItem("inrsLanguage", currentLanguage);
      applyTranslations();
    });
  }

  loadCampus("qc");
  applyTranslations();
});
