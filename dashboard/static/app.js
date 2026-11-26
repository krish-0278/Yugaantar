(() => {
  "use strict";

  const PAPER = "#f4f1ea";
  const INK = "#1d1d1b";
  const MUTED = "#6b675d";
  const RULE = "#cfc9bb";
  const ACCENT = "#0f5c5c";
  const SHOCK = "#c8402a";

  const state = {
    meta: null,
    channels: [
      { key: "t2m", label: "2m Temperature", short: "t2m", unit: "K" },
      { key: "tp", label: "Total Precipitation", short: "tp", unit: "mm/day" },
      { key: "u10", label: "10m Zonal Wind (u10)", short: "u10", unit: "m/s" },
      { key: "v10", label: "10m Meridional Wind (v10)", short: "v10", unit: "m/s" },
      { key: "msl", label: "Mean Sea-Level Pressure", short: "msl", unit: "Pa" },
      { key: "tcwv", label: "Total Column Water Vapour", short: "tcwv", unit: "kg/m^2" }
    ],
    initSource: "climatology",
    initStates: null,
    doy: 166,
    steps: 10,
    preset: "warm_bay_of_bengal",
    falloff: "smooth",
    drawShape: "circle",
    hoverGrid: null,
    dragDraw: null,
    shockStack: [],
    chIdx: 0,
    dayIdx: 0,
    playing: false,
    playSpeedMs: 350,
    playTimer: null,
    chartScope: "both",
    chartHoverIdx: null,
    crosshair: null,
    toastTimer: null,
    runData: null,
    running: false,
    abortCtrl: null
  };

  const el = {
    modelNameChip: document.getElementById("modelNameChip"),
    healthDot: document.getElementById("healthDot"),
    healthText: document.getElementById("healthText"),
    timingReadout: document.getElementById("timingReadout"),
    runBtn: document.getElementById("runBtn"),
    tryWarmBayBtn: document.getElementById("tryWarmBayBtn"),
    emptyState: document.getElementById("emptyState"),
    resultsView: document.getElementById("resultsView"),
    progressBar: document.getElementById("progressBar"),
    initSourceBadge: document.getElementById("initSourceBadge"),
    shockHeaderBadge: document.getElementById("shockHeaderBadge"),
    historyTabBtn: document.getElementById("historyTabBtn"),
    initClimatologyPane: document.getElementById("initClimatologyPane"),
    initUploadPane: document.getElementById("initUploadPane"),
    initHistoryPane: document.getElementById("initHistoryPane"),
    doyInput: document.getElementById("doyInput"),
    uploadFileInput: document.getElementById("uploadFileInput"),
    uploadStatus: document.getElementById("uploadStatus"),
    historyIndexInput: document.getElementById("historyIndexInput"),
    loadHistoryBtn: document.getElementById("loadHistoryBtn"),
    historyStatus: document.getElementById("historyStatus"),
    stepsRange: document.getElementById("stepsRange"),
    stepsInput: document.getElementById("stepsInput"),
    presetDesc: document.getElementById("presetDesc"),
    customControls: document.getElementById("customControls"),
    customLat: document.getElementById("customLat"),
    customLon: document.getElementById("customLon"),
    customRadius: document.getElementById("customRadius"),
    customRadiusRange: document.getElementById("customRadiusRange"),
    drawReadout: document.getElementById("drawReadout"),
    customValidationError: document.getElementById("customValidationError"),
    undoShockBtn: document.getElementById("undoShockBtn"),
    clearCustomBtn: document.getElementById("clearCustomBtn"),
    effRanges: {
      t2m: document.getElementById("eff_range_t2m"),
      tp: document.getElementById("eff_range_tp"),
      u10: document.getElementById("eff_range_u10"),
      v10: document.getElementById("eff_range_v10"),
      msl: document.getElementById("eff_range_msl"),
      tcwv: document.getElementById("eff_range_tcwv")
    },
    effInputs: {
      t2m: document.getElementById("eff_t2m"),
      tp: document.getElementById("eff_tp"),
      u10: document.getElementById("eff_u10"),
      v10: document.getElementById("eff_v10"),
      msl: document.getElementById("eff_msl"),
      tcwv: document.getElementById("eff_tcwv")
    },
    pertSummary: document.getElementById("pertSummary"),
    errorBanner: document.getElementById("errorBanner"),
    loadingOverlay: document.getElementById("loadingOverlay"),
    loadingLabel: document.getElementById("loadingLabel"),
    errorToast: document.getElementById("errorToast"),
    mapsRow: document.getElementById("mapsRow"),
    cellReadout: document.getElementById("cellReadout"),
    playBtn: document.getElementById("playBtn"),
    playSpeedSelect: document.getElementById("playSpeedSelect"),
    prevDayBtn: document.getElementById("prevDayBtn"),
    nextDayBtn: document.getElementById("nextDayBtn"),
    daySlider: document.getElementById("daySlider"),
    dayTicks: document.getElementById("dayTicks"),
    dayReadout: document.getElementById("dayReadout"),
    primeCorner: document.getElementById("primeCorner"),
    multiCorner: document.getElementById("multiCorner"),
    diffCorner: document.getElementById("diffCorner"),
    canvasPrime: document.getElementById("canvasPrime"),
    canvasMulti: document.getElementById("canvasMulti"),
    canvasDiff: document.getElementById("canvasDiff"),
    overlayPrime: document.getElementById("overlayPrime"),
    overlayMulti: document.getElementById("overlayMulti"),
    overlayDiff: document.getElementById("overlayDiff"),
    cbValueCanvas: document.getElementById("cbValueCanvas"),
    cbDiffCanvas: document.getElementById("cbDiffCanvas"),
    cbValMin: document.getElementById("cbValMin"),
    cbValMid: document.getElementById("cbValMid"),
    cbValMax: document.getElementById("cbValMax"),
    cbDiffMin: document.getElementById("cbDiffMin"),
    cbDiffMax: document.getElementById("cbDiffMax"),
    nullCaption: document.getElementById("nullCaption"),
    chartTitle: document.getElementById("chartTitle"),
    chartHoverReadout: document.getElementById("chartHoverReadout"),
    tsChart: document.getElementById("tsChart"),
    impactTitle: document.getElementById("impactTitle"),
    impactTbody: document.getElementById("impactTbody"),
    mapTooltip: document.getElementById("mapTooltip")
  };

  function showError(msg) {
    if (state.toastTimer) {
      clearTimeout(state.toastTimer);
      state.toastTimer = null;
    }
    if (!msg) {
      el.errorBanner.textContent = "";
      el.errorBanner.classList.add("hidden");
      if (el.errorToast) el.errorToast.classList.add("hidden");
    } else {
      el.errorBanner.textContent = msg;
      el.errorBanner.classList.remove("hidden");
      if (el.errorToast) {
        el.errorToast.textContent = msg;
        el.errorToast.classList.remove("hidden");
        state.toastTimer = setTimeout(() => {
          el.errorToast.classList.add("hidden");
        }, 5000);
      }
    }
  }

  function clamp(v, lo, hi) {
    return Math.max(lo, Math.min(hi, v));
  }

  function lerpColor(c1, c2, t) {
    const k = clamp(t, 0, 1);
    return [
      Math.round(c1[0] + (c2[0] - c1[0]) * k),
      Math.round(c1[1] + (c2[1] - c1[1]) * k),
      Math.round(c1[2] + (c2[2] - c1[2]) * k)
    ];
  }

  function sampleStops(stops, t) {
    const k = clamp(t, 0, 1);
    const n = stops.length - 1;
    const pos = k * n;
    const idx = Math.min(n - 1, Math.floor(pos));
    return lerpColor(stops[idx], stops[idx + 1], pos - idx);
  }

  const LUTS = {
    warm: [
      [246, 241, 229],
      [234, 198, 130],
      [210, 118, 56],
      [135, 36, 22]
    ],
    rain: [
      [246, 243, 236],
      [164, 212, 206],
      [42, 136, 138],
      [10, 62, 68]
    ],
    pressure: [
      [236, 239, 238],
      [165, 184, 184],
      [78, 108, 114],
      [26, 44, 52]
    ],
    diverging: [
      [33, 94, 138],
      [148, 188, 210],
      [244, 241, 234],
      [226, 146, 112],
      [190, 48, 28]
    ]
  };

  function lutForChannel(chKey) {
    if (chKey === "t2m") return LUTS.warm;
    if (chKey === "tp" || chKey === "tcwv") return LUTS.rain;
    if (chKey === "u10" || chKey === "v10") return LUTS.diverging;
    return LUTS.pressure;
  }

  function formatNum(v, decimals = 2) {
    if (!Number.isFinite(v)) return "—";
    if (Math.abs(v) >= 1000) return v.toFixed(0);
    if (Math.abs(v) >= 100) return v.toFixed(1);
    return v.toFixed(decimals);
  }

  function rolloutDoy(baseDoy, stepIdx) {
    return ((baseDoy - 1 + (stepIdx + 1)) % 365) + 1;
  }

  function renderColorbar(canvas, stops) {
    const ctx = canvas.getContext("2d");
    const w = canvas.width;
    const h = canvas.height;
    const img = ctx.createImageData(w, h);
    for (let x = 0; x < w; x++) {
      const rgb = sampleStops(stops, x / (w - 1));
      for (let y = 0; y < h; y++) {
        const p = (y * w + x) * 4;
        img.data[p] = rgb[0];
        img.data[p + 1] = rgb[1];
        img.data[p + 2] = rgb[2];
        img.data[p + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  function drawGridCanvas(canvas, field, vmin, vmax, stops) {
    const H = 32;
    const W = 36;
    if (canvas.width !== W || canvas.height !== H) {
      canvas.width = W;
      canvas.height = H;
    }
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(W, H);
    const span = vmax - vmin;

    for (let i = 0; i < H; i++) {
      // North at top: row i=31 maps to y=0, row i=0 maps to y=31
      const canvasY = H - 1 - i;
      const row = field[i];
      for (let j = 0; j < W; j++) {
        const val = row[j];
        const t = span > 1e-12 ? (val - vmin) / span : 0.5;
        const rgb = sampleStops(stops, t);
        const p = (canvasY * W + j) * 4;
        img.data[p] = rgb[0];
        img.data[p + 1] = rgb[1];
        img.data[p + 2] = rgb[2];
        img.data[p + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }

  function buildContourPath(mask) {
    const H = mask.length;
    const W = mask[0].length;
    const segs = [];
    for (let i = 0; i < H; i++) {
      const yTop = H - 1 - i;
      const yBot = H - i;
      for (let j = 0; j < W; j++) {
        const v = Boolean(mask[i][j]);
        const vNorth = i + 1 < H ? Boolean(mask[i + 1][j]) : false;
        const vSouth = i - 1 >= 0 ? Boolean(mask[i - 1][j]) : false;
        const vEast = j + 1 < W ? Boolean(mask[i][j + 1]) : false;
        const vWest = j - 1 >= 0 ? Boolean(mask[i][j - 1]) : false;

        if (v !== vNorth && i + 1 < H) segs.push(`M${j} ${yTop}H${j + 1}`);
        if (v !== vSouth && i === 0) { /* boundary omitted cleanly */ }
        if (v !== vEast && j + 1 < W) segs.push(`M${j + 1} ${yTop}V${yBot}`);
        if (v && i === 0) segs.push(`M${j} ${yBot}H${j + 1}`);
        if (v && i === H - 1) segs.push(`M${j} ${yTop}H${j + 1}`);
        if (v && j === 0) segs.push(`M0 ${yTop}V${yBot}`);
        if (v && j === W - 1) segs.push(`M${W} ${yTop}V${yBot}`);
        void vWest;
      }
    }
    return segs.join("");
  }

  let cachedCoastPath = null;

  function renderMapOverlays() {
    if (!state.meta) return;
    if (!cachedCoastPath) {
      const land = state.meta.grid.land_sea_mask.map((row) => row.map((v) => v >= 0.5));
      cachedCoastPath = buildContourPath(land);
    }

    let shockPath = "";
    let customMarker = "";
    let ghostMarker = "";

    if (state.preset === "custom") {
      const clat = parseFloat(el.customLat.value);
      const clon = parseFloat(el.customLon.value);
      const rad = parseFloat(el.customRadius.value);
      if (Number.isFinite(clat) && Number.isFinite(clon) && Number.isFinite(rad)) {
        const cx = clon - 65.5 + 0.5;
        const cy = 32 - (clat - 6.5 + 0.5);
        if (state.drawShape === "rect") {
          const side = Math.max(1, rad * 2);
          customMarker =
            `<rect x="${(cx - rad).toFixed(2)}" y="${(cy - rad).toFixed(2)}" width="${side.toFixed(2)}" height="${side.toFixed(2)}" fill="none" stroke="${SHOCK}" stroke-width="0.28" stroke-dasharray="0.6 0.4"/>` +
            `<path d="M${(cx - 0.6).toFixed(2)} ${cy.toFixed(2)}H${(cx + 0.6).toFixed(2)}M${cx.toFixed(2)} ${(cy - 0.6).toFixed(2)}V${(cy + 0.6).toFixed(2)}" stroke="${SHOCK}" stroke-width="0.28"/>`;
        } else {
          customMarker =
            `<circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="${rad.toFixed(2)}" fill="none" stroke="${SHOCK}" stroke-width="0.28" stroke-dasharray="0.6 0.4"/>` +
            `<path d="M${(cx - 0.6).toFixed(2)} ${cy.toFixed(2)}H${(cx + 0.6).toFixed(2)}M${cx.toFixed(2)} ${(cy - 0.6).toFixed(2)}V${(cy + 0.6).toFixed(2)}" stroke="${SHOCK}" stroke-width="0.28"/>`;
        }
      }

      if (state.dragDraw) {
        const { startLat, startLon, curLat, curLon } = state.dragDraw;
        if (state.drawShape === "rect") {
          const x0 = Math.min(startLon, curLon) - 65.5 + 0.5;
          const x1 = Math.max(startLon, curLon) - 65.5 + 0.5;
          const y0 = 32 - (Math.max(startLat, curLat) - 6.5 + 0.5);
          const y1 = 32 - (Math.min(startLat, curLat) - 6.5 + 0.5);
          ghostMarker = `<rect x="${x0.toFixed(2)}" y="${y0.toFixed(2)}" width="${Math.max(0.5, x1 - x0).toFixed(2)}" height="${Math.max(0.5, y1 - y0).toFixed(2)}" fill="${SHOCK}" fill-opacity="0.14" stroke="${SHOCK}" stroke-width="0.32"/>`;
        } else {
          const cx = startLon - 65.5 + 0.5;
          const cy = 32 - (startLat - 6.5 + 0.5);
          const r = clamp(Math.hypot(curLat - startLat, curLon - startLon), 0.5, getLimits().max_radius_cells);
          ghostMarker = `<circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="${r.toFixed(2)}" fill="${SHOCK}" fill-opacity="0.14" stroke="${SHOCK}" stroke-width="0.32"/>`;
        }
      } else if (state.hoverGrid && Number.isFinite(rad)) {
        const cx = state.hoverGrid.lon - 65.5 + 0.5;
        const cy = 32 - (state.hoverGrid.lat - 6.5 + 0.5);
        if (state.drawShape === "rect") {
          const side = Math.max(1, rad * 2);
          ghostMarker = `<rect x="${(cx - rad).toFixed(2)}" y="${(cy - rad).toFixed(2)}" width="${side.toFixed(2)}" height="${side.toFixed(2)}" fill="none" stroke="${ACCENT}" stroke-width="0.24" stroke-dasharray="0.4 0.4" stroke-opacity="0.75"/>`;
        } else {
          ghostMarker = `<circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="${rad.toFixed(2)}" fill="none" stroke="${ACCENT}" stroke-width="0.24" stroke-dasharray="0.4 0.4" stroke-opacity="0.75"/>`;
        }
      }
    } else if (state.runData && state.runData.region_mask && state.runData.preset === state.preset) {
      const p = buildContourPath(state.runData.region_mask);
      if (p) {
        shockPath = `<path d="${p}" fill="none" stroke="${SHOCK}" stroke-width="0.30" stroke-dasharray="0.7 0.45"/>`;
      }
    }

    let crosshairSvg = "";
    if (state.crosshair) {
      const { i, j } = state.crosshair;
      const yCell = 31 - i;
      crosshairSvg =
        `<rect x="${j}" y="${yCell}" width="1" height="1" fill="none" stroke="${INK}" stroke-width="0.24"/>` +
        `<line x1="${j + 0.5}" y1="0" x2="${j + 0.5}" y2="32" stroke="${INK}" stroke-width="0.12" stroke-opacity="0.45"/>` +
        `<line x1="0" y1="${yCell + 0.5}" x2="36" y2="${yCell + 0.5}" stroke="${INK}" stroke-width="0.12" stroke-opacity="0.45"/>`;
    }

    const svgInner =
      `<path d="${cachedCoastPath}" fill="none" stroke="${INK}" stroke-width="0.16" stroke-opacity="0.85"/>` +
      shockPath +
      customMarker +
      ghostMarker +
      crosshairSvg;

    el.overlayPrime.innerHTML = svgInner;
    el.overlayMulti.innerHTML = svgInner;
    el.overlayDiff.innerHTML = svgInner;
  }

  function renderPresetMiniMaps() {
    if (!cachedCoastPath) return;
    document.querySelectorAll("[data-mini-preset]").forEach((svg) => {
      const id = svg.dataset.miniPreset;
      let mark = "";
      if (id === "warm_bay_of_bengal") {
        // 8-20 N (i=2..14 -> yTop=32-15=17..29), 82-94 E (j=16.5..28.5)
        mark = `<rect x="16.5" y="17.5" width="12" height="12" fill="${SHOCK}" fill-opacity="0.22" stroke="${SHOCK}" stroke-width="0.9"/>`;
      } else if (id === "weaker_arabian_sea_wind") {
        // 8-20 N, 65-75 E (j=0..9.5)
        mark = `<rect x="0.5" y="17.5" width="9.5" height="12" fill="${SHOCK}" fill-opacity="0.22" stroke="${SHOCK}" stroke-width="0.9"/>`;
      } else if (id === "custom") {
        mark = `<circle cx="20" cy="21" r="4" fill="${SHOCK}" fill-opacity="0.18" stroke="${SHOCK}" stroke-width="0.9" stroke-dasharray="1.5 1"/>`;
      }
      svg.innerHTML =
        `<path d="${cachedCoastPath}" fill="none" stroke="${INK}" stroke-width="0.45" stroke-opacity="0.8"/>` +
        mark;
    });
  }

  function getLimits() {
    const lim = (state.meta && state.meta.limits) || {};
    return {
      min_steps: lim.min_steps ?? 1,
      max_steps: lim.max_steps ?? 30,
      min_doy: lim.min_doy ?? 1,
      max_doy: lim.max_doy ?? 365,
      min_radius_cells: lim.min_radius_cells ?? 0.5,
      max_radius_cells: lim.max_radius_cells ?? 12.0
    };
  }

  function snapshotShock() {
    const eff = {};
    for (const [k, inp] of Object.entries(el.effInputs)) {
      eff[k] = inp.value;
    }
    return {
      preset: state.preset,
      lat: el.customLat.value,
      lon: el.customLon.value,
      radius: el.customRadius.value,
      falloff: state.falloff,
      drawShape: state.drawShape,
      eff
    };
  }

  function pushShockSnapshot() {
    state.shockStack.push(snapshotShock());
    if (state.shockStack.length > 20) state.shockStack.shift();
    if (el.undoShockBtn) el.undoShockBtn.disabled = state.shockStack.length === 0;
  }

  function restoreShockSnapshot(snap) {
    if (!snap) return;
    el.customLat.value = snap.lat;
    el.customLon.value = snap.lon;
    el.customRadius.value = snap.radius;
    if (el.customRadiusRange) el.customRadiusRange.value = snap.radius;
    state.falloff = snap.falloff;
    state.drawShape = snap.drawShape || "circle";
    document.querySelectorAll("[data-falloff]").forEach((b) => b.classList.toggle("active", b.dataset.falloff === state.falloff));
    document.querySelectorAll("[data-draw-shape]").forEach((b) => b.classList.toggle("active", b.dataset.drawShape === state.drawShape));
    for (const [k, v] of Object.entries(snap.eff)) {
      if (el.effInputs[k]) el.effInputs[k].value = v;
      if (el.effRanges[k]) el.effRanges[k].value = v;
    }
    selectPreset(snap.preset);
    validateCustomShock();
    updateDrawReadout();
  }

  function updateDrawReadout(customLat, customLon, customRad) {
    if (!el.drawReadout) return;
    const lat = Number.isFinite(customLat) ? customLat : parseFloat(el.customLat.value);
    const lon = Number.isFinite(customLon) ? customLon : parseFloat(el.customLon.value);
    const rad = Number.isFinite(customRad) ? customRad : parseFloat(el.customRadius.value);
    if (!Number.isFinite(lat) || !Number.isFinite(lon) || !Number.isFinite(rad)) return;
    const approxCells = Math.max(1, Math.round(Math.PI * rad * rad));
    el.drawReadout.textContent = `${lat.toFixed(1)}°N ${lon.toFixed(1)}°E · r=${rad.toFixed(1)} cells (~${approxCells} cells)`;
  }

  function validateCustomShock() {
    if (!el.customValidationError) return true;
    if (state.preset !== "custom") {
      el.customValidationError.classList.add("hidden");
      el.customValidationError.textContent = "";
      return true;
    }
    const lim = getLimits();
    const lat = parseFloat(el.customLat.value);
    const lon = parseFloat(el.customLon.value);
    const rad = parseFloat(el.customRadius.value);
    let errMsg = "";
    if (!Number.isFinite(lat) || lat < 6.5 || lat > 37.5) {
      errMsg = "Center latitude must be between 6.5 and 37.5 °N.";
    } else if (!Number.isFinite(lon) || lon < 65.5 || lon > 100.5) {
      errMsg = "Center longitude must be between 65.5 and 100.5 °E.";
    } else if (!Number.isFinite(rad) || rad < lim.min_radius_cells || rad > lim.max_radius_cells) {
      errMsg = `Radius must be between ${lim.min_radius_cells} and ${lim.max_radius_cells} cells.`;
    } else {
      for (const ch of state.channels) {
        const inp = el.effInputs[ch.key];
        if (!inp) continue;
        const v = parseFloat(inp.value);
        if (!Number.isFinite(v)) {
          errMsg = `${ch.short} magnitude must be a number.`;
          break;
        }
        if (ch.std) {
          const maxMag = 4.0 * ch.std;
          if (Math.abs(v) > maxMag) {
            errMsg = `${ch.short} delta must be within ±${formatNum(maxMag, 2)} ${ch.unit} (4σ limit).`;
            break;
          }
        }
      }
    }
    if (errMsg) {
      el.customValidationError.textContent = errMsg;
      el.customValidationError.classList.remove("hidden");
      return false;
    }
    el.customValidationError.classList.add("hidden");
    el.customValidationError.textContent = "";
    return true;
  }

  function syncCustomChannelBounds() {
    if (!state.meta || !state.meta.channels) return;
    for (const ch of state.meta.channels) {
      const range = el.effRanges[ch.key];
      const num = el.effInputs[ch.key];
      if (!range || !num || !ch.std) continue;
      const maxMag = Number((4.0 * ch.std).toFixed(2));
      range.min = String(-maxMag);
      range.max = String(maxMag);
      num.min = String(-maxMag);
      num.max = String(maxMag);
    }
  }

  function selectPreset(presetId) {
    state.preset = presetId;
    document.querySelectorAll("input[name='preset']").forEach((r) => {
      r.checked = r.value === presetId;
    });
    document.querySelectorAll("[data-preset-card]").forEach((card) => {
      card.classList.toggle("selected", card.dataset.presetCard === presetId);
    });
    if (el.shockHeaderBadge) el.shockHeaderBadge.textContent = presetId;
    el.customControls.classList.toggle("hidden", presetId !== "custom");
    document.querySelectorAll(".map-stage").forEach((s) => s.classList.toggle("pick-mode", presetId === "custom"));
    if (state.meta) {
      const found = state.meta.presets.find((p) => p.id === presetId);
      if (found) el.presetDesc.textContent = found.summary;
    }
    renderMapOverlays();
  }

  function renderMapsAndColorbars() {
    if (!state.runData) return;
    const ch = state.channels[state.chIdx];
    const k = clamp(state.dayIdx, 0, state.runData.steps - 1);
    state.dayIdx = k;

    const pGrid = state.runData.control[k][state.chIdx];
    const mGrid = state.runData.shocked[k][state.chIdx];
    const dGrid = state.runData.diff[k][state.chIdx];

    let vmin = Infinity;
    let vmax = -Infinity;
    let dAbs = 0;

    for (let s = 0; s < state.runData.steps; s++) {
      const pf = state.runData.control[s][state.chIdx];
      const mf = state.runData.shocked[s][state.chIdx];
      const df = state.runData.diff[s][state.chIdx];
      for (let i = 0; i < 32; i++) {
        for (let j = 0; j < 36; j++) {
          const pv = pf[i][j];
          const mv = mf[i][j];
          const dv = Math.abs(df[i][j]);
          if (pv < vmin) vmin = pv;
          if (pv > vmax) vmax = pv;
          if (mv < vmin) vmin = mv;
          if (mv > vmax) vmax = mv;
          if (dv > dAbs) dAbs = dv;
        }
      }
    }

    if (ch.key === "u10" || ch.key === "v10") {
      const sym = Math.max(Math.abs(vmin), Math.abs(vmax), 1e-6);
      vmin = -sym;
      vmax = sym;
    } else if (Math.abs(vmax - vmin) < 1e-6) {
      vmin -= 1;
      vmax += 1;
    }

    const diffSpan = Math.max(dAbs, 1e-6);
    const valStops = lutForChannel(ch.key);

    drawGridCanvas(el.canvasPrime, pGrid, vmin, vmax, valStops);
    drawGridCanvas(el.canvasMulti, mGrid, vmin, vmax, valStops);
    drawGridCanvas(el.canvasDiff, dGrid, -diffSpan, diffSpan, LUTS.diverging);

    renderColorbar(el.cbValueCanvas, valStops);
    renderColorbar(el.cbDiffCanvas, LUTS.diverging);

    el.cbValMin.textContent = formatNum(vmin, 1);
    el.cbValMid.textContent = formatNum((vmin + vmax) * 0.5, 1);
    el.cbValMax.textContent = formatNum(vmax, 1);

    if (state.runData.max_abs_diff === 0) {
      el.cbDiffMin.textContent = "0";
      el.cbDiffMax.textContent = "0";
      el.nullCaption.classList.remove("hidden");
      el.nullCaption.textContent = `difference = 0 exactly (max |Δ| = ${state.runData.max_abs_diff.toFixed(1)})`;
      el.diffCorner.textContent = `max |Δ| = ${state.runData.max_abs_diff.toFixed(1)}`;
    } else {
      el.cbDiffMin.textContent = `-${formatNum(dAbs, 2)}`;
      el.cbDiffMax.textContent = `+${formatNum(dAbs, 2)}`;
      el.nullCaption.classList.add("hidden");
      el.diffCorner.textContent = `day ${k + 1} · max |Δ| = ${formatNum(dAbs, 2)}`;
    }

    const curDoy = rolloutDoy(state.runData.doy, k);
    el.daySlider.max = String(state.runData.steps - 1);
    el.daySlider.value = String(k);
    if (el.dayTicks && el.dayTicks.childElementCount !== state.runData.steps) {
      let opts = "";
      for (let s = 0; s < state.runData.steps; s++) {
        opts += `<option value="${s}"></option>`;
      }
      el.dayTicks.innerHTML = opts;
    }
    el.dayReadout.textContent = `Day ${k + 1} / ${state.runData.steps} (DOY ${curDoy})`;
    el.primeCorner.textContent = `day ${k + 1} (doy ${curDoy})`;
    el.multiCorner.textContent = `day ${k + 1} (doy ${curDoy})`;

    renderMapOverlays();
  }

  function triggerCrossfade() {
    const canvases = [el.canvasPrime, el.canvasMulti, el.canvasDiff];
    canvases.forEach((c) => c && c.classList.add("crossfade"));
    requestAnimationFrame(() => {
      requestAnimationFrame(() => {
        canvases.forEach((c) => c && c.classList.remove("crossfade"));
      });
    });
  }

  function renderTimeseries() {
    const svg = el.tsChart;
    if (!state.runData || !state.runData.timeseries) {
      svg.innerHTML = `<text x="270" y="85" text-anchor="middle" fill="${MUTED}" font-size="12">Run an experiment to plot domain means.</text>`;
      return;
    }

    const ch = state.channels[state.chIdx];
    const steps = state.runData.steps;
    const allP = state.runData.timeseries.all[ch.key].prime;
    const allM = state.runData.timeseries.all[ch.key].multiverse;
    const landP = state.runData.timeseries.land[ch.key].prime;
    const landM = state.runData.timeseries.land[ch.key].multiverse;

    el.chartTitle.textContent = `Domain Mean: ${ch.short} (${ch.unit})`;

    const showAll = state.chartScope === "both" || state.chartScope === "all";
    const showLand = state.chartScope === "both" || state.chartScope === "land";
    let vals = [];
    if (showAll) vals = vals.concat(allP, allM);
    if (showLand) vals = vals.concat(landP, landM);
    let yMin = Math.min(...vals);
    let yMax = Math.max(...vals);
    const pad = Math.max((yMax - yMin) * 0.12, 1e-4);
    yMin -= pad;
    yMax += pad;

    const W = 540;
    const H = 165;
    const mL = 48;
    const mR = 68;
    const mT = 12;
    const mB = 22;
    const plotW = W - mL - mR;
    const plotH = H - mT - mB;

    const xOf = (idx) => (steps <= 1 ? mL + plotW * 0.5 : mL + (idx / (steps - 1)) * plotW);
    const yOf = (v) => mT + ((yMax - v) / (yMax - yMin)) * plotH;

    const toPath = (arr) =>
      arr
        .map((v, i) => `${i === 0 ? "M" : "L"}${xOf(i).toFixed(1)} ${yOf(v).toFixed(1)}`)
        .join(" ");

    let out = "";
    for (let t = 0; t <= 2; t++) {
      const v = yMin + (t / 2) * (yMax - yMin);
      const y = yOf(v);
      out += `<line x1="${mL}" y1="${y.toFixed(1)}" x2="${W - mR}" y2="${y.toFixed(1)}" stroke="${RULE}" stroke-width="0.8"/>`;
      out += `<text x="${mL - 5}" y="${(y + 3.5).toFixed(1)}" text-anchor="end" fill="${MUTED}" font-size="10" font-family="ui-monospace, monospace">${formatNum(v, 2)}</text>`;
    }

    const tickDays = steps <= 6 ? Array.from({ length: steps }, (_, i) => i) : [0, Math.floor((steps - 1) / 2), steps - 1];
    tickDays.forEach((idx) => {
      const x = xOf(idx);
      out += `<text x="${x.toFixed(1)}" y="${H - 6}" text-anchor="middle" fill="${MUTED}" font-size="10" font-family="ui-monospace, monospace">d${idx + 1}</text>`;
    });

    const curX = xOf(state.dayIdx);
    out += `<line x1="${curX.toFixed(1)}" y1="${mT}" x2="${curX.toFixed(1)}" y2="${H - mB}" stroke="${ACCENT}" stroke-width="1.4"/>`;

    if (state.chartHoverIdx !== null && state.chartHoverIdx >= 0 && state.chartHoverIdx < steps) {
      const hx = xOf(state.chartHoverIdx);
      out += `<line x1="${hx.toFixed(1)}" y1="${mT}" x2="${hx.toFixed(1)}" y2="${H - mB}" stroke="${MUTED}" stroke-width="1.0" stroke-dasharray="2 2"/>`;
    }

    if (showLand) {
      const dash = state.chartScope === "land" ? "" : ` stroke-dasharray="3.5 2.5" stroke-opacity="0.75"`;
      out += `<path d="${toPath(landP)}" fill="none" stroke="${INK}" stroke-width="1.5"${dash}/>`;
      out += `<path d="${toPath(landM)}" fill="none" stroke="${SHOCK}" stroke-width="1.5"${dash}/>`;
    }
    if (showAll) {
      out += `<path d="${toPath(allP)}" fill="none" stroke="${INK}" stroke-width="2.0"/>`;
      out += `<path d="${toPath(allM)}" fill="none" stroke="${SHOCK}" stroke-width="1.6"/>`;
    }

    const primaryP = showAll ? allP : landP;
    const primaryM = showAll ? allM : landM;
    const lastIdx = steps - 1;
    const endX = xOf(lastIdx) + 5;
    let yEndP = yOf(primaryP[lastIdx]);
    let yEndM = yOf(primaryM[lastIdx]);
    if (Math.abs(yEndP - yEndM) < 12) {
      if (yEndM >= yEndP) {
        yEndM = yEndP + 12;
        if (yEndM > H - mB - 2) {
          yEndM = H - mB - 2;
          yEndP = yEndM - 12;
        }
      } else {
        yEndM = yEndP - 12;
        if (yEndM < mT + 8) {
          yEndM = mT + 8;
          yEndP = yEndM + 12;
        }
      }
    }
    out += `<text x="${endX.toFixed(1)}" y="${(yEndP + 3).toFixed(1)}" fill="${INK}" font-size="10">Prime</text>`;
    out += `<text x="${endX.toFixed(1)}" y="${(yEndM + 3).toFixed(1)}" fill="${SHOCK}" font-size="10">Multiverse</text>`;

    out += `<circle cx="${curX.toFixed(1)}" cy="${yOf(primaryP[state.dayIdx]).toFixed(1)}" r="2.8" fill="${INK}"/>`;
    out += `<circle cx="${curX.toFixed(1)}" cy="${yOf(primaryM[state.dayIdx]).toFixed(1)}" r="2.5" fill="${SHOCK}"/>`;

    svg.innerHTML = out;
  }

  function renderImpacts() {
    if (!state.runData || !state.runData.impact) return;
    const imp = state.runData.impact;
    const p = imp.prime;
    const m = imp.multiverse;
    const d = imp.delta;
    el.impactTitle.textContent = `Physical Proxy Impacts (over ${imp.steps} days)`;

    const rows = [
      {
        label: "Heatwave (≥ +3 K, ≥2 d)",
        hint: "Land cell-days with t2m - clim_t2m >= 3.0 K for >= 2 consecutive days.",
        pv: formatNum(p.heatwave_cell_days, 1),
        mv: formatNum(m.heatwave_cell_days, 1),
        dv: d.heatwave_cell_days.abs,
        fmtDelta: (v) => `${v >= 0 ? "+" : ""}${formatNum(v, 1)} cell-d`
      },
      {
        label: "Heavy rain ≥ 64.5 mm/d",
        hint: "Share of land cell-days with tp >= 64.5 mm/day.",
        pv: `${(p.heavy_rain_64_share * 100).toFixed(2)}%`,
        mv: `${(m.heavy_rain_64_share * 100).toFixed(2)}%`,
        dv: d.heavy_rain_64_share.abs * 100,
        fmtDelta: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)} pp`
      },
      {
        label: "Very heavy ≥ 115.6 mm/d",
        hint: "Share of land cell-days with tp >= 115.6 mm/day.",
        pv: `${(p.heavy_rain_115_share * 100).toFixed(2)}%`,
        mv: `${(m.heavy_rain_115_share * 100).toFixed(2)}%`,
        dv: d.heavy_rain_115_share.abs * 100,
        fmtDelta: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)} pp`
      },
      {
        label: "Rain deficit (> 20% below clim)",
        hint: "Share of land cells with cumulative rollout tp > 20% below cumulative climatology.",
        pv: `${(p.rain_deficit_share * 100).toFixed(1)}%`,
        mv: `${(m.rain_deficit_share * 100).toFixed(1)}%`,
        dv: d.rain_deficit_share.abs * 100,
        fmtDelta: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)} pp`
      },
      {
        label: "Crop stress CSI (weighted)",
        hint: "Land area-weighted mean of clip(0.6*waterDeficit + 0.4*heatStress, 0, 1).",
        pv: p.crop_stress_csi.toFixed(3),
        mv: m.crop_stress_csi.toFixed(3),
        dv: d.crop_stress_csi.abs,
        fmtDelta: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(3)}`
      }
    ];

    el.impactTbody.innerHTML = rows
      .map((r) => {
        const eps = 1e-9;
        let marker = "[=]";
        let cls = "muted-text";
        if (r.dv > eps) {
          marker = "[+]";
          cls = "delta-up";
        } else if (r.dv < -eps) {
          marker = "[-]";
          cls = "delta-down";
        }
        return `<tr>
          <td><span class="impact-row-hint" title="${r.hint}">${r.label}</span></td>
          <td class="num">${r.pv}</td>
          <td class="num">${r.mv}</td>
          <td class="num ${cls}">${marker} ${r.fmtDelta(r.dv)}</td>
        </tr>`;
      })
      .join("");
  }

  function renderPertSummary() {
    if (!state.runData || !state.runData.perturbation_summary) return;
    const sum = state.runData.perturbation_summary;
    if (sum.n_cells === 0) {
      el.pertSummary.textContent = "0 perturbed cells (all channels = 0).";
      return;
    }
    const parts = [];
    for (const ch of state.channels) {
      const item = sum.channels[ch.key];
      if (item && (Math.abs(item.max) > 1e-6 || Math.abs(item.min) > 1e-6)) {
        parts.push(`${ch.key}: mean ${formatNum(item.mean, 2)} ${ch.unit} (${item.max_abs_std.toFixed(2)}σ max)`);
      }
    }
    el.pertSummary.textContent = `${sum.n_cells} cells perturbed — ${parts.join("; ") || "zero magnitude"}.`;
  }

  function buildShockPayload() {
    if (state.preset !== "custom") {
      return { preset: state.preset };
    }
    const effects = {};
    for (const [key, input] of Object.entries(el.effInputs)) {
      const v = parseFloat(input.value);
      if (Number.isFinite(v) && Math.abs(v) > 0) {
        effects[key] = v;
      }
    }
    if (Object.keys(effects).length === 0) {
      effects.t2m = 0.0;
    }
    return {
      preset: "custom",
      center_lat: parseFloat(el.customLat.value),
      center_lon: parseFloat(el.customLon.value),
      radius_cells: parseFloat(el.customRadius.value),
      falloff: state.falloff,
      effects
    };
  }

  async function refreshHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (data.model && data.model.model_name && el.modelNameChip) {
        el.modelNameChip.textContent = data.model.model_name;
      }
      const ready = data.model_reachable && data.model && data.model.status === "ready";
      el.healthDot.classList.remove("ready", "warn");
      if (ready) {
        el.healthDot.classList.add("ready");
        el.healthText.textContent = "ready";
      } else if (data.model_reachable) {
        el.healthDot.classList.add("warn");
        el.healthText.textContent = "loading";
      } else {
        el.healthDot.classList.add("warn");
        el.healthText.textContent = "unreachable";
      }
      return ready;
    } catch {
      el.healthDot.classList.remove("ready");
      el.healthDot.classList.add("warn");
      el.healthText.textContent = "unreachable";
      return false;
    }
  }

  async function ensureInitStates() {
    if (state.initSource === "climatology") {
      const res = await fetch(`/api/init/climatology?doy=${state.doy}`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Failed to load climatology init");
      state.initStates = data.init_states;
      el.initSourceBadge.textContent = data.label;
    }
    if (!state.initStates) {
      throw new Error("Please select or upload an initial state first.");
    }
  }

  async function runExperiment() {
    if (!validateCustomShock()) {
      showError("Fix custom shock limits before running.");
      return;
    }
    if (state.running && state.abortCtrl) {
      state.abortCtrl.abort();
    }
    showError("");
    state.running = true;
    state.abortCtrl = new AbortController();
    const timeoutId = setTimeout(() => state.abortCtrl.abort(), 20000);

    el.runBtn.disabled = true;
    el.progressBar.classList.add("active");
    if (el.mapsRow) el.mapsRow.classList.add("is-loading");
    if (el.loadingOverlay) el.loadingOverlay.classList.remove("hidden");
    if (el.loadingLabel) el.loadingLabel.textContent = `Running ${state.steps}-day rollout…`;

    try {
      await ensureInitStates();
      const payload = {
        doy: state.doy,
        steps: state.steps,
        init_states: state.initStates,
        shock: buildShockPayload()
      };
      const res = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
        signal: state.abortCtrl.signal
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.error || `Request failed (${res.status})`);
      }

      state.runData = data;
      if (el.emptyState) el.emptyState.classList.add("hidden");
      if (el.resultsView) el.resultsView.classList.remove("hidden");
      state.dayIdx = clamp(state.dayIdx, 0, data.steps - 1);
      const inf = data.inference_ms || {};
      const ctrlText = Number.isFinite(inf.control) ? `${Math.round(inf.control)} ms` : "-";
      const shockText = Number.isFinite(inf.shocked) ? `${Math.round(inf.shocked)} ms` : "-";
      const wallText = Number.isFinite(data.wall_ms) ? `${(data.wall_ms / 1000).toFixed(1)} s` : "-";
      el.timingReadout.textContent = `control ${ctrlText} · shocked ${shockText} · wall ${wallText}`;
      renderPertSummary();
      renderMapsAndColorbars();
      renderTimeseries();
      renderImpacts();
      await refreshHealth();
    } catch (err) {
      if (err.name === "AbortError") {
        showError("Request timed out after 20 s.");
      } else {
        showError(err.message || "Experiment failed.");
      }
    } finally {
      clearTimeout(timeoutId);
      state.running = false;
      el.runBtn.disabled = false;
      el.progressBar.classList.remove("active");
      if (el.mapsRow) el.mapsRow.classList.remove("is-loading");
      if (el.loadingOverlay) el.loadingOverlay.classList.add("hidden");
    }
  }

  function stopPlayback() {
    state.playing = false;
    el.playBtn.textContent = "Play";
    if (state.playTimer) {
      clearInterval(state.playTimer);
      state.playTimer = null;
    }
  }

  function startPlayTimer() {
    if (state.playTimer) clearInterval(state.playTimer);
    state.playTimer = setInterval(() => {
      if (!state.runData) return stopPlayback();
      state.dayIdx = (state.dayIdx + 1) % state.runData.steps;
      triggerCrossfade();
      renderMapsAndColorbars();
      renderTimeseries();
    }, state.playSpeedMs);
  }

  function togglePlayback() {
    if (!state.runData) return;
    if (state.playing) {
      stopPlayback();
      return;
    }
    state.playing = true;
    el.playBtn.textContent = "Pause";
    startPlayTimer();
  }

  function stepDay(delta) {
    if (!state.runData) return;
    stopPlayback();
    state.dayIdx = clamp(state.dayIdx + delta, 0, state.runData.steps - 1);
    triggerCrossfade();
    renderMapsAndColorbars();
    renderTimeseries();
  }

  function bindEvents() {
    el.runBtn.addEventListener("click", runExperiment);
    if (el.tryWarmBayBtn) {
      el.tryWarmBayBtn.addEventListener("click", () => {
        selectPreset("warm_bay_of_bengal");
        runExperiment();
      });
    }

    document.querySelectorAll("[data-init-tab]").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.disabled) return;
        const tab = btn.dataset.initTab;
        state.initSource = tab;
        document.querySelectorAll("[data-init-tab]").forEach((b) => b.classList.toggle("active", b === btn));
        el.initClimatologyPane.classList.toggle("hidden", tab !== "climatology");
        el.initUploadPane.classList.toggle("hidden", tab !== "upload");
        el.initHistoryPane.classList.toggle("hidden", tab !== "history");
        if (tab === "climatology") {
          el.initSourceBadge.textContent = "smooth climatology";
        }
      });
    });

    el.doyInput.addEventListener("change", () => {
      const v = clamp(Math.round(parseFloat(el.doyInput.value) || 166), 1, 365);
      state.doy = v;
      el.doyInput.value = String(v);
    });

    document.querySelectorAll("[data-doy]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.doy = parseInt(btn.dataset.doy, 10);
        el.doyInput.value = String(state.doy);
      });
    });

    el.uploadFileInput.addEventListener("change", async () => {
      const file = el.uploadFileInput.files && el.uploadFileInput.files[0];
      if (!file) return;
      showError("");
      el.uploadStatus.textContent = `Uploading ${file.name}…`;
      try {
        const buf = await file.arrayBuffer();
        const isJson = file.name.toLowerCase().endsWith(".json");
        const res = await fetch("/api/init/upload", {
          method: "POST",
          headers: { "Content-Type": isJson ? "application/json" : "application/octet-stream" },
          body: buf
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "Upload failed");
        state.initSource = "upload";
        state.initStates = data.init_states;
        if (data.doy) {
          state.doy = data.doy;
          el.doyInput.value = String(data.doy);
        }
        el.uploadStatus.textContent = `Loaded ${file.name}`;
        el.initSourceBadge.textContent = data.label;
      } catch (err) {
        el.uploadStatus.textContent = "Upload error";
        showError(err.message);
      }
    });

    el.loadHistoryBtn.addEventListener("click", async () => {
      showError("");
      const idx = parseInt(el.historyIndexInput.value, 10);
      try {
        const res = await fetch(`/api/init/history?index=${encodeURIComponent(idx)}`);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || "History load failed");
        state.initSource = "history";
        state.initStates = data.init_states;
        state.doy = data.doy;
        el.doyInput.value = String(data.doy);
        el.historyStatus.textContent = `Loaded #${data.index} (DOY ${data.doy})`;
        el.initSourceBadge.textContent = data.label;
      } catch (err) {
        showError(err.message);
      }
    });

    const syncSteps = (rawVal) => {
      const v = clamp(Math.round(parseFloat(rawVal) || 10), 1, 30);
      state.steps = v;
      el.stepsRange.value = String(v);
      el.stepsInput.value = String(v);
    };
    el.stepsRange.addEventListener("input", () => syncSteps(el.stepsRange.value));
    el.stepsInput.addEventListener("change", () => syncSteps(el.stepsInput.value));

    document.querySelectorAll("input[name='preset']").forEach((radio) => {
      radio.addEventListener("change", () => {
        pushShockSnapshot();
        selectPreset(radio.value);
        validateCustomShock();
      });
    });

    document.querySelectorAll("[data-draw-shape]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.drawShape = btn.dataset.drawShape;
        document.querySelectorAll("[data-draw-shape]").forEach((b) => b.classList.toggle("active", b === btn));
        renderMapOverlays();
      });
    });

    document.querySelectorAll("[data-falloff]").forEach((btn) => {
      btn.addEventListener("click", () => {
        pushShockSnapshot();
        state.falloff = btn.dataset.falloff;
        document.querySelectorAll("[data-falloff]").forEach((b) => b.classList.toggle("active", b === btn));
      });
    });

    if (el.customRadiusRange) {
      el.customRadiusRange.addEventListener("input", () => {
        el.customRadius.value = el.customRadiusRange.value;
        validateCustomShock();
        updateDrawReadout();
        renderMapOverlays();
      });
    }

    el.customRadius.addEventListener("input", () => {
      if (el.customRadiusRange) el.customRadiusRange.value = el.customRadius.value;
      validateCustomShock();
      updateDrawReadout();
      renderMapOverlays();
    });

    [el.customLat, el.customLon].forEach((inp) => {
      inp.addEventListener("input", () => {
        validateCustomShock();
        updateDrawReadout();
        renderMapOverlays();
      });
    });

    for (const [key, range] of Object.entries(el.effRanges)) {
      const num = el.effInputs[key];
      if (!range || !num) continue;
      range.addEventListener("input", () => {
        num.value = range.value;
        validateCustomShock();
      });
      num.addEventListener("input", () => {
        range.value = num.value;
        validateCustomShock();
      });
    }

    if (el.undoShockBtn) {
      el.undoShockBtn.addEventListener("click", () => {
        if (state.shockStack.length === 0) return;
        const prev = state.shockStack.pop();
        restoreShockSnapshot(prev);
        el.undoShockBtn.disabled = state.shockStack.length === 0;
      });
    }

    el.clearCustomBtn.addEventListener("click", () => {
      pushShockSnapshot();
      el.customLat.value = "16.5";
      el.customLon.value = "85.5";
      el.customRadius.value = "3.0";
      if (el.customRadiusRange) el.customRadiusRange.value = "3.0";
      Object.values(el.effInputs).forEach((inp) => {
        inp.value = "0";
      });
      Object.values(el.effRanges).forEach((inp) => {
        if (inp) inp.value = "0";
      });
      el.effInputs.t2m.value = "3.0";
      if (el.effRanges.t2m) el.effRanges.t2m.value = "3.0";
      el.effInputs.tcwv.value = "4.0";
      if (el.effRanges.tcwv) el.effRanges.tcwv.value = "4.0";
      validateCustomShock();
      updateDrawReadout();
      renderMapOverlays();
    });

    document.querySelectorAll("[data-ch]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.chIdx = parseInt(btn.dataset.ch, 10);
        document.querySelectorAll("[data-ch]").forEach((b) => b.classList.toggle("active", b === btn));
        renderMapsAndColorbars();
        renderTimeseries();
      });
    });

    el.playBtn.addEventListener("click", togglePlayback);
    if (el.playSpeedSelect) {
      el.playSpeedSelect.addEventListener("change", () => {
        state.playSpeedMs = parseInt(el.playSpeedSelect.value, 10) || 350;
        if (state.playing) startPlayTimer();
      });
    }
    el.prevDayBtn.addEventListener("click", () => stepDay(-1));
    el.nextDayBtn.addEventListener("click", () => stepDay(1));
    el.daySlider.addEventListener("input", () => {
      stopPlayback();
      state.dayIdx = parseInt(el.daySlider.value, 10);
      triggerCrossfade();
      renderMapsAndColorbars();
      renderTimeseries();
    });

    document.querySelectorAll("[data-chart-scope]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.chartScope = btn.dataset.chartScope;
        document.querySelectorAll("[data-chart-scope]").forEach((b) => b.classList.toggle("active", b === btn));
        renderTimeseries();
      });
    });

    const getChartDayIdx = (evt) => {
      if (!state.runData) return null;
      const rect = el.tsChart.getBoundingClientRect();
      const svgX = ((evt.clientX - rect.left) / rect.width) * 540;
      const mL = 48;
      const mR = 68;
      const plotW = 540 - mL - mR;
      const frac = clamp((svgX - mL) / plotW, 0, 1);
      const steps = state.runData.steps;
      return steps <= 1 ? 0 : Math.round(frac * (steps - 1));
    };

    el.tsChart.addEventListener("mousemove", (evt) => {
      const idx = getChartDayIdx(evt);
      if (idx === null) return;
      state.chartHoverIdx = idx;
      const ch = state.channels[state.chIdx];
      const ts = state.runData.timeseries;
      const ap = ts.all[ch.key].prime[idx];
      const am = ts.all[ch.key].multiverse[idx];
      const lp = ts.land[ch.key].prime[idx];
      const lm = ts.land[ch.key].multiverse[idx];
      if (el.chartHoverReadout) {
        el.chartHoverReadout.textContent =
          `Day ${idx + 1}: All P=${formatNum(ap, 2)} M=${formatNum(am, 2)} · Land P=${formatNum(lp, 2)} M=${formatNum(lm, 2)} ${ch.unit}`;
      }
      renderTimeseries();
    });

    el.tsChart.addEventListener("mouseleave", () => {
      state.chartHoverIdx = null;
      if (el.chartHoverReadout) {
        el.chartHoverReadout.textContent = "Hover chart to inspect day values · click a day to jump";
      }
      renderTimeseries();
    });

    el.tsChart.addEventListener("click", (evt) => {
      const idx = getChartDayIdx(evt);
      if (idx === null) return;
      stopPlayback();
      state.dayIdx = idx;
      triggerCrossfade();
      renderMapsAndColorbars();
      renderTimeseries();
    });

    document.querySelectorAll(".map-frame").forEach((frame) => {
      const canvas = frame.querySelector("canvas");
      const getGridPoint = (evt) => {
        const rect = canvas.getBoundingClientRect();
        const fx = clamp((evt.clientX - rect.left) / rect.width, 0, 0.9999);
        const fy = clamp((evt.clientY - rect.top) / rect.height, 0, 0.9999);
        const j = Math.floor(fx * 36);
        const i = 31 - Math.floor(fy * 32);
        return { i, j, lat: 6.5 + i, lon: 65.5 + j };
      };

      frame.addEventListener("mousedown", (evt) => {
        if (state.preset !== "custom" || evt.button !== 0) return;
        evt.preventDefault();
        const pt = getGridPoint(evt);
        state.dragDraw = { startLat: pt.lat, startLon: pt.lon, curLat: pt.lat, curLon: pt.lon };
        renderMapOverlays();
      });

      frame.addEventListener("mousemove", (evt) => {
        const pt = getGridPoint(evt);
        if (state.preset === "custom") {
          state.hoverGrid = pt;
          if (state.dragDraw) {
            state.dragDraw.curLat = pt.lat;
            state.dragDraw.curLon = pt.lon;
            const lim = getLimits();
            if (state.drawShape === "rect") {
              const midLat = Math.round((state.dragDraw.startLat + pt.lat)) * 0.5;
              const midLon = Math.round((state.dragDraw.startLon + pt.lon)) * 0.5;
              const halfSpan = Math.max(Math.abs(pt.lat - state.dragDraw.startLat), Math.abs(pt.lon - state.dragDraw.startLon)) * 0.5;
              const r = clamp(Math.round(halfSpan * 2) * 0.5, lim.min_radius_cells, lim.max_radius_cells);
              updateDrawReadout(midLat, midLon, r);
            } else {
              const dist = Math.hypot(pt.lat - state.dragDraw.startLat, pt.lon - state.dragDraw.startLon);
              const r = clamp(Math.round(dist * 2) * 0.5, lim.min_radius_cells, lim.max_radius_cells);
              updateDrawReadout(state.dragDraw.startLat, state.dragDraw.startLon, r);
            }
          } else {
            updateDrawReadout(pt.lat, pt.lon);
          }
          renderMapOverlays();
        }
        if (!state.runData) return;
        const { i, j, lat, lon } = pt;
        state.crosshair = { i, j };
        renderMapOverlays();
        const ch = state.channels[state.chIdx];
        const k = state.dayIdx;
        const pv = state.runData.control[k][state.chIdx][i][j];
        const mv = state.runData.shocked[k][state.chIdx][i][j];
        const dv = state.runData.diff[k][state.chIdx][i][j];
        const isLand = state.meta && state.meta.grid.land_sea_mask[i][j] >= 0.5;
        const summary = `${lat.toFixed(1)}°N ${lon.toFixed(1)}°E (${isLand ? "land" : "sea"}) · Prime ${formatNum(pv, 2)} ${ch.unit} · Multiverse ${formatNum(mv, 2)} ${ch.unit} · Δ ${dv >= 0 ? "+" : ""}${formatNum(dv, 2)} ${ch.unit}`;
        if (el.cellReadout) el.cellReadout.textContent = summary;

        el.mapTooltip.textContent =
          `${lat.toFixed(1)}°N ${lon.toFixed(1)}°E (${isLand ? "land" : "sea"}) | P:${formatNum(pv, 2)} M:${formatNum(mv, 2)} Δ:${dv >= 0 ? "+" : ""}${formatNum(dv, 2)} ${ch.unit}`;
        el.mapTooltip.classList.remove("hidden");
        const tx = Math.min(window.innerWidth - 320, evt.clientX + 12);
        const ty = Math.max(8, evt.clientY - 28);
        el.mapTooltip.style.left = `${tx}px`;
        el.mapTooltip.style.top = `${ty}px`;
      });

      frame.addEventListener("mouseleave", () => {
        el.mapTooltip.classList.add("hidden");
        state.crosshair = null;
        if (state.hoverGrid) {
          state.hoverGrid = null;
          if (!state.dragDraw) updateDrawReadout();
        }
        renderMapOverlays();
      });
    });

    window.addEventListener("mouseup", () => {
      if (!state.dragDraw) return;
      const { startLat, startLon, curLat, curLon } = state.dragDraw;
      state.dragDraw = null;
      pushShockSnapshot();
      const lim = getLimits();
      let newLat = startLat;
      let newLon = startLon;
      let newRad = parseFloat(el.customRadius.value) || 3.0;
      const dist = Math.hypot(curLat - startLat, curLon - startLon);
      if (dist >= 0.5) {
        if (state.drawShape === "rect") {
          newLat = clamp(Math.round((startLat + curLat)) * 0.5, 6.5, 37.5);
          newLon = clamp(Math.round((startLon + curLon)) * 0.5, 65.5, 100.5);
          const halfSpan = Math.max(Math.abs(curLat - startLat), Math.abs(curLon - startLon)) * 0.5;
          newRad = clamp(Math.round(halfSpan * 2) * 0.5, lim.min_radius_cells, lim.max_radius_cells);
        } else {
          newRad = clamp(Math.round(dist * 2) * 0.5, lim.min_radius_cells, lim.max_radius_cells);
        }
      }
      el.customLat.value = newLat.toFixed(1);
      el.customLon.value = newLon.toFixed(1);
      el.customRadius.value = newRad.toFixed(1);
      if (el.customRadiusRange) el.customRadiusRange.value = newRad.toFixed(1);
      validateCustomShock();
      updateDrawReadout();
      renderMapOverlays();
    });

    window.addEventListener("keydown", (evt) => {
      if (evt.target && ["INPUT", "TEXTAREA", "SELECT"].includes(evt.target.tagName)) return;
      if (evt.key === "ArrowLeft") {
        evt.preventDefault();
        stepDay(-1);
      } else if (evt.key === "ArrowRight") {
        evt.preventDefault();
        stepDay(1);
      } else if (evt.key === " ") {
        evt.preventDefault();
        togglePlayback();
      } else if (/^[1-6]$/.test(evt.key)) {
        const idx = parseInt(evt.key, 10) - 1;
        state.chIdx = idx;
        document.querySelectorAll("[data-ch]").forEach((b) => b.classList.toggle("active", parseInt(b.dataset.ch, 10) === idx));
        renderMapsAndColorbars();
        renderTimeseries();
      }
    });
  }

  async function init() {
    bindEvents();
    try {
      const metaRes = await fetch("/api/meta");
      state.meta = await metaRes.json();
      if (state.meta.channels) state.channels = state.meta.channels;
      syncCustomChannelBounds();
      updateDrawReadout();
      if (!state.meta.history_available) {
        el.historyTabBtn.disabled = true;
        el.historyTabBtn.title = "History dataset not available";
      }
      renderMapOverlays();
      renderPresetMiniMaps();
    } catch {
      showError("Failed to load /api/meta from dashboard server.");
    }

    const ready = await refreshHealth();
    if (!ready) {
      showError("Model server unreachable — start serve.py on port 8000, then click Run Experiment.");
    }
  }

  init();
})();
