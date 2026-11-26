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
    chIdx: 0,
    dayIdx: 0,
    playing: false,
    playTimer: null,
    runData: null,
    running: false,
    abortCtrl: null
  };

  const el = {
    healthDot: document.getElementById("healthDot"),
    healthText: document.getElementById("healthText"),
    timingReadout: document.getElementById("timingReadout"),
    runBtn: document.getElementById("runBtn"),
    progressBar: document.getElementById("progressBar"),
    initSourceBadge: document.getElementById("initSourceBadge"),
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
    clearCustomBtn: document.getElementById("clearCustomBtn"),
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
    playBtn: document.getElementById("playBtn"),
    prevDayBtn: document.getElementById("prevDayBtn"),
    nextDayBtn: document.getElementById("nextDayBtn"),
    daySlider: document.getElementById("daySlider"),
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
    tsChart: document.getElementById("tsChart"),
    impactTitle: document.getElementById("impactTitle"),
    impactTbody: document.getElementById("impactTbody"),
    mapTooltip: document.getElementById("mapTooltip")
  };

  function showError(msg) {
    if (!msg) {
      el.errorBanner.textContent = "";
      el.errorBanner.classList.add("hidden");
    } else {
      el.errorBanner.textContent = msg;
      el.errorBanner.classList.remove("hidden");
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

    if (state.preset === "custom") {
      const clat = parseFloat(el.customLat.value);
      const clon = parseFloat(el.customLon.value);
      const rad = parseFloat(el.customRadius.value);
      if (Number.isFinite(clat) && Number.isFinite(clon) && Number.isFinite(rad)) {
        const cx = clon - 65.5 + 0.5;
        const cy = 32 - (clat - 6.5 + 0.5);
        customMarker =
          `<circle cx="${cx.toFixed(2)}" cy="${cy.toFixed(2)}" r="${rad.toFixed(2)}" fill="none" stroke="${SHOCK}" stroke-width="0.28" stroke-dasharray="0.6 0.4"/>` +
          `<path d="M${(cx - 0.6).toFixed(2)} ${cy.toFixed(2)}H${(cx + 0.6).toFixed(2)}M${cx.toFixed(2)} ${(cy - 0.6).toFixed(2)}V${(cy + 0.6).toFixed(2)}" stroke="${SHOCK}" stroke-width="0.28"/>`;
      }
    } else if (state.runData && state.runData.region_mask && state.runData.preset === state.preset) {
      const p = buildContourPath(state.runData.region_mask);
      if (p) {
        shockPath = `<path d="${p}" fill="none" stroke="${SHOCK}" stroke-width="0.30" stroke-dasharray="0.7 0.45"/>`;
      }
    }

    const svgInner =
      `<path d="${cachedCoastPath}" fill="none" stroke="${INK}" stroke-width="0.16" stroke-opacity="0.85"/>` +
      shockPath +
      customMarker;

    el.overlayPrime.innerHTML = svgInner;
    el.overlayMulti.innerHTML = svgInner;
    el.overlayDiff.innerHTML = svgInner;
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

    if (state.runData.max_abs_diff === 0 || state.runData.preset === "null") {
      el.cbDiffMin.textContent = "0";
      el.cbDiffMax.textContent = "0";
      el.nullCaption.classList.remove("hidden");
      el.nullCaption.textContent = `difference = 0 exactly (max |Δ| = ${state.runData.max_abs_diff.toFixed(1)})`;
      el.diffCorner.textContent = "max |Δ| = 0.0";
    } else {
      el.cbDiffMin.textContent = `-${formatNum(dAbs, 2)}`;
      el.cbDiffMax.textContent = `+${formatNum(dAbs, 2)}`;
      el.nullCaption.classList.add("hidden");
      el.diffCorner.textContent = `day ${k + 1} · max |Δ| = ${formatNum(dAbs, 2)}`;
    }

    const curDoy = rolloutDoy(state.runData.doy, k);
    el.daySlider.max = String(state.runData.steps - 1);
    el.daySlider.value = String(k);
    el.dayReadout.textContent = `Day ${k + 1} / ${state.runData.steps} (DOY ${curDoy})`;
    el.primeCorner.textContent = `day ${k + 1} (doy ${curDoy})`;
    el.multiCorner.textContent = `day ${k + 1} (doy ${curDoy})`;

    renderMapOverlays();
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

    const vals = allP.concat(allM, landP, landM);
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
    out += `<line x1="${curX.toFixed(1)}" y1="${mT}" x2="${curX.toFixed(1)}" y2="${H - mB}" stroke="${ACCENT}" stroke-width="1.2"/>`;

    out += `<path d="${toPath(landP)}" fill="none" stroke="${INK}" stroke-width="1.4" stroke-dasharray="3.5 2.5" stroke-opacity="0.7"/>`;
    out += `<path d="${toPath(landM)}" fill="none" stroke="${SHOCK}" stroke-width="1.4" stroke-dasharray="3.5 2.5" stroke-opacity="0.85"/>`;
    out += `<path d="${toPath(allP)}" fill="none" stroke="${INK}" stroke-width="2.0"/>`;
    out += `<path d="${toPath(allM)}" fill="none" stroke="${SHOCK}" stroke-width="1.6"/>`;

    const lastIdx = steps - 1;
    const endX = xOf(lastIdx) + 5;
    let yEndP = yOf(allP[lastIdx]);
    let yEndM = yOf(allM[lastIdx]);
    if (Math.abs(yEndP - yEndM) < 11) {
      yEndP -= 5;
      yEndM += 6;
    }
    out += `<text x="${endX.toFixed(1)}" y="${(yEndP + 3).toFixed(1)}" fill="${INK}" font-size="10">Prime</text>`;
    out += `<text x="${endX.toFixed(1)}" y="${(yEndM + 3).toFixed(1)}" fill="${SHOCK}" font-size="10">Multiverse</text>`;

    out += `<circle cx="${curX.toFixed(1)}" cy="${yOf(allP[state.dayIdx]).toFixed(1)}" r="2.8" fill="${INK}"/>`;
    out += `<circle cx="${curX.toFixed(1)}" cy="${yOf(allM[state.dayIdx]).toFixed(1)}" r="2.5" fill="${SHOCK}"/>`;

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
        pv: formatNum(p.heatwave_cell_days, 1),
        mv: formatNum(m.heatwave_cell_days, 1),
        dv: d.heatwave_cell_days.abs,
        fmtDelta: (v) => `${v >= 0 ? "+" : ""}${formatNum(v, 1)} cell-d`
      },
      {
        label: "Heavy rain ≥ 64.5 mm/d",
        pv: `${(p.heavy_rain_64_share * 100).toFixed(2)}%`,
        mv: `${(m.heavy_rain_64_share * 100).toFixed(2)}%`,
        dv: d.heavy_rain_64_share.abs * 100,
        fmtDelta: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)} pp`
      },
      {
        label: "Very heavy ≥ 115.6 mm/d",
        pv: `${(p.heavy_rain_115_share * 100).toFixed(2)}%`,
        mv: `${(m.heavy_rain_115_share * 100).toFixed(2)}%`,
        dv: d.heavy_rain_115_share.abs * 100,
        fmtDelta: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)} pp`
      },
      {
        label: "Rain deficit (> 20% below clim)",
        pv: `${(p.rain_deficit_share * 100).toFixed(1)}%`,
        mv: `${(m.rain_deficit_share * 100).toFixed(1)}%`,
        dv: d.rain_deficit_share.abs * 100,
        fmtDelta: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(1)} pp`
      },
      {
        label: "Crop stress CSI (weighted)",
        pv: p.crop_stress_csi.toFixed(3),
        mv: m.crop_stress_csi.toFixed(3),
        dv: d.crop_stress_csi.abs,
        fmtDelta: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(3)}`
      }
    ];

    el.impactTbody.innerHTML = rows
      .map((r) => {
        const eps = 1e-9;
        let arrow = "·";
        let cls = "muted-text";
        if (r.dv > eps) {
          arrow = "▲";
          cls = "delta-up";
        } else if (r.dv < -eps) {
          arrow = "▼";
          cls = "delta-down";
        }
        return `<tr>
          <td>${r.label}</td>
          <td class="num">${r.pv}</td>
          <td class="num">${r.mv}</td>
          <td class="num ${cls}">${arrow} ${r.fmtDelta(r.dv)}</td>
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
      const ready = data.model_reachable && data.model && data.model.status === "ready";
      el.healthDot.classList.remove("ready", "warn");
      if (ready) {
        el.healthDot.classList.add("ready");
        el.healthText.textContent = "model ready";
      } else if (data.model_reachable) {
        el.healthDot.classList.add("warn");
        el.healthText.textContent = "model loading";
      } else {
        el.healthDot.classList.add("warn");
        el.healthText.textContent = "model offline";
      }
      return ready;
    } catch {
      el.healthDot.classList.remove("ready");
      el.healthDot.classList.add("warn");
      el.healthText.textContent = "server offline";
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
    if (state.running && state.abortCtrl) {
      state.abortCtrl.abort();
    }
    showError("");
    state.running = true;
    state.abortCtrl = new AbortController();
    const timeoutId = setTimeout(() => state.abortCtrl.abort(), 20000);

    el.runBtn.disabled = true;
    el.progressBar.classList.add("active");

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
      state.dayIdx = clamp(state.dayIdx, 0, data.steps - 1);
      el.timingReadout.textContent = `inf ${data.inference_ms.total.toFixed(0)} ms · wall ${data.wall_ms.toFixed(0)} ms`;
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

  function togglePlayback() {
    if (!state.runData) return;
    if (state.playing) {
      stopPlayback();
      return;
    }
    state.playing = true;
    el.playBtn.textContent = "Pause";
    state.playTimer = setInterval(() => {
      if (!state.runData) return stopPlayback();
      state.dayIdx = (state.dayIdx + 1) % state.runData.steps;
      renderMapsAndColorbars();
      renderTimeseries();
    }, 350);
  }

  function stepDay(delta) {
    if (!state.runData) return;
    stopPlayback();
    state.dayIdx = clamp(state.dayIdx + delta, 0, state.runData.steps - 1);
    renderMapsAndColorbars();
    renderTimeseries();
  }

  function bindEvents() {
    el.runBtn.addEventListener("click", runExperiment);

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
        state.preset = radio.value;
        el.customControls.classList.toggle("hidden", state.preset !== "custom");
        document.querySelectorAll(".map-stage").forEach((s) => s.classList.toggle("pick-mode", state.preset === "custom"));
        if (state.meta) {
          const found = state.meta.presets.find((p) => p.id === state.preset);
          if (found) el.presetDesc.textContent = found.summary;
        }
        renderMapOverlays();
      });
    });

    document.querySelectorAll("[data-falloff]").forEach((btn) => {
      btn.addEventListener("click", () => {
        state.falloff = btn.dataset.falloff;
        document.querySelectorAll("[data-falloff]").forEach((b) => b.classList.toggle("active", b === btn));
      });
    });

    [el.customLat, el.customLon, el.customRadius].forEach((inp) => {
      inp.addEventListener("input", renderMapOverlays);
    });

    el.clearCustomBtn.addEventListener("click", () => {
      Object.values(el.effInputs).forEach((inp) => {
        inp.value = "0";
      });
      el.effInputs.t2m.value = "3.0";
      el.effInputs.tcwv.value = "4.0";
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
    el.prevDayBtn.addEventListener("click", () => stepDay(-1));
    el.nextDayBtn.addEventListener("click", () => stepDay(1));
    el.daySlider.addEventListener("input", () => {
      stopPlayback();
      state.dayIdx = parseInt(el.daySlider.value, 10);
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

      frame.addEventListener("mousemove", (evt) => {
        if (!state.runData) return;
        const { i, j, lat, lon } = getGridPoint(evt);
        const ch = state.channels[state.chIdx];
        const k = state.dayIdx;
        const pv = state.runData.control[k][state.chIdx][i][j];
        const mv = state.runData.shocked[k][state.chIdx][i][j];
        const dv = state.runData.diff[k][state.chIdx][i][j];
        const isLand = state.meta && state.meta.grid.land_sea_mask[i][j] >= 0.5;

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
      });

      frame.addEventListener("click", (evt) => {
        if (state.preset !== "custom") return;
        const { lat, lon } = getGridPoint(evt);
        el.customLat.value = lat.toFixed(1);
        el.customLon.value = lon.toFixed(1);
        renderMapOverlays();
      });
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
      }
    });
  }

  async function init() {
    bindEvents();
    try {
      const metaRes = await fetch("/api/meta");
      state.meta = await metaRes.json();
      if (state.meta.channels) state.channels = state.meta.channels;
      if (!state.meta.history_available) {
        el.historyTabBtn.disabled = true;
        el.historyTabBtn.title = "History dataset not available";
      }
      renderMapOverlays();
    } catch {
      showError("Failed to load /api/meta from dashboard server.");
    }

    const ready = await refreshHealth();
    if (ready) {
      runExperiment();
    } else {
      showError("Model server offline — start serve.py on port 8000, then click Run Experiment.");
    }
  }

  init();
})();
