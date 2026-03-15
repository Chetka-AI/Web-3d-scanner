/**
 * RealtimeScanner v2 — stabilized live 3D reconstruction.
 *
 * Main upgrades over the first implementation:
 *  - yaw unwrap + pose fusion from IMU and image tracking
 *  - keyframe selection before fusion
 *  - confidence-based voxel integration instead of binary carving
 *  - stable world-space rendering backed by reusable GPU buffers
 *  - generic camera pose math for projection / triangulation
 */
const RealtimeScanner = (() => {
  const cfg = {
    procSize: 240,
    cannyLo: 30,
    cannyHi: 80,
    blurRadius: 1,
    sharpen: false,
    sharpenAmount: 0.4,

    minLineLen: 12,
    cornerDist: 8,
    dpEpsilon: 3.0,

    voxelRes: 48,
    voxelWorld: 1.6,
    cameraR: 2.3,
    fovDeg: 60,

    silhouetteMode: 'auto',
    manualThreshold: 128,
    invertSilhouette: false,
    morphType: 'close',
    morphIterations: 1,

    showEdges: true,
    showLines: true,
    showCorners: true,
    showAngles: true,
    showSilhouette: true,
    overlayOpacity: 0.35,

    minFrameMs: 50,
    motionThreshold: 0.005,
    smoothOrientation: true,
    smoothFactor: 0.24,
    poseAssist: true,
    poseAssistGain: 0.65,
    targetCoverage: 0.22,
    poseRefineEnabled: true,
    poseRefineIterations: 2,
    poseRefineSamples: 180,
    poseRefineStepYaw: 0.035,
    poseRefineStepOffset: 0.025,
    poseRefineStepRadius: 0.04,
    poseRefineMapWeight: 0.95,

    keyframeMinQuality: 42,
    keyframeMinRotation: 0.08,
    keyframeMinTranslation: 0.04,
    keyframeMinCentroid: 0.035,
    reprojectionThresholdPx: 12,
    minBaselineDeg: 1.6,
    vertexMergeDist: 0.05,

    syncCamera: true,
    showWireframe: true,
    showVoxels: true,
    showSurfels: true,
    renderBudget: 22000,

    descRadius: 3,
    matchThreshold: 0.55,
    minAngleDiff: 0.15,
    maxHistory: 8,

    voxelHitWeight: 0.28,
    voxelMissWeight: 0.32,
    voxelSupportThreshold: 1.0,
    voxelRemoveThreshold: 1.4,
    voxelMinObservations: 2,
    surfelCell: 0.028,
    surfelMinConfidence: 2,
  };

  const PRESETS = {
    fast: {
      procSize: 180,
      cannyLo: 40,
      cannyHi: 100,
      blurRadius: 1,
      sharpen: false,
      minLineLen: 16,
      voxelRes: 32,
      minFrameMs: 34,
      morphType: 'none',
      morphIterations: 0,
      dpEpsilon: 4,
      maxHistory: 5,
      renderBudget: 14000,
      poseRefineIterations: 1,
      poseRefineSamples: 96,
      keyframeMinQuality: 48,
      keyframeMinRotation: 0.11,
      keyframeMinTranslation: 0.06,
      voxelMinObservations: 2,
      reprojectionThresholdPx: 15,
      minBaselineDeg: 2.1,
      surfelMinConfidence: 2,
    },
    balanced: {
      procSize: 240,
      cannyLo: 30,
      cannyHi: 80,
      blurRadius: 1,
      sharpen: false,
      minLineLen: 12,
      voxelRes: 48,
      minFrameMs: 50,
      morphType: 'close',
      morphIterations: 1,
      dpEpsilon: 3,
      maxHistory: 8,
      renderBudget: 22000,
      poseRefineIterations: 2,
      poseRefineSamples: 180,
      keyframeMinQuality: 42,
      keyframeMinRotation: 0.08,
      keyframeMinTranslation: 0.04,
      voxelMinObservations: 2,
      reprojectionThresholdPx: 12,
      minBaselineDeg: 1.6,
      surfelMinConfidence: 2,
    },
    quality: {
      procSize: 320,
      cannyLo: 20,
      cannyHi: 65,
      blurRadius: 2,
      sharpen: true,
      sharpenAmount: 0.6,
      minLineLen: 8,
      voxelRes: 64,
      minFrameMs: 75,
      morphType: 'close',
      morphIterations: 2,
      dpEpsilon: 2,
      maxHistory: 10,
      renderBudget: 36000,
      poseRefineIterations: 3,
      poseRefineSamples: 260,
      keyframeMinQuality: 36,
      keyframeMinRotation: 0.06,
      keyframeMinTranslation: 0.03,
      voxelMinObservations: 3,
      reprojectionThresholdPx: 10,
      minBaselineDeg: 1.1,
      surfelMinConfidence: 3,
    },
  };

  let videoEl = null;
  let overlayCanvas = null;
  let overlayCtx = null;
  let previewCanvas = null;
  let stream = null;

  let scene;
  let cam3d;
  let renderer;
  let previewPoints = null;
  let previewGeometry = null;
  let previewMaterial = null;
  let previewPositions = new Float32Array(0);
  let previewColors = new Float32Array(0);
  let previewCapacity = 0;
  let previewCount = 0;
  let previewConfidence = 0;

  let scanning = false;
  let paused = false;
  let processing = false;
  let renderFrame = null;
  let frameCanvas = null;
  let frameCtx = null;

  let voxelState = null;
  let voxelEvidence = null;
  let voxelInside = null;
  let voxelOutside = null;
  let voxelColorAcc = null;
  let voxelColorWeight = null;
  let voxelsDirty = false;

  let frameCount = 0;
  let edgeCount = 0;
  let cornerCount = 0;
  let fps = 0;
  let lastProcTime = 0;
  let quality = 0;
  let keyframeCount = 0;
  let trackingMatches = 0;
  let liveOrientation = null;
  let poseLock = 0;

  let sensorYaw = 0;
  let sensorPitch = 0;
  let lastSensorAlpha = null;
  let sensorReady = false;

  let poseState = null;
  let lastPose = null;
  let lastIntegratedPose = null;
  let lastTrackingFrame = null;

  let keyframes = [];
  let vertexHistory = [];
  let vertices3D = [];
  let edges3D = new Set();
  let surfels = [];
  let surfelIndex = new Map();
  let wireVertexObj = null;
  let wireEdgeObj = null;
  let surfelObj = null;
  let wireframeDirty = false;

  let onStatsUpdate = null;
  let onStatusChange = null;

  function set(key, value) {
    if (!(key in cfg)) return;
    cfg[key] = value;
    if (scanning && ['voxelRes', 'voxelWorld', 'cameraR'].includes(key)) {
      initVoxelGrid();
      resetTrackingState();
      wireframeDirty = true;
      voxelsDirty = true;
    }
    if (['showWireframe', 'showSurfels', 'surfelMinConfidence'].includes(key)) {
      wireframeDirty = true;
      voxelsDirty = true;
    }
    if (['voxelSupportThreshold', 'voxelMinObservations', 'renderBudget'].includes(key)) {
      voxelsDirty = true;
    }
    if (key === 'surfelCell' && scanning) {
      rebuildSurfelsFromVertices();
      wireframeDirty = true;
      voxelsDirty = true;
    }
  }

  function get(key) {
    return cfg[key];
  }

  function getAll() {
    return { ...cfg };
  }

  function applyPreset(name) {
    const preset = PRESETS[name];
    if (!preset) return;
    Object.keys(preset).forEach(key => {
      cfg[key] = preset[key];
    });
    if (scanning) {
      initVoxelGrid();
      resetTrackingState();
      clear3DScene();
      ensurePreviewObject();
    }
  }

  function init(video, overlay, canvas3d, callbacks) {
    videoEl = video;
    overlayCanvas = overlay;
    overlayCtx = overlay.getContext('2d');
    previewCanvas = canvas3d;
    onStatsUpdate = callbacks?.onStats || null;
    onStatusChange = callbacks?.onStatus || null;
    setup3D();
  }

  async function startCamera() {
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: 'environment',
          width: { ideal: 1280 },
          height: { ideal: 960 },
        },
        audio: false,
      });
      videoEl.srcObject = stream;
      await videoEl.play();
      if (typeof CameraModule !== 'undefined' && CameraModule.startOrientation) {
        CameraModule.startOrientation();
      }
      return true;
    } catch (err) {
      console.error('Camera error:', err);
      return false;
    }
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      stream = null;
    }
    if (videoEl) {
      videoEl.srcObject = null;
    }
    if (typeof CameraModule !== 'undefined' && CameraModule.stopOrientation) {
      CameraModule.stopOrientation();
    }
  }

  function startScanning() {
    scanning = true;
    paused = false;
    processing = false;
    frameCount = 0;
    edgeCount = 0;
    cornerCount = 0;
    fps = 0;
    quality = 0;
    trackingMatches = 0;
    keyframeCount = 0;
    previewConfidence = 0;
    resetTrackingState();
    initVoxelGrid();
    clear3DScene();
    ensurePreviewObject();
    startRenderLoop();
    emitStatus('scanning');
    scheduleFrame();
  }

  function pauseScanning() {
    paused = true;
    emitStatus('paused');
  }

  function resumeScanning() {
    paused = false;
    emitStatus('scanning');
    scheduleFrame();
  }

  function stopScanning() {
    scanning = false;
    paused = false;
    processing = false;
    emitStatus('stopped');
  }

  function reset() {
    stopScanning();
    initVoxelGrid();
    resetTrackingState();
    clear3DScene();
    ensurePreviewObject();
    if (overlayCtx && overlayCanvas) {
      overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    }
    emitStats();
  }

  function destroy() {
    stopScanning();
    stopCamera();
    stopRenderLoop();
    clear3DScene();
    voxelState = null;
    voxelEvidence = null;
    voxelInside = null;
    voxelOutside = null;
    voxelColorAcc = null;
    voxelColorWeight = null;
    surfels = [];
    surfelIndex = new Map();
  }

  function isScanning() {
    return scanning;
  }

  function isPaused() {
    return paused;
  }

  function isActive() {
    return stream !== null;
  }

  function getPointCloud() {
    const entries = collectRenderableMapEntries(Infinity);
    const positions = new Float32Array(entries.length * 3);
    const colors = new Float32Array(entries.length * 3);
    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      positions[i * 3] = entry.x;
      positions[i * 3 + 1] = entry.y;
      positions[i * 3 + 2] = entry.z;
      colors[i * 3] = entry.r;
      colors[i * 3 + 1] = entry.g;
      colors[i * 3 + 2] = entry.b;
    }
    return { positions, colors };
  }

  function getStats() {
    return {
      frames: frameCount,
      points: previewCount,
      edges: edgeCount,
      corners: cornerCount,
      verts3D: vertices3D.length,
      edges3D: edges3D.size,
      fps,
      quality,
      scanning,
      paused,
      keyframes: keyframeCount,
      matches: trackingMatches,
      mapConfidence: Math.round(previewConfidence * 100),
      poseLock: Math.round(poseLock * 100),
      surfels: surfels.length,
    };
  }

  function resetTrackingState() {
    sensorYaw = 0;
    sensorPitch = 0;
    lastSensorAlpha = null;
    sensorReady = false;
    poseState = {
      yaw: 0,
      pitch: 0,
      offsetX: 0,
      offsetY: 0,
      radius: cfg.cameraR,
      yawBias: 0,
      pitchBias: 0,
      confidence: 0,
    };
    lastPose = null;
    lastIntegratedPose = null;
    lastTrackingFrame = null;
    keyframes = [];
    vertexHistory = [];
    vertices3D = [];
    edges3D = new Set();
    surfels = [];
    surfelIndex = new Map();
    poseLock = 0;
    wireframeDirty = true;
    liveOrientation = buildPose(0, 0, cfg.cameraR, 0, 0);
  }

  function ensureFrameCanvas(w, h) {
    if (!frameCanvas) {
      frameCanvas = document.createElement('canvas');
      frameCtx = frameCanvas.getContext('2d', { willReadFrequently: true });
    }
    if (frameCanvas.width !== w || frameCanvas.height !== h) {
      frameCanvas.width = w;
      frameCanvas.height = h;
    }
  }

  function toGray(data, n) {
    const out = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      const j = i * 4;
      out[i] = 0.299 * data[j] + 0.587 * data[j + 1] + 0.114 * data[j + 2];
    }
    return out;
  }

  function unsharpMask(gray, w, h, amount) {
    const blurred = gaussianBlur(gray, w, h, 2);
    const out = new Float32Array(w * h);
    for (let i = 0; i < out.length; i++) {
      out[i] = clamp(gray[i] + amount * (gray[i] - blurred[i]), 0, 255);
    }
    return out;
  }

  function cannyEdges(gray, w, h) {
    const blurred = cfg.blurRadius > 0 ? gaussianBlur(gray, w, h, cfg.blurRadius) : gray;
    const { mag, dir } = sobelGradients(blurred, w, h);
    const thin = nonMaxSuppress(mag, dir, w, h);
    return doubleThreshold(thin, w, h, cfg.cannyLo, cfg.cannyHi);
  }

  function gaussianBlur(src, w, h, radius) {
    if (radius <= 0) return src;
    const dst = new Float32Array(w * h);
    if (radius === 1) {
      const k = [1 / 16, 2 / 16, 1 / 16, 2 / 16, 4 / 16, 2 / 16, 1 / 16, 2 / 16, 1 / 16];
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          let sum = 0;
          let ki = 0;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              sum += src[(y + dy) * w + (x + dx)] * k[ki++];
            }
          }
          dst[y * w + x] = sum;
        }
      }
      return dst;
    }
    const k = [
      1, 4, 6, 4, 1,
      4, 16, 24, 16, 4,
      6, 24, 36, 24, 6,
      4, 16, 24, 16, 4,
      1, 4, 6, 4, 1,
    ];
    for (let y = 2; y < h - 2; y++) {
      for (let x = 2; x < w - 2; x++) {
        let sum = 0;
        let ki = 0;
        for (let dy = -2; dy <= 2; dy++) {
          for (let dx = -2; dx <= 2; dx++) {
            sum += src[(y + dy) * w + (x + dx)] * k[ki++];
          }
        }
        dst[y * w + x] = sum / 256;
      }
    }
    return dst;
  }

  function sobelGradients(src, w, h) {
    const mag = new Float32Array(w * h);
    const dir = new Float32Array(w * h);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const tl = src[(y - 1) * w + x - 1];
        const tc = src[(y - 1) * w + x];
        const tr = src[(y - 1) * w + x + 1];
        const ml = src[y * w + x - 1];
        const mr = src[y * w + x + 1];
        const bl = src[(y + 1) * w + x - 1];
        const bc = src[(y + 1) * w + x];
        const br = src[(y + 1) * w + x + 1];
        const gx = -tl + tr - 2 * ml + 2 * mr - bl + br;
        const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
        const idx = y * w + x;
        mag[idx] = Math.sqrt(gx * gx + gy * gy);
        dir[idx] = Math.atan2(gy, gx);
      }
    }
    return { mag, dir };
  }

  function nonMaxSuppress(mag, dir, w, h) {
    const out = new Float32Array(w * h);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;
        const m = mag[idx];
        const angle = ((dir[idx] * 180 / Math.PI) + 180) % 180;
        let n1;
        let n2;
        if (angle < 22.5 || angle >= 157.5) {
          n1 = mag[idx - 1];
          n2 = mag[idx + 1];
        } else if (angle < 67.5) {
          n1 = mag[(y - 1) * w + x + 1];
          n2 = mag[(y + 1) * w + x - 1];
        } else if (angle < 112.5) {
          n1 = mag[(y - 1) * w + x];
          n2 = mag[(y + 1) * w + x];
        } else {
          n1 = mag[(y - 1) * w + x - 1];
          n2 = mag[(y + 1) * w + x + 1];
        }
        out[idx] = m >= n1 && m >= n2 ? m : 0;
      }
    }
    return out;
  }

  function doubleThreshold(thin, w, h, lo, hi) {
    const edge = new Uint8Array(w * h);
    for (let i = 0; i < thin.length; i++) {
      if (thin[i] >= hi) edge[i] = 2;
      else if (thin[i] >= lo) edge[i] = 1;
    }
    let changed = true;
    while (changed) {
      changed = false;
      for (let y = 1; y < h - 1; y++) {
        for (let x = 1; x < w - 1; x++) {
          const idx = y * w + x;
          if (edge[idx] !== 1) continue;
          for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
              if (edge[(y + dy) * w + (x + dx)] === 2) {
                edge[idx] = 2;
                changed = true;
                dx = 2;
                dy = 2;
              }
            }
          }
        }
      }
    }
    for (let i = 0; i < edge.length; i++) {
      edge[i] = edge[i] === 2 ? 1 : 0;
    }
    return edge;
  }

  function detectLines(edgeMap, w, h) {
    const lines = [];
    const visited = new Uint8Array(w * h);
    for (let y = 2; y < h - 2; y++) {
      for (let x = 2; x < w - 2; x++) {
        const idx = y * w + x;
        if (!edgeMap[idx] || visited[idx]) continue;
        const chain = traceChain(edgeMap, visited, x, y, w, h);
        if (chain.length < cfg.minLineLen) continue;
        const segs = splitToSegments(chain);
        for (const seg of segs) {
          if (segLen(seg) >= cfg.minLineLen) lines.push(seg);
        }
      }
    }
    return lines;
  }

  function traceChain(edgeMap, visited, sx, sy, w, h) {
    const chain = [];
    const dirs = [
      [-1, -1], [-1, 0], [-1, 1],
      [0, -1], [0, 1],
      [1, -1], [1, 0], [1, 1],
    ];
    let cx = sx;
    let cy = sy;
    for (let step = 0; step < 600; step++) {
      const idx = cy * w + cx;
      if (visited[idx]) break;
      visited[idx] = 1;
      chain.push(cx, cy);
      let found = false;
      for (const [dy, dx] of dirs) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 1 || nx >= w - 1 || ny < 1 || ny >= h - 1) continue;
        if (edgeMap[ny * w + nx] && !visited[ny * w + nx]) {
          cx = nx;
          cy = ny;
          found = true;
          break;
        }
      }
      if (!found) break;
    }
    return chain;
  }

  function splitToSegments(chain) {
    const pts = [];
    for (let i = 0; i < chain.length; i += 2) {
      pts.push({ x: chain[i], y: chain[i + 1] });
    }
    if (pts.length < 2) return [];
    const ids = douglasPeucker(pts, 0, pts.length - 1, cfg.dpEpsilon);
    const segs = [];
    for (let i = 0; i < ids.length - 1; i++) {
      segs.push({
        x1: pts[ids[i]].x,
        y1: pts[ids[i]].y,
        x2: pts[ids[i + 1]].x,
        y2: pts[ids[i + 1]].y,
      });
    }
    return segs;
  }

  function douglasPeucker(pts, start, end, epsilon) {
    if (end <= start + 1) return [start, end];
    let maxDist = 0;
    let maxIndex = start;
    const lx = pts[end].x - pts[start].x;
    const ly = pts[end].y - pts[start].y;
    const lenSq = lx * lx + ly * ly;
    for (let i = start + 1; i < end; i++) {
      const dx = pts[i].x - pts[start].x;
      const dy = pts[i].y - pts[start].y;
      const dist = lenSq > 0
        ? Math.abs(dx * ly - dy * lx) / Math.sqrt(lenSq)
        : Math.sqrt(dx * dx + dy * dy);
      if (dist > maxDist) {
        maxDist = dist;
        maxIndex = i;
      }
    }
    if (maxDist <= epsilon) return [start, end];
    return douglasPeucker(pts, start, maxIndex, epsilon)
      .concat(douglasPeucker(pts, maxIndex, end, epsilon).slice(1));
  }

  function segLen(seg) {
    return Math.sqrt((seg.x2 - seg.x1) ** 2 + (seg.y2 - seg.y1) ** 2);
  }

  function segAng(seg) {
    return Math.atan2(seg.y2 - seg.y1, seg.x2 - seg.x1);
  }

  function detectCornersAndAngles(lines) {
    const raw = [];
    for (let i = 0; i < lines.length; i++) {
      for (let j = i + 1; j < lines.length; j++) {
        const a = lines[i];
        const b = lines[j];
        const pairs = [
          [a.x1, a.y1, b.x1, b.y1],
          [a.x1, a.y1, b.x2, b.y2],
          [a.x2, a.y2, b.x1, b.y1],
          [a.x2, a.y2, b.x2, b.y2],
        ];
        for (const [ax, ay, bx, by] of pairs) {
          if (Math.sqrt((ax - bx) ** 2 + (ay - by) ** 2) >= cfg.cornerDist) continue;
          const a1 = segAng(a);
          const a2 = segAng(b);
          let diff = Math.abs(a1 - a2);
          if (diff > Math.PI) diff = 2 * Math.PI - diff;
          const deg = Math.round(diff * 180 / Math.PI);
          if (deg > 8 && deg < 172) {
            raw.push({ x: (ax + bx) * 0.5, y: (ay + by) * 0.5, angle: deg, a1, a2 });
          }
        }
      }
    }
    const out = [];
    for (const corner of raw) {
      let duplicate = false;
      for (const existing of out) {
        if (Math.sqrt((corner.x - existing.x) ** 2 + (corner.y - existing.y) ** 2) < cfg.cornerDist * 2) {
          duplicate = true;
          break;
        }
      }
      if (!duplicate) out.push(corner);
    }
    return out;
  }

  function extractDescriptor(gray, w, h, cx, cy) {
    const radius = cfg.descRadius;
    const size = (radius * 2 + 1) ** 2;
    const desc = new Float32Array(size);
    let mean = 0;
    let count = 0;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const px = cx + dx;
        const py = cy + dy;
        const value = px >= 0 && px < w && py >= 0 && py < h ? gray[py * w + px] : 0;
        desc[count++] = value;
        mean += value;
      }
    }
    mean /= Math.max(count, 1);
    let norm = 0;
    for (let i = 0; i < desc.length; i++) {
      desc[i] -= mean;
      norm += desc[i] * desc[i];
    }
    norm = Math.sqrt(norm) || 1;
    for (let i = 0; i < desc.length; i++) {
      desc[i] /= norm;
    }
    return desc;
  }

  function prepareCornersForTracking(corners, gray, w, h) {
    for (const corner of corners) {
      corner.descriptor = extractDescriptor(gray, w, h, Math.round(corner.x), Math.round(corner.y));
      corner.v3dIdx = -1;
    }
  }

  function ncc(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
      sum += a[i] * b[i];
    }
    return sum;
  }

  function matchCornerSets(current, previous, minScore) {
    if (!current.length || !previous.length) return [];
    const candidates = [];
    for (let ai = 0; ai < current.length; ai++) {
      let bestScore = minScore;
      let secondScore = -Infinity;
      let bestIdx = -1;
      for (let bi = 0; bi < previous.length; bi++) {
        const score = ncc(current[ai].descriptor, previous[bi].descriptor);
        if (score > bestScore) {
          secondScore = bestScore;
          bestScore = score;
          bestIdx = bi;
        } else if (score > secondScore) {
          secondScore = score;
        }
      }
      if (bestIdx >= 0 && bestScore - secondScore > 0.015) {
        candidates.push({ ai, bi: bestIdx, score: bestScore });
      }
    }
    candidates.sort((a, b) => b.score - a.score);
    const usedPrev = new Set();
    const matches = [];
    for (const match of candidates) {
      if (usedPrev.has(match.bi)) continue;
      usedPrev.add(match.bi);
      matches.push(match);
      if (matches.length >= 48) break;
    }
    return matches;
  }

  function extractSilhouette(gray, w, h) {
    const threshold = cfg.silhouetteMode === 'manual'
      ? cfg.manualThreshold
      : otsuThreshold(gray, w, h);
    const mask = new Uint8Array(w * h);
    for (let i = 0; i < gray.length; i++) {
      const value = gray[i] < threshold ? 1 : 0;
      mask[i] = cfg.invertSilhouette ? 1 - value : value;
    }
    if (cfg.morphType !== 'none' && cfg.morphIterations > 0) {
      applyMorphology(mask, w, h, cfg.morphType, cfg.morphIterations);
    }
    const outside = new Uint8Array(w * h);
    const queue = [];
    for (let x = 0; x < w; x++) {
      if (!mask[x]) {
        outside[x] = 1;
        queue.push(x, 0);
      }
      const bottom = (h - 1) * w + x;
      if (!mask[bottom]) {
        outside[bottom] = 1;
        queue.push(x, h - 1);
      }
    }
    for (let y = 0; y < h; y++) {
      const left = y * w;
      const right = y * w + w - 1;
      if (!mask[left]) {
        outside[left] = 1;
        queue.push(0, y);
      }
      if (!mask[right]) {
        outside[right] = 1;
        queue.push(w - 1, y);
      }
    }
    while (queue.length) {
      const qy = queue.pop();
      const qx = queue.pop();
      const neighbors = [
        [qx - 1, qy],
        [qx + 1, qy],
        [qx, qy - 1],
        [qx, qy + 1],
      ];
      for (const [nx, ny] of neighbors) {
        if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
        const ni = ny * w + nx;
        if (!outside[ni] && !mask[ni]) {
          outside[ni] = 1;
          queue.push(nx, ny);
        }
      }
    }
    const sil = new Uint8Array(w * h);
    for (let i = 0; i < gray.length; i++) {
      sil[i] = outside[i] ? 0 : 1;
    }
    return sil;
  }

  function otsuThreshold(gray, w, h) {
    const hist = new Int32Array(256);
    const total = w * h;
    for (let i = 0; i < total; i++) {
      hist[clamp(Math.round(gray[i]), 0, 255)]++;
    }
    let sum = 0;
    for (let i = 0; i < 256; i++) sum += i * hist[i];
    let sumB = 0;
    let wB = 0;
    let best = 0;
    let bestT = 128;
    for (let t = 0; t < 256; t++) {
      wB += hist[t];
      if (!wB) continue;
      const wF = total - wB;
      if (!wF) break;
      sumB += t * hist[t];
      const diff = sumB / wB - (sum - sumB) / wF;
      const between = wB * wF * diff * diff;
      if (between > best) {
        best = between;
        bestT = t;
      }
    }
    return bestT;
  }

  function applyMorphology(mask, w, h, type, iterations) {
    for (let iter = 0; iter < iterations; iter++) {
      if (type === 'dilate') dilate(mask, w, h);
      else if (type === 'erode') erode(mask, w, h);
      else if (type === 'open') {
        erode(mask, w, h);
        dilate(mask, w, h);
      } else if (type === 'close') {
        dilate(mask, w, h);
        erode(mask, w, h);
      }
    }
  }

  function dilate(mask, w, h) {
    const tmp = new Uint8Array(mask);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        if (tmp[y * w + x]) continue;
        if (tmp[(y - 1) * w + x] || tmp[(y + 1) * w + x] || tmp[y * w + x - 1] || tmp[y * w + x + 1]) {
          mask[y * w + x] = 1;
        }
      }
    }
  }

  function erode(mask, w, h) {
    const tmp = new Uint8Array(mask);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        if (!tmp[y * w + x]) continue;
        if (!tmp[(y - 1) * w + x] || !tmp[(y + 1) * w + x] || !tmp[y * w + x - 1] || !tmp[y * w + x + 1]) {
          mask[y * w + x] = 0;
        }
      }
    }
  }

  function measureSilhouette(sil, w, h) {
    let count = 0;
    let sumX = 0;
    let sumY = 0;
    let minX = w;
    let minY = h;
    let maxX = 0;
    let maxY = 0;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        if (!sil[y * w + x]) continue;
        count++;
        sumX += x;
        sumY += y;
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
    if (!count) {
      return {
        coverage: 0,
        cx: w / 2,
        cy: h / 2,
        normCx: 0,
        normCy: 0,
        bboxW: 0,
        bboxH: 0,
      };
    }
    const cx = sumX / count;
    const cy = sumY / count;
    return {
      coverage: count / (w * h),
      cx,
      cy,
      normCx: (cx - w / 2) / w,
      normCy: (cy - h / 2) / h,
      bboxW: (maxX - minX + 1) / w,
      bboxH: (maxY - minY + 1) / h,
    };
  }

  function computeImageQuality(edgeMap, silStats, w, h) {
    let edgePixels = 0;
    const cx = Math.floor(w / 2);
    const cy = Math.floor(h / 2);
    const radius = Math.floor(Math.min(w, h) * 0.32);
    let total = 0;
    for (let y = cy - radius; y <= cy + radius; y++) {
      for (let x = cx - radius; x <= cx + radius; x++) {
        if (x < 0 || y < 0 || x >= w || y >= h) continue;
        total++;
        if (edgeMap[y * w + x]) edgePixels++;
      }
    }
    const edgeDensity = total ? edgePixels / total : 0;
    const centeredness = 1 - clamp(Math.hypot(silStats.normCx * 1.6, silStats.normCy * 1.6), 0, 1);
    const coverageScore = 1 - clamp(Math.abs(silStats.coverage - cfg.targetCoverage) / Math.max(cfg.targetCoverage, 0.001), 0, 1);
    return Math.round(
      clamp(edgeDensity * 2.8, 0, 1) * 40 +
      coverageScore * 25 +
      centeredness * 35
    );
  }

  function getSmoothedOrientation() {
    let yaw = 0;
    let pitch = 0;
    if (typeof CameraModule !== 'undefined' && CameraModule.getOrientation) {
      const o = CameraModule.getOrientation();
      const alpha = ((o.alpha || 0) * Math.PI) / 180;
      const beta = (((o.beta || 90) - 90) * Math.PI) / 180;
      if (!sensorReady) {
        sensorYaw = alpha;
        sensorPitch = beta;
        lastSensorAlpha = alpha;
        sensorReady = true;
      } else {
        let delta = alpha - lastSensorAlpha;
        if (delta > Math.PI) delta -= Math.PI * 2;
        else if (delta < -Math.PI) delta += Math.PI * 2;
        sensorYaw += delta;
        sensorPitch = beta;
        lastSensorAlpha = alpha;
      }
      yaw = sensorYaw;
      pitch = sensorPitch;
    } else {
      yaw = performance.now() / 1000 * 0.3;
      pitch = 0;
    }
    if (!cfg.smoothOrientation) {
      return { yaw, pitch };
    }
    poseState.yaw = mix(poseState.yaw, yaw + poseState.yawBias, cfg.smoothFactor);
    poseState.pitch = mix(poseState.pitch, pitch + poseState.pitchBias, cfg.smoothFactor);
    return { yaw: poseState.yaw, pitch: poseState.pitch };
  }

  function estimateVisualDrift(corners, silStats, w, h) {
    if (!cfg.poseAssist || !lastTrackingFrame) {
      return { matchCount: 0, confidence: 0, yawDelta: 0, pitchDelta: 0, offsetX: 0, offsetY: 0, radiusDelta: 0 };
    }
    const matches = matchCornerSets(corners, lastTrackingFrame.corners, cfg.matchThreshold);
    if (!matches.length) {
      return { matchCount: 0, confidence: 0, yawDelta: 0, pitchDelta: 0, offsetX: 0, offsetY: 0, radiusDelta: 0 };
    }
    const dx = [];
    const dy = [];
    for (const match of matches) {
      dx.push(corners[match.ai].x - lastTrackingFrame.corners[match.bi].x);
      dy.push(corners[match.ai].y - lastTrackingFrame.corners[match.bi].y);
    }
    const medianDx = median(dx);
    const medianDy = median(dy);
    const coverageDelta = silStats.coverage - lastTrackingFrame.silStats.coverage;
    const confidence = clamp(matches.length / 18, 0, 1);
    const fov = cfg.fovDeg * Math.PI / 180;
    return {
      matchCount: matches.length,
      confidence,
      yawDelta: -medianDx / w * fov * 0.55,
      pitchDelta: medianDy / h * 0.32,
      offsetX: -medianDx / w * cfg.voxelWorld * 0.22,
      offsetY: medianDy / h * cfg.voxelWorld * 0.18,
      radiusDelta: -coverageDelta * cfg.cameraR * 0.9,
    };
  }

  function buildPose(yaw, pitch, radius, offsetX, offsetY) {
    const lift = Math.sin(pitch) * radius * 0.45;
    return {
      yaw,
      pitch,
      radius,
      offsetX,
      offsetY,
      target: [0, 0, 0],
      position: [
        Math.sin(yaw) * radius + offsetX,
        lift + offsetY,
        Math.cos(yaw) * radius,
      ],
      confidence: poseState ? poseState.confidence : 0,
    };
  }

  function estimatePose(corners, silStats, w, h, imageQuality) {
    const sensor = getSmoothedOrientation();
    const visual = estimateVisualDrift(corners, silStats, w, h);
    const gain = cfg.poseAssist ? cfg.poseAssistGain : 0;
    const targetRadius = clamp(
      cfg.cameraR + (cfg.targetCoverage - silStats.coverage) * cfg.cameraR * 2.1 + visual.radiusDelta,
      cfg.cameraR * 0.7,
      cfg.cameraR * 1.5
    );
    const targetOffsetX = clamp(
      -silStats.normCx * targetRadius * 1.15 + visual.offsetX * gain,
      -cfg.voxelWorld * 0.32,
      cfg.voxelWorld * 0.32
    );
    const targetOffsetY = clamp(
      silStats.normCy * targetRadius * 0.85 + visual.offsetY * gain,
      -cfg.voxelWorld * 0.24,
      cfg.voxelWorld * 0.24
    );
    poseState.yawBias = mix(poseState.yawBias, visual.yawDelta * gain, 0.18);
    poseState.pitchBias = mix(poseState.pitchBias, visual.pitchDelta * gain, 0.18);
    poseState.radius = mix(poseState.radius, targetRadius, 0.16);
    poseState.offsetX = mix(poseState.offsetX, targetOffsetX, 0.18);
    poseState.offsetY = mix(poseState.offsetY, targetOffsetY, 0.18);
    poseState.confidence = clamp(
      imageQuality / 100 * 0.45 +
      clamp(1 - Math.abs(silStats.coverage - cfg.targetCoverage) / Math.max(cfg.targetCoverage, 0.001), 0, 1) * 0.2 +
      visual.confidence * 0.35,
      0,
      1
    );
    let pose = buildPose(sensor.yaw + poseState.yawBias, sensor.pitch + poseState.pitchBias, poseState.radius, poseState.offsetX, poseState.offsetY);
    pose.confidence = poseState.confidence;
    const refine = refinePoseAgainstMap(pose, silStats, w, h);
    pose = refine.pose;
    poseLock = refine.lock;
    poseState.yawBias = mix(poseState.yawBias, pose.yaw - sensor.yaw, 0.16);
    poseState.pitchBias = mix(poseState.pitchBias, pose.pitch - sensor.pitch, 0.16);
    poseState.radius = mix(poseState.radius, pose.radius, 0.16);
    poseState.offsetX = mix(poseState.offsetX, pose.offsetX, 0.16);
    poseState.offsetY = mix(poseState.offsetY, pose.offsetY, 0.16);
    poseState.confidence = clamp(poseState.confidence * 0.68 + refine.lock * 0.32, 0, 1);
    pose.confidence = poseState.confidence;
    return { pose, visual, refine };
  }

  function shouldIntegrateKeyframe(pose, silStats, frameQuality) {
    if (frameQuality < cfg.keyframeMinQuality) return false;
    if (silStats.coverage < 0.03) return false;
    if (!lastIntegratedPose) return true;
    const rotationDelta = Math.abs(pose.yaw - lastIntegratedPose.yaw);
    const translationDelta = distance3(pose.position, lastIntegratedPose.position);
    const centroidDelta = lastTrackingFrame
      ? Math.hypot(silStats.normCx - lastTrackingFrame.silStats.normCx, silStats.normCy - lastTrackingFrame.silStats.normCy)
      : 0;
    return (
      rotationDelta >= cfg.keyframeMinRotation ||
      translationDelta >= cfg.keyframeMinTranslation ||
      centroidDelta >= cfg.keyframeMinCentroid
    );
  }

  function storeTrackingFrame(corners, silStats, pose) {
    lastTrackingFrame = {
      corners: snapshotCorners(corners),
      silStats: { ...silStats },
      pose: clonePose(pose),
    };
  }

  function snapshotCorners(corners) {
    return corners.map(corner => ({
      x: corner.x,
      y: corner.y,
      angle: corner.angle,
      a1: corner.a1,
      a2: corner.a2,
      descriptor: new Float32Array(corner.descriptor),
      v3dIdx: corner.v3dIdx,
    }));
  }

  function clonePose(pose) {
    return {
      yaw: pose.yaw,
      pitch: pose.pitch,
      radius: pose.radius,
      offsetX: pose.offsetX,
      offsetY: pose.offsetY,
      confidence: pose.confidence,
      position: [pose.position[0], pose.position[1], pose.position[2]],
      target: [pose.target[0], pose.target[1], pose.target[2]],
    };
  }

  function collectPoseAnchors(limit) {
    const anchors = [];
    for (const surfel of surfels) {
      if (surfel.confidence < cfg.surfelMinConfidence) continue;
      anchors.push({
        pos: surfel.pos,
        weight: clamp(surfel.confidence / 4, 0.6, 1.8),
      });
    }
    if (anchors.length < limit * 0.6) {
      const voxelAnchors = collectRenderableVoxels(limit, false);
      for (const anchor of voxelAnchors) {
        anchors.push({
          pos: [anchor.x, anchor.y, anchor.z],
          weight: clamp(anchor.confidence * 1.2, 0.35, 1.4),
        });
      }
    }
    if (anchors.length <= limit) return anchors;
    const stride = Math.ceil(anchors.length / limit);
    const sampled = [];
    for (let i = 0; i < anchors.length; i += stride) {
      sampled.push(anchors[i]);
    }
    return sampled;
  }

  function scorePoseAgainstMap(pose, anchors, silStats, w, h) {
    if (!anchors.length) return -Infinity;
    let support = 0;
    let penalty = 0;
    let sumX = 0;
    let sumY = 0;
    let sumW = 0;
    let projected = 0;
    for (const anchor of anchors) {
      const proj = projectPoint(anchor.pos, pose, w, h);
      if (!proj || proj.px < 0 || proj.px >= w || proj.py < 0 || proj.py >= h) {
        penalty += anchor.weight * 0.8;
        continue;
      }
      projected++;
      sumX += proj.px * anchor.weight;
      sumY += proj.py * anchor.weight;
      sumW += anchor.weight;
      const dx = (proj.px - silStats.cx) / w;
      const dy = (proj.py - silStats.cy) / h;
      const dist = Math.hypot(dx, dy);
      support += anchor.weight * Math.max(0, 1 - dist * 4.4);
      if (dist > 0.34) {
        penalty += anchor.weight * (dist - 0.34) * 2.8;
      }
    }
    if (!projected || !sumW) return -Infinity;
    const centroidDx = (sumX / sumW - silStats.cx) / w;
    const centroidDy = (sumY / sumW - silStats.cy) / h;
    const centroidPenalty = Math.hypot(centroidDx, centroidDy);
    const coveragePenalty = Math.abs(projected / anchors.length - clamp(silStats.coverage / Math.max(cfg.targetCoverage, 0.001), 0, 1.2));
    return (
      support * cfg.poseRefineMapWeight -
      penalty -
      centroidPenalty * anchors.length * 0.9 -
      coveragePenalty * anchors.length * 0.45
    );
  }

  function refinePoseAgainstMap(initialPose, silStats, w, h) {
    if (!cfg.poseRefineEnabled) {
      return { pose: clonePose(initialPose), lock: 0 };
    }
    const anchors = collectPoseAnchors(cfg.poseRefineSamples);
    if (anchors.length < 12) {
      return { pose: clonePose(initialPose), lock: 0 };
    }
    let bestPose = clonePose(initialPose);
    let bestScore = scorePoseAgainstMap(bestPose, anchors, silStats, w, h);
    const baseScore = bestScore;
    let stepYaw = cfg.poseRefineStepYaw;
    let stepOffset = cfg.poseRefineStepOffset;
    let stepRadius = cfg.poseRefineStepRadius;
    for (let iter = 0; iter < cfg.poseRefineIterations; iter++) {
      let improved = false;
      const deltas = [
        [stepYaw, 0, 0, 0, 0],
        [-stepYaw, 0, 0, 0, 0],
        [0, stepYaw * 0.65, 0, 0, 0],
        [0, -stepYaw * 0.65, 0, 0, 0],
        [0, 0, 0, stepOffset, 0],
        [0, 0, 0, -stepOffset, 0],
        [0, 0, stepOffset * 0.8, 0, 0],
        [0, 0, -stepOffset * 0.8, 0, 0],
        [0, 0, 0, 0, stepRadius],
        [0, 0, 0, 0, -stepRadius],
      ];
      for (const [dyaw, dpitch, dox, doy, dr] of deltas) {
        const candidate = buildPose(
          bestPose.yaw + dyaw,
          clamp(bestPose.pitch + dpitch, -0.75, 0.75),
          clamp(bestPose.radius + dr, cfg.cameraR * 0.7, cfg.cameraR * 1.5),
          clamp(bestPose.offsetX + dox, -cfg.voxelWorld * 0.35, cfg.voxelWorld * 0.35),
          clamp(bestPose.offsetY + doy, -cfg.voxelWorld * 0.25, cfg.voxelWorld * 0.25)
        );
        const score = scorePoseAgainstMap(candidate, anchors, silStats, w, h);
        if (score > bestScore) {
          bestPose = candidate;
          bestScore = score;
          improved = true;
        }
      }
      if (!improved) {
        stepYaw *= 0.55;
        stepOffset *= 0.55;
        stepRadius *= 0.6;
      }
    }
    const improvement = Number.isFinite(baseScore) ? Math.max(0, bestScore - baseScore) : 0;
    const lock = clamp(
      anchors.length / Math.max(cfg.poseRefineSamples, 1) * 0.3 +
      Math.max(0, bestScore) / Math.max(anchors.length, 1) * 0.55 +
      improvement / Math.max(anchors.length, 1) * 0.35,
      0,
      1
    );
    bestPose.confidence = clamp(bestPose.confidence || 0, 0, 1);
    return { pose: bestPose, lock };
  }

  function processVertexTracking(corners, lines, w, h, pose, imgData) {
    const adjacency = buildAdjacency(corners, lines);
    for (const hist of keyframes) {
      if (Math.abs(pose.yaw - hist.pose.yaw) < cfg.minAngleDiff) continue;
      const matches = matchCornerSets(corners, hist.corners, cfg.matchThreshold);
      for (const match of matches) {
        const cur = corners[match.ai];
        const prev = hist.corners[match.bi];
        if (cur.v3dIdx >= 0 && prev.v3dIdx >= 0 && cur.v3dIdx !== prev.v3dIdx) continue;
        const pos = triangulate2Views(cur.x, cur.y, pose, prev.x, prev.y, hist.pose, w, h);
        if (!pos) continue;
        if (computeBaselineAngleDeg(cur.x, cur.y, pose, prev.x, prev.y, hist.pose, w, h) < cfg.minBaselineDeg) continue;
        const dist = length3(pos);
        if (dist > cfg.voxelWorld * 0.85) continue;
        const reprojA = reprojectionError(pos, cur.x, cur.y, pose, w, h);
        const reprojB = reprojectionError(pos, prev.x, prev.y, hist.pose, w, h);
        if (Math.max(reprojA, reprojB) > cfg.reprojectionThresholdPx) continue;
        const px = clamp(Math.round(cur.x), 0, w - 1);
        const py = clamp(Math.round(cur.y), 0, h - 1);
        const pi = (py * w + px) * 4;
        const col = [
          imgData.data[pi] / 255,
          imgData.data[pi + 1] / 255,
          imgData.data[pi + 2] / 255,
        ];
        if (prev.v3dIdx >= 0) {
          const vertex = vertices3D[prev.v3dIdx];
          if (vertex.confidence >= 2 && distance3(vertex.pos, pos) > cfg.vertexMergeDist * 2.2) continue;
          const f = 1 / (vertex.confidence + 1);
          vertex.pos[0] += (pos[0] - vertex.pos[0]) * f;
          vertex.pos[1] += (pos[1] - vertex.pos[1]) * f;
          vertex.pos[2] += (pos[2] - vertex.pos[2]) * f;
          vertex.col[0] += (col[0] - vertex.col[0]) * f;
          vertex.col[1] += (col[1] - vertex.col[1]) * f;
          vertex.col[2] += (col[2] - vertex.col[2]) * f;
          vertex.confidence++;
          cur.v3dIdx = prev.v3dIdx;
          upsertSurfel(vertex.pos, vertex.col, vertex.confidence);
        } else if (cur.v3dIdx >= 0) {
          prev.v3dIdx = cur.v3dIdx;
          const vertex = vertices3D[cur.v3dIdx];
          if (vertex) upsertSurfel(vertex.pos, vertex.col, vertex.confidence);
        } else {
          const nearbyIdx = findNearbyVertexIndex(pos, cfg.vertexMergeDist);
          const idx = nearbyIdx >= 0 ? nearbyIdx : vertices3D.length;
          cur.v3dIdx = idx;
          prev.v3dIdx = idx;
          if (nearbyIdx >= 0) {
            const vertex = vertices3D[nearbyIdx];
            const f = 1 / (vertex.confidence + 1);
            vertex.pos[0] += (pos[0] - vertex.pos[0]) * f;
            vertex.pos[1] += (pos[1] - vertex.pos[1]) * f;
            vertex.pos[2] += (pos[2] - vertex.pos[2]) * f;
            vertex.col[0] += (col[0] - vertex.col[0]) * f;
            vertex.col[1] += (col[1] - vertex.col[1]) * f;
            vertex.col[2] += (col[2] - vertex.col[2]) * f;
            vertex.confidence++;
            upsertSurfel(vertex.pos, vertex.col, vertex.confidence);
          } else {
            vertices3D.push({ pos, col, confidence: 1 });
            upsertSurfel(pos, col, 1);
          }
        }
        wireframeDirty = true;
      }
    }
    for (let i = 0; i < corners.length; i++) {
      if (corners[i].v3dIdx < 0) continue;
      const neighbors = adjacency.get(i);
      if (!neighbors) continue;
      for (const j of neighbors) {
        if (corners[j].v3dIdx < 0) continue;
        const a = Math.min(corners[i].v3dIdx, corners[j].v3dIdx);
        const b = Math.max(corners[i].v3dIdx, corners[j].v3dIdx);
        const key = a * 100000 + b;
        if (!edges3D.has(key)) {
          edges3D.add(key);
          wireframeDirty = true;
        }
      }
    }
  }

  function buildAdjacency(corners, lines) {
    const adjacency = new Map();
    for (let i = 0; i < corners.length; i++) {
      adjacency.set(i, new Set());
    }
    for (const line of lines) {
      let ci = -1;
      let cj = -1;
      for (let k = 0; k < corners.length; k++) {
        const corner = corners[k];
        const d1 = Math.sqrt((corner.x - line.x1) ** 2 + (corner.y - line.y1) ** 2);
        const d2 = Math.sqrt((corner.x - line.x2) ** 2 + (corner.y - line.y2) ** 2);
        if (d1 < cfg.cornerDist * 2.5) ci = k;
        if (d2 < cfg.cornerDist * 2.5) cj = k;
      }
      if (ci >= 0 && cj >= 0 && ci !== cj) {
        adjacency.get(ci).add(cj);
        adjacency.get(cj).add(ci);
      }
    }
    return adjacency;
  }

  function rememberKeyframe(corners, pose, silStats) {
    keyframes.push({
      corners: snapshotCorners(corners),
      pose: clonePose(pose),
      silStats: { ...silStats },
    });
    vertexHistory.push(keyframes[keyframes.length - 1]);
    if (keyframes.length > cfg.maxHistory) keyframes.shift();
    if (vertexHistory.length > cfg.maxHistory) vertexHistory.shift();
    keyframeCount++;
  }

  function findNearbyVertexIndex(pos, maxDist) {
    let bestIdx = -1;
    let bestDist = maxDist;
    for (let i = 0; i < vertices3D.length; i++) {
      const d = distance3(vertices3D[i].pos, pos);
      if (d < bestDist) {
        bestDist = d;
        bestIdx = i;
      }
    }
    return bestIdx;
  }

  function upsertSurfel(pos, col, confidence) {
    const cell = cfg.surfelCell;
    const key = [
      Math.floor(pos[0] / cell),
      Math.floor(pos[1] / cell),
      Math.floor(pos[2] / cell),
    ].join(',');
    if (surfelIndex.has(key)) {
      const surfel = surfels[surfelIndex.get(key)];
      const weight = Math.max(0.5, confidence);
      const f = weight / (surfel.confidence + weight);
      surfel.pos[0] += (pos[0] - surfel.pos[0]) * f;
      surfel.pos[1] += (pos[1] - surfel.pos[1]) * f;
      surfel.pos[2] += (pos[2] - surfel.pos[2]) * f;
      surfel.col[0] += (col[0] - surfel.col[0]) * f;
      surfel.col[1] += (col[1] - surfel.col[1]) * f;
      surfel.col[2] += (col[2] - surfel.col[2]) * f;
      surfel.confidence += weight * 0.6;
      surfel.hits++;
    } else {
      surfelIndex.set(key, surfels.length);
      surfels.push({
        pos: [pos[0], pos[1], pos[2]],
        col: [col[0], col[1], col[2]],
        confidence: Math.max(1, confidence),
        hits: 1,
      });
    }
    wireframeDirty = true;
    voxelsDirty = true;
  }

  function rebuildSurfelsFromVertices() {
    surfels = [];
    surfelIndex = new Map();
    for (const vertex of vertices3D) {
      upsertSurfel(vertex.pos, vertex.col, vertex.confidence);
    }
  }

  function buildCameraBasis(pose) {
    const forward = normalize3(sub3(pose.target, pose.position));
    let right = cross3(forward, [0, 1, 0]);
    if (length3(right) < 1e-5) right = [1, 0, 0];
    right = normalize3(right);
    const up = normalize3(cross3(right, forward));
    return { forward, right, up };
  }

  function backprojectRay(px, py, pose, w, h) {
    const fov = cfg.fovDeg * Math.PI / 180;
    const fx = w / (2 * Math.tan(fov / 2));
    const fy = fx;
    const cx = w / 2;
    const cy = h / 2;
    const x = (px - cx) / fx;
    const y = -(py - cy) / fy;
    const basis = buildCameraBasis(pose);
    const dir = normalize3([
      basis.right[0] * x + basis.up[0] * y + basis.forward[0],
      basis.right[1] * x + basis.up[1] * y + basis.forward[1],
      basis.right[2] * x + basis.up[2] * y + basis.forward[2],
    ]);
    return { origin: pose.position, dir };
  }

  function projectPoint(point, pose, w, h) {
    const fov = cfg.fovDeg * Math.PI / 180;
    const fx = w / (2 * Math.tan(fov / 2));
    const fy = fx;
    const cx = w / 2;
    const cy = h / 2;
    const basis = buildCameraBasis(pose);
    const rel = sub3(point, pose.position);
    const camX = dot3(rel, basis.right);
    const camY = dot3(rel, basis.up);
    const camZ = dot3(rel, basis.forward);
    if (camZ <= 0.04) return null;
    return {
      px: Math.round(fx * camX / camZ + cx),
      py: Math.round(cy - fy * camY / camZ),
      depth: camZ,
    };
  }

  function reprojectionError(point, px, py, pose, w, h) {
    const proj = projectPoint(point, pose, w, h);
    if (!proj) return Infinity;
    return Math.hypot(proj.px - px, proj.py - py);
  }

  function computeBaselineAngleDeg(px1, py1, pose1, px2, py2, pose2, w, h) {
    const ray1 = backprojectRay(px1, py1, pose1, w, h);
    const ray2 = backprojectRay(px2, py2, pose2, w, h);
    const cosine = clamp(dot3(ray1.dir, ray2.dir), -1, 1);
    return Math.acos(cosine) * 180 / Math.PI;
  }

  function triangulate2Views(px1, py1, pose1, px2, py2, pose2, w, h) {
    const ray1 = backprojectRay(px1, py1, pose1, w, h);
    const ray2 = backprojectRay(px2, py2, pose2, w, h);
    const w0 = sub3(ray1.origin, ray2.origin);
    const a = dot3(ray1.dir, ray1.dir);
    const b = dot3(ray1.dir, ray2.dir);
    const c = dot3(ray2.dir, ray2.dir);
    const d = dot3(ray1.dir, w0);
    const e = dot3(ray2.dir, w0);
    const den = a * c - b * b;
    if (Math.abs(den) < 1e-8) return null;
    const sc = (b * e - c * d) / den;
    const tc = (a * e - b * d) / den;
    if (sc <= 0.01 || tc <= 0.01) return null;
    const p1 = add3(ray1.origin, scale3(ray1.dir, sc));
    const p2 = add3(ray2.origin, scale3(ray2.dir, tc));
    if (distance3(p1, p2) > 0.18) return null;
    return [
      (p1[0] + p2[0]) * 0.5,
      (p1[1] + p2[1]) * 0.5,
      (p1[2] + p2[2]) * 0.5,
    ];
  }

  function initVoxelGrid() {
    const n = cfg.voxelRes ** 3;
    voxelState = new Uint8Array(n);
    voxelState.fill(1);
    voxelEvidence = new Float32Array(n);
    voxelInside = new Uint16Array(n);
    voxelOutside = new Uint16Array(n);
    voxelColorAcc = new Float32Array(n * 3);
    voxelColorWeight = new Float32Array(n);
    previewCount = 0;
    voxelsDirty = true;
  }

  function carveVisualHull(sil, sw, sh, pose) {
    if (!voxelState) return;
    const R = cfg.voxelRes;
    const step = cfg.voxelWorld / R;
    const half = cfg.voxelWorld / 2;
    let changed = false;
    for (let i = 0; i < R; i++) {
      const wx = i * step - half;
      for (let j = 0; j < R; j++) {
        const wy = j * step - half;
        for (let k = 0; k < R; k++) {
          const idx = i * R * R + j * R + k;
          if (!voxelState[idx]) continue;
          const wz = k * step - half;
          const proj = projectPoint([wx, wy, wz], pose, sw, sh);
          if (!proj || proj.px < 0 || proj.px >= sw || proj.py < 0 || proj.py >= sh) {
            voxelEvidence[idx] -= cfg.voxelMissWeight * 0.55;
            voxelOutside[idx]++;
          } else if (sil[proj.py * sw + proj.px]) {
            voxelEvidence[idx] += cfg.voxelHitWeight;
            voxelInside[idx]++;
          } else {
            voxelEvidence[idx] -= cfg.voxelMissWeight;
            voxelOutside[idx]++;
          }
          if (voxelEvidence[idx] < -cfg.voxelRemoveThreshold && voxelInside[idx] < cfg.voxelMinObservations) {
            voxelState[idx] = 0;
            changed = true;
          } else if (voxelEvidence[idx] > cfg.voxelSupportThreshold || voxelOutside[idx] > 0) {
            changed = true;
          }
        }
      }
    }
    if (changed) voxelsDirty = true;
  }

  function colorVoxels(imgData, w, h, pose, sil) {
    if (!voxelState) return;
    const R = cfg.voxelRes;
    const step = cfg.voxelWorld / R;
    const half = cfg.voxelWorld / 2;
    const data = imgData.data;
    let changed = false;
    for (let i = 0; i < R; i++) {
      const wx = i * step - half;
      for (let j = 0; j < R; j++) {
        const wy = j * step - half;
        for (let k = 0; k < R; k++) {
          const idx = i * R * R + j * R + k;
          if (!voxelState[idx]) continue;
          if (voxelEvidence[idx] < cfg.voxelSupportThreshold * 0.35) continue;
          const wz = k * step - half;
          const proj = projectPoint([wx, wy, wz], pose, w, h);
          if (!proj || proj.px < 0 || proj.px >= w || proj.py < 0 || proj.py >= h) continue;
          if (!sil[proj.py * w + proj.px]) continue;
          const pi = (proj.py * w + proj.px) * 4;
          const base = idx * 3;
          const weight = 0.25 + Math.min(1.4, voxelInside[idx] * 0.08);
          voxelColorAcc[base] += data[pi] / 255 * weight;
          voxelColorAcc[base + 1] += data[pi + 1] / 255 * weight;
          voxelColorAcc[base + 2] += data[pi + 2] / 255 * weight;
          voxelColorWeight[idx] += weight;
          changed = true;
        }
      }
    }
    if (changed) voxelsDirty = true;
  }

  function isRenderableVoxel(idx) {
    return voxelState[idx] &&
      voxelEvidence[idx] >= cfg.voxelSupportThreshold &&
      voxelInside[idx] >= cfg.voxelMinObservations;
  }

  function collectRenderableVoxels(maxPoints, includeColor = true) {
    if (!voxelState) {
      return [];
    }
    const R = cfg.voxelRes;
    const step = cfg.voxelWorld / R;
    const half = cfg.voxelWorld / 2;
    let count = 0;
    for (let i = 0; i < voxelState.length; i++) {
      if (isRenderableVoxel(i)) count++;
    }
    if (!count) return [];
    const stride = maxPoints !== Infinity && count > maxPoints
      ? Math.ceil(count / maxPoints)
      : 1;
    const entries = [];
    let seen = 0;
    for (let i = 0; i < R; i++) {
      const wx = i * step - half;
      for (let j = 0; j < R; j++) {
        const wy = j * step - half;
        for (let k = 0; k < R; k++) {
          const idx = i * R * R + j * R + k;
          if (!isRenderableVoxel(idx)) continue;
          if (seen++ % stride !== 0) continue;
          const base = idx * 3;
          const colorWeight = voxelColorWeight[idx] || 1;
          const confidence = clamp(
            voxelEvidence[idx] / (cfg.voxelSupportThreshold + 2) +
            voxelInside[idx] / Math.max(cfg.voxelMinObservations + 4, 1),
            0,
            1
          );
          entries.push({
            x: wx,
            y: wy,
            z: k * step - half,
            r: includeColor ? (colorWeight > 0 ? voxelColorAcc[base] / colorWeight : 0.42) : 0.42,
            g: includeColor ? (colorWeight > 0 ? voxelColorAcc[base + 1] / colorWeight : 0.72) : 0.72,
            b: includeColor ? (colorWeight > 0 ? voxelColorAcc[base + 2] / colorWeight : 0.92) : 0.92,
            confidence,
          });
        }
      }
    }
    return entries;
  }

  function collectRenderableSurfels(maxPoints) {
    const entries = [];
    for (const surfel of surfels) {
      if (surfel.confidence < cfg.surfelMinConfidence) continue;
      entries.push({
        x: surfel.pos[0],
        y: surfel.pos[1],
        z: surfel.pos[2],
        r: surfel.col[0],
        g: surfel.col[1],
        b: surfel.col[2],
        confidence: clamp(surfel.confidence / 8, 0, 1),
      });
    }
    if (maxPoints === Infinity || entries.length <= maxPoints) return entries;
    const stride = Math.ceil(entries.length / maxPoints);
    const sampled = [];
    for (let i = 0; i < entries.length; i += stride) {
      sampled.push(entries[i]);
    }
    return sampled;
  }

  function collectRenderableMapEntries(maxPoints) {
    const surfelBudget = maxPoints === Infinity ? Infinity : Math.max(300, Math.round(maxPoints * 0.35));
    const voxelBudget = maxPoints === Infinity ? Infinity : Math.max(1000, maxPoints - surfelBudget);
    let entries = collectRenderableVoxels(voxelBudget, true).concat(collectRenderableSurfels(surfelBudget));
    if (maxPoints !== Infinity && entries.length > maxPoints) {
      const stride = Math.ceil(entries.length / maxPoints);
      entries = entries.filter((_, idx) => idx % stride === 0);
    }
    previewConfidence = entries.length
      ? entries.reduce((sum, entry) => sum + (entry.confidence || 0), 0) / entries.length
      : 0;
    return entries;
  }

  function setup3D() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080c16);
    cam3d = new THREE.PerspectiveCamera(50, 1, 0.01, 50);
    cam3d.position.set(0, 0.4, 2);
    renderer = new THREE.WebGLRenderer({ canvas: previewCanvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    scene.add(new THREE.GridHelper(2, 10, 0x162030, 0x0e1520));
    scene.add(new THREE.AmbientLight(0xffffff, 0.6));
    const light = new THREE.DirectionalLight(0xffffff, 0.5);
    light.position.set(1, 2, 1);
    scene.add(light);
    ensurePreviewObject();
  }

  function ensurePreviewObject() {
    if (previewPoints) return;
    previewGeometry = new THREE.BufferGeometry();
    previewGeometry.setAttribute('position', new THREE.BufferAttribute(previewPositions, 3));
    previewGeometry.setAttribute('color', new THREE.BufferAttribute(previewColors, 3));
    previewGeometry.setDrawRange(0, 0);
    previewMaterial = new THREE.PointsMaterial({
      size: Math.max(cfg.voxelWorld / cfg.voxelRes * 0.92, 0.018),
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.95,
    });
    previewPoints = new THREE.Points(previewGeometry, previewMaterial);
    previewPoints.visible = cfg.showVoxels;
    scene.add(previewPoints);
  }

  function ensurePreviewCapacity(count) {
    if (count <= previewCapacity) return;
    previewCapacity = Math.max(count, Math.ceil(previewCapacity * 1.5), 2048);
    previewPositions = new Float32Array(previewCapacity * 3);
    previewColors = new Float32Array(previewCapacity * 3);
    previewGeometry.setAttribute('position', new THREE.BufferAttribute(previewPositions, 3));
    previewGeometry.setAttribute('color', new THREE.BufferAttribute(previewColors, 3));
  }

  function update3DPreview() {
    if (!previewPoints || !cfg.showVoxels) {
      if (previewPoints) previewPoints.visible = false;
      return;
    }
    previewPoints.visible = true;
    if (!voxelsDirty) return;
    const entries = collectRenderableMapEntries(cfg.renderBudget);
    ensurePreviewCapacity(entries.length);
    for (let i = 0; i < entries.length; i++) {
      const entry = entries[i];
      previewPositions[i * 3] = entry.x;
      previewPositions[i * 3 + 1] = entry.y;
      previewPositions[i * 3 + 2] = entry.z;
      previewColors[i * 3] = entry.r;
      previewColors[i * 3 + 1] = entry.g;
      previewColors[i * 3 + 2] = entry.b;
    }
    previewGeometry.attributes.position.needsUpdate = true;
    previewGeometry.attributes.color.needsUpdate = true;
    previewGeometry.setDrawRange(0, entries.length);
    previewGeometry.computeBoundingSphere();
    previewMaterial.size = Math.max(cfg.voxelWorld / cfg.voxelRes * 0.92, 0.018);
    previewCount = entries.length;
    voxelsDirty = false;
  }

  function updateWireframePreview() {
    if (!wireframeDirty) return;
    wireframeDirty = false;
    if (surfelObj) {
      scene.remove(surfelObj);
      surfelObj.geometry.dispose();
      surfelObj.material.dispose();
      surfelObj = null;
    }
    if (wireVertexObj) {
      scene.remove(wireVertexObj);
      wireVertexObj.geometry.dispose();
      wireVertexObj.material.dispose();
      wireVertexObj = null;
    }
    if (wireEdgeObj) {
      scene.remove(wireEdgeObj);
      wireEdgeObj.geometry.dispose();
      wireEdgeObj.material.dispose();
      wireEdgeObj = null;
    }
    if (cfg.showWireframe && vertices3D.length) {
      const vertexPos = [];
      const vertexCol = [];
      for (const vertex of vertices3D) {
        vertexPos.push(vertex.pos[0], vertex.pos[1], vertex.pos[2]);
        vertexCol.push(vertex.col[0], vertex.col[1], vertex.col[2]);
      }
      const vGeo = new THREE.BufferGeometry();
      vGeo.setAttribute('position', new THREE.Float32BufferAttribute(vertexPos, 3));
      vGeo.setAttribute('color', new THREE.Float32BufferAttribute(vertexCol, 3));
      wireVertexObj = new THREE.Points(
        vGeo,
        new THREE.PointsMaterial({ size: 0.035, vertexColors: true, sizeAttenuation: true })
      );
      scene.add(wireVertexObj);
    }
    if (cfg.showSurfels && surfels.length) {
      const surfelPos = [];
      const surfelCol = [];
      for (const surfel of surfels) {
        if (surfel.confidence < cfg.surfelMinConfidence) continue;
        surfelPos.push(surfel.pos[0], surfel.pos[1], surfel.pos[2]);
        surfelCol.push(surfel.col[0], surfel.col[1], surfel.col[2]);
      }
      if (surfelPos.length) {
        const sGeo = new THREE.BufferGeometry();
        sGeo.setAttribute('position', new THREE.Float32BufferAttribute(surfelPos, 3));
        sGeo.setAttribute('color', new THREE.Float32BufferAttribute(surfelCol, 3));
        surfelObj = new THREE.Points(
          sGeo,
          new THREE.PointsMaterial({
            size: Math.max(cfg.surfelCell * 0.8, 0.02),
            vertexColors: true,
            transparent: true,
            opacity: 0.8,
            sizeAttenuation: true,
          })
        );
        scene.add(surfelObj);
      }
    }
    if (!cfg.showWireframe || !vertices3D.length) return;
    if (!edges3D.size) return;
    const edgePos = [];
    for (const key of edges3D) {
      const a = Math.floor(key / 100000);
      const b = key % 100000;
      if (a >= vertices3D.length || b >= vertices3D.length) continue;
      edgePos.push(vertices3D[a].pos[0], vertices3D[a].pos[1], vertices3D[a].pos[2]);
      edgePos.push(vertices3D[b].pos[0], vertices3D[b].pos[1], vertices3D[b].pos[2]);
    }
    if (!edgePos.length) return;
    const eGeo = new THREE.BufferGeometry();
    eGeo.setAttribute('position', new THREE.Float32BufferAttribute(edgePos, 3));
    wireEdgeObj = new THREE.LineSegments(
      eGeo,
      new THREE.LineBasicMaterial({ color: 0x00d4ff, transparent: true, opacity: 0.7 })
    );
    scene.add(wireEdgeObj);
  }

  function clear3DScene() {
    if (previewPoints) {
      previewGeometry.setDrawRange(0, 0);
      previewPoints.visible = cfg.showVoxels;
    }
    if (surfelObj) {
      scene.remove(surfelObj);
      surfelObj.geometry.dispose();
      surfelObj.material.dispose();
      surfelObj = null;
    }
    if (wireVertexObj) {
      scene.remove(wireVertexObj);
      wireVertexObj.geometry.dispose();
      wireVertexObj.material.dispose();
      wireVertexObj = null;
    }
    if (wireEdgeObj) {
      scene.remove(wireEdgeObj);
      wireEdgeObj.geometry.dispose();
      wireEdgeObj.material.dispose();
      wireEdgeObj = null;
    }
  }

  function startRenderLoop() {
    if (renderFrame) return;
    function loop() {
      renderFrame = requestAnimationFrame(loop);
      const pose = liveOrientation || lastPose || buildPose(0, 0, cfg.cameraR, 0, 0);
      if (cfg.syncCamera && scanning) {
        const scale = 0.72;
        cam3d.position.set(
          pose.position[0] * scale,
          pose.position[1] * scale + 0.18,
          pose.position[2] * scale
        );
      } else {
        const t = performance.now() * 0.00045;
        cam3d.position.set(Math.sin(t) * 1.8, 0.6, Math.cos(t) * 1.8);
      }
      cam3d.lookAt(0, 0, 0);
      const parent = previewCanvas.parentElement;
      if (parent && parent.clientWidth > 0) {
        renderer.setSize(parent.clientWidth, parent.clientHeight);
        cam3d.aspect = parent.clientWidth / parent.clientHeight;
        cam3d.updateProjectionMatrix();
      }
      update3DPreview();
      updateWireframePreview();
      renderer.render(scene, cam3d);
    }
    loop();
  }

  function stopRenderLoop() {
    if (!renderFrame) return;
    cancelAnimationFrame(renderFrame);
    renderFrame = null;
  }

  function drawOverlay(edgeMap, lines, corners, sil, pw, ph, pose, integrateFrame) {
    const ow = overlayCanvas.width;
    const oh = overlayCanvas.height;
    const sx = ow / pw;
    const sy = oh / ph;
    overlayCtx.clearRect(0, 0, ow, oh);
    if (cfg.showSilhouette) {
      overlayCtx.globalAlpha = cfg.overlayOpacity * 0.42;
      overlayCtx.fillStyle = '#34d399';
      for (let y = 0; y < ph; y += 2) {
        for (let x = 0; x < pw; x += 2) {
          if (sil[y * pw + x]) {
            overlayCtx.fillRect(x * sx, y * sy, Math.ceil(sx * 2), Math.ceil(sy * 2));
          }
        }
      }
    }
    if (cfg.showEdges) {
      overlayCtx.globalAlpha = cfg.overlayOpacity;
      overlayCtx.fillStyle = '#00d4ff';
      for (let y = 0; y < ph; y++) {
        for (let x = 0; x < pw; x++) {
          if (edgeMap[y * pw + x]) overlayCtx.fillRect(x * sx, y * sy, Math.ceil(sx), Math.ceil(sy));
        }
      }
    }
    overlayCtx.globalAlpha = 1;
    if (cfg.showLines) {
      overlayCtx.strokeStyle = '#facc15';
      overlayCtx.lineWidth = 1.5;
      for (const line of lines) {
        overlayCtx.beginPath();
        overlayCtx.moveTo(line.x1 * sx, line.y1 * sy);
        overlayCtx.lineTo(line.x2 * sx, line.y2 * sy);
        overlayCtx.stroke();
      }
    }
    if (cfg.showCorners || cfg.showAngles) {
      for (const corner of corners) {
        if (cfg.showCorners) {
          overlayCtx.strokeStyle = '#f472b6';
          overlayCtx.lineWidth = 1.5;
          overlayCtx.beginPath();
          overlayCtx.arc(corner.x * sx, corner.y * sy, 5, 0, Math.PI * 2);
          overlayCtx.stroke();
        }
        if (cfg.showAngles) {
          overlayCtx.strokeStyle = 'rgba(244,114,182,0.55)';
          overlayCtx.lineWidth = 1;
          overlayCtx.beginPath();
          overlayCtx.arc(corner.x * sx, corner.y * sy, 16, corner.a1, corner.a2);
          overlayCtx.stroke();
          overlayCtx.fillStyle = '#f472b6';
          overlayCtx.font = '600 9px Inter, sans-serif';
          overlayCtx.fillText(corner.angle + 'deg', corner.x * sx + 8, corner.y * sy - 8);
        }
      }
    }
    overlayCtx.fillStyle = integrateFrame ? 'rgba(52,211,153,0.9)' : 'rgba(250,204,21,0.9)';
    overlayCtx.font = '600 11px Inter, sans-serif';
    overlayCtx.fillText(integrateFrame ? 'KF' : 'TRACK', 10, 18);
    overlayCtx.fillStyle = 'rgba(255,255,255,0.82)';
    overlayCtx.font = '500 10px Inter, sans-serif';
    overlayCtx.fillText(
      `yaw ${radToDeg(pose.yaw).toFixed(1)} | map ${previewCount} | conf ${Math.round(previewConfidence * 100)}%`,
      10,
      33
    );
    overlayCtx.fillText(`lock ${Math.round(poseLock * 100)}% | surfels ${surfels.length}`, 10, 48);
  }

  function scheduleFrame() {
    if (!scanning || paused || processing) return;
    const wait = Math.max(0, cfg.minFrameMs - (performance.now() - lastProcTime));
    setTimeout(() => {
      if (scanning && !paused) processFrame();
    }, wait);
  }

  function processFrame() {
    if (!scanning || paused || processing) return;
    if (!videoEl || videoEl.readyState < 2) {
      setTimeout(scheduleFrame, 200);
      return;
    }
    processing = true;
    const started = performance.now();
    try {
      const vw = videoEl.videoWidth;
      const vh = videoEl.videoHeight;
      const scale = cfg.procSize / Math.max(vw, vh);
      const pw = Math.max(32, Math.round(vw * scale));
      const ph = Math.max(32, Math.round(vh * scale));
      ensureFrameCanvas(pw, ph);
      frameCtx.drawImage(videoEl, 0, 0, pw, ph);
      const imgData = frameCtx.getImageData(0, 0, pw, ph);
      let gray = toGray(imgData.data, pw * ph);
      if (cfg.sharpen) {
        gray = unsharpMask(gray, pw, ph, cfg.sharpenAmount);
      }
      const edgeMap = cannyEdges(gray, pw, ph);
      const lines = detectLines(edgeMap, pw, ph);
      const corners = detectCornersAndAngles(lines);
      prepareCornersForTracking(corners, gray, pw, ph);
      const sil = extractSilhouette(gray, pw, ph);
      const silStats = measureSilhouette(sil, pw, ph);
      const imageQuality = computeImageQuality(edgeMap, silStats, pw, ph);
      const poseInfo = estimatePose(corners, silStats, pw, ph, imageQuality);
      const pose = poseInfo.pose;
      liveOrientation = pose;
      lastPose = clonePose(pose);
      trackingMatches = poseInfo.visual.matchCount;
      edgeCount = lines.length;
      cornerCount = corners.length;
      quality = Math.round(imageQuality * 0.68 + pose.confidence * 32);
      processVertexTracking(corners, lines, pw, ph, pose, imgData);
      const integrateFrame = shouldIntegrateKeyframe(pose, silStats, quality);
      if (integrateFrame || frameCount < 2) {
        carveVisualHull(sil, pw, ph, pose);
        colorVoxels(imgData, pw, ph, pose, sil);
        rememberKeyframe(corners, pose, silStats);
        lastIntegratedPose = clonePose(pose);
      }
      storeTrackingFrame(corners, silStats, pose);
      resizeOverlay();
      drawOverlay(edgeMap, lines, corners, sil, pw, ph, pose, integrateFrame);
      frameCount++;
      const dt = performance.now() - started;
      const currentFps = dt > 0 ? 1000 / dt : 0;
      fps = fps ? Math.round(fps * 0.7 + currentFps * 0.3) : Math.round(currentFps);
      lastProcTime = performance.now();
      emitStats();
    } catch (err) {
      console.error('Realtime frame error:', err);
    }
    processing = false;
    scheduleFrame();
  }

  function resizeOverlay() {
    if (!overlayCanvas || !videoEl) return;
    const rect = videoEl.getBoundingClientRect();
    if (overlayCanvas.width !== rect.width || overlayCanvas.height !== rect.height) {
      overlayCanvas.width = rect.width;
      overlayCanvas.height = rect.height;
    }
  }

  function emitStats() {
    if (onStatsUpdate) onStatsUpdate(getStats());
  }

  function emitStatus(status) {
    if (onStatusChange) onStatusChange(status);
  }

  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  function mix(a, b, t) {
    return a * (1 - t) + b * t;
  }

  function median(values) {
    if (!values.length) return 0;
    const sorted = values.slice().sort((a, b) => a - b);
    const mid = Math.floor(sorted.length / 2);
    return sorted.length % 2
      ? sorted[mid]
      : (sorted[mid - 1] + sorted[mid]) * 0.5;
  }

  function sub3(a, b) {
    return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
  }

  function add3(a, b) {
    return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
  }

  function scale3(v, s) {
    return [v[0] * s, v[1] * s, v[2] * s];
  }

  function length3(v) {
    return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }

  function normalize3(v) {
    const len = length3(v) || 1;
    return [v[0] / len, v[1] / len, v[2] / len];
  }

  function dot3(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  function cross3(a, b) {
    return [
      a[1] * b[2] - a[2] * b[1],
      a[2] * b[0] - a[0] * b[2],
      a[0] * b[1] - a[1] * b[0],
    ];
  }

  function distance3(a, b) {
    const dx = a[0] - b[0];
    const dy = a[1] - b[1];
    const dz = a[2] - b[2];
    return Math.sqrt(dx * dx + dy * dy + dz * dz);
  }

  function radToDeg(rad) {
    return rad * 180 / Math.PI;
  }

  return {
    init,
    startCamera,
    stopCamera,
    startScanning,
    pauseScanning,
    resumeScanning,
    stopScanning,
    reset,
    destroy,
    getPointCloud,
    getStats,
    isScanning,
    isPaused,
    isActive,
    stopRenderLoop,
    set,
    get,
    getAll,
    applyPreset,
  };
})();
