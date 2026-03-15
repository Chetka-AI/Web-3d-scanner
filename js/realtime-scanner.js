/**
 * RealtimeScanner — Edge-based live 3D reconstruction.
 *
 * Pipeline per frame (~40-100 ms on mobile):
 *  1. Optional unsharp-mask sharpening
 *  2. Canny edge detection (blur → Sobel → NMS → hysteresis)
 *  3. Line segment extraction (chain trace + Douglas-Peucker)
 *  4. Corner detection + angle computation
 *  5. Silhouette extraction (Otsu / manual threshold + flood fill)
 *  6. Morphological ops on silhouette (dilate / erode / open / close)
 *  7. Visual-hull carving on voxel grid
 *  8. Sparse optical flow (Lucas-Kanade) for inter-frame tracking
 *  9. Temporal point fusion with confidence-weighted ICP alignment
 * 10. Overlay rendering (edges / lines / corners / angles / silhouette)
 * 11. Quality metric calculation
 */
const RealtimeScanner = (() => {

  /* ============== Dynamic Configuration ============== */
  const cfg = {
    procSize:       240,
    cannyLo:        30,
    cannyHi:        80,
    blurRadius:     1,          // 0 = off, 1 = 3×3, 2 = 5×5
    sharpen:        false,
    sharpenAmount:  0.4,

    minLineLen:     12,
    cornerDist:     8,
    dpEpsilon:      3.0,

    voxelRes:       48,
    voxelWorld:     1.6,
    cameraR:        2.5,
    fovDeg:         60,

    silhouetteMode: 'auto',     // 'auto' | 'manual'
    manualThreshold: 128,
    invertSilhouette: false,
    morphType:      'close',    // 'none' | 'dilate' | 'erode' | 'open' | 'close'
    morphIterations: 1,

    showEdges:      true,
    showLines:      true,
    showCorners:    true,
    showAngles:     true,
    showSilhouette: true,
    overlayOpacity: 0.35,

    minFrameMs:     50,
    motionThreshold: 0.005,
    smoothOrientation: true,
    smoothFactor:   0.3,

    syncCamera:     true,
    showWireframe:  true,
    showVoxels:     true,
    descRadius:     3,
    matchThreshold: 0.55,
    minAngleDiff:   0.15,
    maxHistory:     6,

    useKalman:          true,
    kalmanQ:            0.001,   // process noise covariance
    kalmanR:            0.05,    // measurement noise covariance

    useOpticalFlow:     true,
    optFlowWin:         7,       // LK window half-size
    optFlowMaxPts:      60,      // max tracked feature points
    optFlowMinDisp:     1.5,     // min displacement to count as motion (px)

    useTemporalFusion:  true,
    fusionVoxelSize:    0.02,    // spatial hash cell size for point fusion
    fusionMaxAge:       30,      // frames after which unfused points decay
    fusionMinConf:      2,       // min observations for a point to be stable

    adaptiveFps:        true,
    targetFps:          15,

    useICP:             true,
    icpIterations:      5,
    icpMaxCorrespDist:  0.15,
  };

  const PRESETS = {
    fast:     { procSize:180, cannyLo:40, cannyHi:100, blurRadius:1, sharpen:false, minLineLen:16, voxelRes:32, minFrameMs:30, morphType:'none', morphIterations:0, dpEpsilon:4, maxHistory:4, useOpticalFlow:false, useTemporalFusion:false, useICP:false, adaptiveFps:true },
    balanced: { procSize:240, cannyLo:30, cannyHi:80,  blurRadius:1, sharpen:false, minLineLen:12, voxelRes:48, minFrameMs:50, morphType:'close', morphIterations:1, dpEpsilon:3, maxHistory:6, useOpticalFlow:true,  useTemporalFusion:true,  useICP:false, adaptiveFps:true },
    quality:  { procSize:320, cannyLo:20, cannyHi:65,  blurRadius:2, sharpen:true,  minLineLen:8,  voxelRes:64, minFrameMs:80, morphType:'close', morphIterations:2, dpEpsilon:2, maxHistory:8, useOpticalFlow:true,  useTemporalFusion:true,  useICP:true,  adaptiveFps:true },
  };

  function set(key, value) { if (key in cfg) cfg[key] = value; }
  function get(key) { return cfg[key]; }
  function getAll() { return { ...cfg }; }
  function applyPreset(name) {
    const p = PRESETS[name];
    if (!p) return;
    for (const k in p) cfg[k] = p[k];
    if (scanning) { initVoxelGrid(); clear3DScene(); resetFusionBuffer(); }
  }

  /* ============== Kalman Filter (1D, per-axis) ============== */
  function makeKalman1D(q, r, initial) {
    return { x: initial || 0, p: 1.0, q: q, r: r };
  }
  function kalmanUpdate(kf, measurement) {
    kf.p += kf.q;
    const k = kf.p / (kf.p + kf.r);
    kf.x += k * (measurement - kf.x);
    kf.p *= (1 - k);
    return kf.x;
  }

  /* ============== State ============== */
  let videoEl = null, overlayCanvas = null, overlayCtx = null, previewCanvas = null;
  let stream = null;
  let scene, cam3d, renderer, voxelMesh = null;

  let scanning = false, paused = false, processing = false;
  let voxels = null, voxelsDirty = false;
  let frameCount = 0, totalVoxels = 0, edgeCount = 0, cornerCount = 0;
  let fps = 0, lastProcTime = 0, quality = 0;

  let renderFrame = null, prevAngle = 0;
  let onStatsUpdate = null, onStatusChange = null;

  let smoothYaw = 0, smoothPitch = 0;
  let lastYaw = null;

  let kfYaw = null, kfPitch = null;

  let prevGray = null, prevFeatures = null;
  let opticalFlowDelta = { dx: 0, dy: 0, count: 0 };

  let fusionMap = null;
  let fusionDirty = false;
  let fusionPointObj = null;
  const FUSION_MAX_POINTS = 100000;

  let voxelBufGeo = null, voxelBufMat = null;
  const VOXEL_BUF_MAX = 200000;

  let recentFpsSamples = [];

  let vertexHistory = [];
  let vertices3D = [];
  let edges3D = new Set();
  let wireVertexObj = null, wireEdgeObj = null;
  let wireframeDirty = false;
  let liveOrientation = { yaw: 0, pitch: 0 };

  /* ============== Public API ============== */

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
        video: { facingMode: 'environment', width: { ideal: 1280 }, height: { ideal: 960 } },
        audio: false,
      });
      videoEl.srcObject = stream;
      await videoEl.play();
      return true;
    } catch (e) {
      console.error('Camera error:', e);
      return false;
    }
  }

  function stopCamera() {
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    if (videoEl) videoEl.srcObject = null;
  }

  function startScanning() {
    scanning = true; paused = false; processing = false;
    frameCount = 0; edgeCount = 0; cornerCount = 0; quality = 0;
    smoothYaw = 0; smoothPitch = 0; lastYaw = null;
    kfYaw = null; kfPitch = null;
    prevGray = null; prevFeatures = null;
    opticalFlowDelta = { dx: 0, dy: 0, count: 0 };
    recentFpsSamples = [];
    vertexHistory = []; vertices3D = []; edges3D = new Set();
    wireframeDirty = false;
    initVoxelGrid();
    resetFusionBuffer();
    clear3DScene();
    startRenderLoop();
    emitStatus('scanning');
    scheduleFrame();
  }

  function pauseScanning()  { paused = true;  emitStatus('paused'); }
  function resumeScanning() { paused = false; emitStatus('scanning'); scheduleFrame(); }
  function stopScanning()   { scanning = false; paused = false; emitStatus('stopped'); }

  function reset() {
    stopScanning(); frameCount = 0; quality = 0;
    initVoxelGrid(); resetFusionBuffer(); clear3DScene();
    prevGray = null; prevFeatures = null;
    kfYaw = null; kfPitch = null;
    if (overlayCtx) overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    emitStats();
  }

  function destroy() {
    stopScanning(); stopCamera(); stopRenderLoop();
    clear3DScene(); voxels = null; fusionMap = null;
    prevGray = null; prevFeatures = null;
  }

  function getPointCloud() {
    if (cfg.useTemporalFusion) {
      const fused = getFusedPointCloud();
      if (fused) return fused;
    }
    if (!voxels) return { positions: new Float32Array(0), colors: new Float32Array(0) };
    const R = cfg.voxelRes, half = cfg.voxelWorld / 2, step = cfg.voxelWorld / R;
    const pos = [], col = [];
    for (let i = 0; i < R; i++)
      for (let j = 0; j < R; j++)
        for (let k = 0; k < R; k++) {
          const v = voxels[i*R*R + j*R + k];
          if (!v) continue;
          pos.push(i*step - half, j*step - half, k*step - half);
          const r = (v >> 16 & 0xff) / 255, g = (v >> 8 & 0xff) / 255, b = (v & 0xff) / 255;
          col.push(r || 0.4, g || 0.7, b || 0.9);
        }
    return { positions: new Float32Array(pos), colors: new Float32Array(col) };
  }

  function getStats() {
    return {
      frames: frameCount, points: totalVoxels,
      edges: edgeCount, corners: cornerCount,
      verts3D: vertices3D.length, edges3D: edges3D.size,
      fps, quality, scanning, paused,
      flowTracked: opticalFlowDelta.count,
      fusedPts: fusionMap ? fusionMap.size : 0,
      procSize: cfg.procSize,
    };
  }
  function isScanning() { return scanning; }
  function isPaused()   { return paused; }
  function isActive()   { return stream !== null; }

  /* ============== Image Pre-processing ============== */

  function toGray(data, n) {
    const g = new Float32Array(n);
    for (let i = 0; i < n; i++) { const j = i*4; g[i] = 0.299*data[j] + 0.587*data[j+1] + 0.114*data[j+2]; }
    return g;
  }

  function unsharpMask(gray, w, h, amount) {
    const blurred = gaussianBlur(gray, w, h, 2);
    const out = new Float32Array(w * h);
    for (let i = 0; i < out.length; i++) {
      out[i] = Math.min(255, Math.max(0, gray[i] + amount * (gray[i] - blurred[i])));
    }
    return out;
  }

  /* ============== Edge Detection (Canny) ============== */

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
      const k = [1/16,2/16,1/16, 2/16,4/16,2/16, 1/16,2/16,1/16];
      for (let y = 1; y < h-1; y++)
        for (let x = 1; x < w-1; x++) {
          let s = 0, ki = 0;
          for (let dy = -1; dy <= 1; dy++)
            for (let dx = -1; dx <= 1; dx++)
              s += src[(y+dy)*w+(x+dx)] * k[ki++];
          dst[y*w+x] = s;
        }
    } else {
      const k = [1,4,6,4,1, 4,16,24,16,4, 6,24,36,24,6, 4,16,24,16,4, 1,4,6,4,1];
      const kSum = 256;
      for (let y = 2; y < h-2; y++)
        for (let x = 2; x < w-2; x++) {
          let s = 0, ki = 0;
          for (let dy = -2; dy <= 2; dy++)
            for (let dx = -2; dx <= 2; dx++)
              s += src[(y+dy)*w+(x+dx)] * k[ki++];
          dst[y*w+x] = s / kSum;
        }
    }
    return dst;
  }

  function sobelGradients(src, w, h) {
    const mag = new Float32Array(w*h), dir = new Float32Array(w*h);
    for (let y = 1; y < h-1; y++)
      for (let x = 1; x < w-1; x++) {
        const tl=src[(y-1)*w+x-1], tc=src[(y-1)*w+x], tr=src[(y-1)*w+x+1];
        const ml=src[y*w+x-1], mr=src[y*w+x+1];
        const bl=src[(y+1)*w+x-1], bc=src[(y+1)*w+x], br=src[(y+1)*w+x+1];
        const gx = -tl+tr - 2*ml+2*mr - bl+br;
        const gy = -tl-2*tc-tr + bl+2*bc+br;
        const idx = y*w+x;
        mag[idx] = Math.sqrt(gx*gx + gy*gy);
        dir[idx] = Math.atan2(gy, gx);
      }
    return { mag, dir };
  }

  function nonMaxSuppress(mag, dir, w, h) {
    const out = new Float32Array(w*h);
    for (let y = 1; y < h-1; y++)
      for (let x = 1; x < w-1; x++) {
        const idx = y*w+x, m = mag[idx];
        let a = ((dir[idx]*180/Math.PI)+180)%180, n1, n2;
        if (a<22.5||a>=157.5)     { n1=mag[idx-1]; n2=mag[idx+1]; }
        else if (a<67.5)          { n1=mag[(y-1)*w+x+1]; n2=mag[(y+1)*w+x-1]; }
        else if (a<112.5)         { n1=mag[(y-1)*w+x]; n2=mag[(y+1)*w+x]; }
        else                      { n1=mag[(y-1)*w+x-1]; n2=mag[(y+1)*w+x+1]; }
        out[idx] = (m>=n1 && m>=n2) ? m : 0;
      }
    return out;
  }

  function doubleThreshold(thin, w, h, lo, hi) {
    const edge = new Uint8Array(w*h);
    for (let i = 0; i < thin.length; i++) {
      if (thin[i]>=hi) edge[i]=2; else if (thin[i]>=lo) edge[i]=1;
    }
    let changed = true;
    while (changed) {
      changed = false;
      for (let y = 1; y < h-1; y++)
        for (let x = 1; x < w-1; x++) {
          const idx = y*w+x;
          if (edge[idx]!==1) continue;
          for (let dy=-1; dy<=1; dy++)
            for (let dx=-1; dx<=1; dx++)
              if (edge[(y+dy)*w+(x+dx)]===2) { edge[idx]=2; changed=true; }
        }
    }
    for (let i = 0; i < edge.length; i++) edge[i] = edge[i]===2 ? 1 : 0;
    return edge;
  }

  /* ============== Line Segments ============== */

  function detectLines(edgeMap, w, h) {
    const lines = [], visited = new Uint8Array(w*h);
    for (let y = 2; y < h-2; y++)
      for (let x = 2; x < w-2; x++) {
        const idx = y*w+x;
        if (!edgeMap[idx] || visited[idx]) continue;
        const chain = traceChain(edgeMap, visited, x, y, w, h);
        if (chain.length < cfg.minLineLen) continue;
        const segs = splitToSegments(chain);
        for (const s of segs) if (segLen(s) >= cfg.minLineLen) lines.push(s);
      }
    return lines;
  }

  function traceChain(em, vis, sx, sy, w, h) {
    const chain = [], dirs = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,1],[1,-1],[1,0],[1,1]];
    let cx=sx, cy=sy;
    for (let s=0; s<600; s++) {
      const idx = cy*w+cx;
      if (vis[idx]) break;
      vis[idx]=1; chain.push(cx,cy);
      let found=false;
      for (const [dy,dx] of dirs) {
        const nx=cx+dx, ny=cy+dy;
        if (nx<1||nx>=w-1||ny<1||ny>=h-1) continue;
        if (em[ny*w+nx] && !vis[ny*w+nx]) { cx=nx; cy=ny; found=true; break; }
      }
      if (!found) break;
    }
    return chain;
  }

  function splitToSegments(chain) {
    const pts = [];
    for (let i=0; i<chain.length; i+=2) pts.push({x:chain[i],y:chain[i+1]});
    if (pts.length<2) return [];
    const ids = douglasPeucker(pts, 0, pts.length-1, cfg.dpEpsilon);
    const segs = [];
    for (let i=0; i<ids.length-1; i++)
      segs.push({x1:pts[ids[i]].x, y1:pts[ids[i]].y, x2:pts[ids[i+1]].x, y2:pts[ids[i+1]].y});
    return segs;
  }

  function douglasPeucker(pts, s, e, eps) {
    if (e<=s+1) return [s,e];
    let md=0, mi=s;
    const lx=pts[e].x-pts[s].x, ly=pts[e].y-pts[s].y, lsq=lx*lx+ly*ly;
    for (let i=s+1; i<e; i++) {
      const dx=pts[i].x-pts[s].x, dy=pts[i].y-pts[s].y;
      const d = lsq>0 ? Math.abs(dx*ly-dy*lx)/Math.sqrt(lsq) : Math.sqrt(dx*dx+dy*dy);
      if (d>md) { md=d; mi=i; }
    }
    if (md<=eps) return [s,e];
    return douglasPeucker(pts,s,mi,eps).concat(douglasPeucker(pts,mi,e,eps).slice(1));
  }

  function segLen(s) { return Math.sqrt((s.x2-s.x1)**2+(s.y2-s.y1)**2); }
  function segAng(s) { return Math.atan2(s.y2-s.y1, s.x2-s.x1); }

  /* ============== Corners & Angles ============== */

  function detectCornersAndAngles(lines) {
    const raw = [];
    for (let i=0; i<lines.length; i++)
      for (let j=i+1; j<lines.length; j++) {
        const a=lines[i], b=lines[j];
        const pairs=[[a.x1,a.y1,b.x1,b.y1],[a.x1,a.y1,b.x2,b.y2],[a.x2,a.y2,b.x1,b.y1],[a.x2,a.y2,b.x2,b.y2]];
        for (const [ax,ay,bx,by] of pairs) {
          if (Math.sqrt((ax-bx)**2+(ay-by)**2) < cfg.cornerDist) {
            const a1=segAng(a), a2=segAng(b);
            let diff = Math.abs(a1-a2);
            if (diff>Math.PI) diff = 2*Math.PI-diff;
            const deg = Math.round(diff*180/Math.PI);
            if (deg>8 && deg<172)
              raw.push({x:(ax+bx)/2, y:(ay+by)/2, angle:deg, a1, a2});
          }
        }
      }
    const out = [];
    for (const c of raw) {
      let dup = false;
      for (const o of out) if (Math.sqrt((c.x-o.x)**2+(c.y-o.y)**2) < cfg.cornerDist*2) { dup=true; break; }
      if (!dup) out.push(c);
    }
    return out;
  }

  /* ============== Silhouette ============== */

  function extractSilhouette(gray, w, h) {
    const thr = cfg.silhouetteMode === 'manual' ? cfg.manualThreshold : otsuThreshold(gray, w, h);
    const mask = new Uint8Array(w*h);
    for (let i=0; i<gray.length; i++) {
      const v = gray[i] < thr ? 1 : 0;
      mask[i] = cfg.invertSilhouette ? (1-v) : v;
    }

    if (cfg.morphType !== 'none' && cfg.morphIterations > 0) {
      applyMorphology(mask, w, h, cfg.morphType, cfg.morphIterations);
    }

    const outside = new Uint8Array(w*h);
    const queue = [];
    for (let x=0; x<w; x++) {
      if (!mask[x])        { outside[x]=1; queue.push(x,0); }
      if (!mask[(h-1)*w+x]){ outside[(h-1)*w+x]=1; queue.push(x,h-1); }
    }
    for (let y=0; y<h; y++) {
      if (!mask[y*w])      { outside[y*w]=1; queue.push(0,y); }
      if (!mask[y*w+w-1])  { outside[y*w+w-1]=1; queue.push(w-1,y); }
    }
    while (queue.length) {
      const qy=queue.pop(), qx=queue.pop();
      for (const [nx,ny] of [[qx-1,qy],[qx+1,qy],[qx,qy-1],[qx,qy+1]]) {
        if (nx<0||nx>=w||ny<0||ny>=h) continue;
        const ni=ny*w+nx;
        if (!outside[ni] && !mask[ni]) { outside[ni]=1; queue.push(nx,ny); }
      }
    }
    const sil = new Uint8Array(w*h);
    for (let i=0; i<gray.length; i++) sil[i] = outside[i] ? 0 : 1;
    return sil;
  }

  function otsuThreshold(gray, w, h) {
    const hist = new Int32Array(256), total = w*h;
    for (let i=0; i<total; i++) hist[Math.min(255,Math.max(0,Math.round(gray[i])))]++;
    let sum=0; for (let i=0;i<256;i++) sum+=i*hist[i];
    let sumB=0, wB=0, best=0, bestT=128;
    for (let t=0;t<256;t++) {
      wB+=hist[t]; if(!wB) continue;
      const wF=total-wB; if(!wF) break;
      sumB+=t*hist[t];
      const d=sumB/wB - (sum-sumB)/wF;
      const between = wB*wF*d*d;
      if (between>best) { best=between; bestT=t; }
    }
    return bestT;
  }

  /* ============== Morphological Operations ============== */

  function applyMorphology(mask, w, h, type, iterations) {
    for (let iter = 0; iter < iterations; iter++) {
      if (type === 'dilate')      dilate(mask, w, h);
      else if (type === 'erode')  erode(mask, w, h);
      else if (type === 'open')   { erode(mask, w, h); dilate(mask, w, h); }
      else if (type === 'close')  { dilate(mask, w, h); erode(mask, w, h); }
    }
  }

  function dilate(mask, w, h) {
    const tmp = new Uint8Array(mask);
    for (let y=1; y<h-1; y++)
      for (let x=1; x<w-1; x++) {
        if (tmp[y*w+x]) continue;
        if (tmp[(y-1)*w+x]||tmp[(y+1)*w+x]||tmp[y*w+x-1]||tmp[y*w+x+1])
          mask[y*w+x]=1;
      }
  }

  function erode(mask, w, h) {
    const tmp = new Uint8Array(mask);
    for (let y=1; y<h-1; y++)
      for (let x=1; x<w-1; x++) {
        if (!tmp[y*w+x]) continue;
        if (!tmp[(y-1)*w+x]||!tmp[(y+1)*w+x]||!tmp[y*w+x-1]||!tmp[y*w+x+1])
          mask[y*w+x]=0;
      }
  }

  /* ============== Vertex Tracking & Wireframe ============== */

  function extractDescriptor(gray, w, h, cx, cy) {
    const r = cfg.descRadius, size = (2*r+1)*(2*r+1);
    const d = new Float32Array(size);
    let mean = 0, n = 0;
    for (let dy=-r; dy<=r; dy++)
      for (let dx=-r; dx<=r; dx++) {
        const px=cx+dx, py=cy+dy;
        d[n] = (px>=0&&px<w&&py>=0&&py<h) ? gray[py*w+px] : 0;
        mean += d[n]; n++;
      }
    mean /= n;
    let norm = 0;
    for (let i=0; i<d.length; i++) { d[i]-=mean; norm+=d[i]*d[i]; }
    norm = Math.sqrt(norm)||1;
    for (let i=0; i<d.length; i++) d[i]/=norm;
    return d;
  }

  function ncc(a, b) {
    let s=0; for (let i=0; i<a.length; i++) s+=a[i]*b[i]; return s;
  }

  function processVertexTracking(corners, lines, gray, w, h, angle, imgData) {
    for (const c of corners) {
      c.descriptor = extractDescriptor(gray, w, h, Math.round(c.x), Math.round(c.y));
      c.v3dIdx = -1;
    }

    const adjacency = buildAdjacency(corners, lines);

    for (const hist of vertexHistory) {
      const angleDiff = Math.abs(angle - hist.angle);
      if (angleDiff < cfg.minAngleDiff) continue;

      for (const cur of corners) {
        if (cur.v3dIdx >= 0) continue;
        let bestScore = cfg.matchThreshold, bestHist = null;
        for (const hv of hist.vertices) {
          const score = ncc(cur.descriptor, hv.descriptor);
          if (score > bestScore) { bestScore = score; bestHist = hv; }
        }
        if (!bestHist) continue;

        const pos = triangulate2Views(cur.x, cur.y, angle, bestHist.x, bestHist.y, hist.angle, w, h);
        if (!pos) continue;

        const dist = Math.sqrt(pos[0]*pos[0]+pos[1]*pos[1]+pos[2]*pos[2]);
        if (dist > cfg.voxelWorld) continue;

        const px = Math.round(cur.x), py = Math.round(cur.y);
        const pi = (py*w+px)*4;
        const col = [imgData.data[pi]/255, imgData.data[pi+1]/255, imgData.data[pi+2]/255];

        if (bestHist.v3dIdx >= 0) {
          const v = vertices3D[bestHist.v3dIdx];
          const f = 1 / (v.confidence + 1);
          v.pos[0] += (pos[0]-v.pos[0])*f;
          v.pos[1] += (pos[1]-v.pos[1])*f;
          v.pos[2] += (pos[2]-v.pos[2])*f;
          v.confidence++;
          cur.v3dIdx = bestHist.v3dIdx;
        } else {
          cur.v3dIdx = vertices3D.length;
          bestHist.v3dIdx = cur.v3dIdx;
          vertices3D.push({ pos, col, confidence: 1 });
        }
        wireframeDirty = true;
      }
    }

    for (let i=0; i<corners.length; i++) {
      if (corners[i].v3dIdx < 0) continue;
      const neighbors = adjacency.get(i);
      if (!neighbors) continue;
      for (const j of neighbors) {
        if (corners[j].v3dIdx < 0) continue;
        const a = Math.min(corners[i].v3dIdx, corners[j].v3dIdx);
        const b = Math.max(corners[i].v3dIdx, corners[j].v3dIdx);
        const key = a * 100000 + b;
        if (!edges3D.has(key)) { edges3D.add(key); wireframeDirty = true; }
      }
    }

    vertexHistory.push({ vertices: corners, angle });
    if (vertexHistory.length > cfg.maxHistory) vertexHistory.shift();
  }

  function buildAdjacency(corners, lines) {
    const adj = new Map();
    for (let i=0; i<corners.length; i++) adj.set(i, new Set());
    for (const l of lines) {
      let ci=-1, cj=-1;
      for (let k=0; k<corners.length; k++) {
        const c = corners[k];
        const d1 = Math.sqrt((c.x-l.x1)**2+(c.y-l.y1)**2);
        const d2 = Math.sqrt((c.x-l.x2)**2+(c.y-l.y2)**2);
        if (d1 < cfg.cornerDist*2.5) ci = k;
        if (d2 < cfg.cornerDist*2.5) cj = k;
      }
      if (ci>=0 && cj>=0 && ci!==cj) { adj.get(ci).add(cj); adj.get(cj).add(ci); }
    }
    return adj;
  }

  function triangulate2Views(px1, py1, a1, px2, py2, a2, w, h) {
    const fov = cfg.fovDeg*Math.PI/180;
    const fx = w/(2*Math.tan(fov/2));
    const cxI=w/2, cyI=h/2, R=cfg.cameraR;

    const cam1 = [R*Math.sin(a1), 0, R*Math.cos(a1)];
    const cam2 = [R*Math.sin(a2), 0, R*Math.cos(a2)];

    const r1c = [(px1-cxI)/fx, -(py1-cyI)/fx, -1];
    const c1=Math.cos(a1), s1=Math.sin(a1);
    const d1 = normVec([r1c[0]*c1-r1c[2]*s1, r1c[1], r1c[0]*s1+r1c[2]*c1]);

    const r2c = [(px2-cxI)/fx, -(py2-cyI)/fx, -1];
    const c2=Math.cos(a2), s2=Math.sin(a2);
    const d2 = normVec([r2c[0]*c2-r2c[2]*s2, r2c[1], r2c[0]*s2+r2c[2]*c2]);

    const w0 = [cam1[0]-cam2[0], cam1[1]-cam2[1], cam1[2]-cam2[2]];
    const a=dot(d1,d1), b=dot(d1,d2), c=dot(d2,d2), d=dot(d1,w0), e=dot(d2,w0);
    const den = a*c-b*b;
    if (Math.abs(den)<1e-8) return null;
    const sc = (b*e-c*d)/den, tc = (a*e-b*d)/den;
    const p1 = [cam1[0]+sc*d1[0], cam1[1]+sc*d1[1], cam1[2]+sc*d1[2]];
    const p2 = [cam2[0]+tc*d2[0], cam2[1]+tc*d2[1], cam2[2]+tc*d2[2]];
    const err = Math.sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2+(p1[2]-p2[2])**2);
    if (err > 0.8) return null;
    return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2];
  }

  function normVec(v) {
    const l=Math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])||1;
    return [v[0]/l,v[1]/l,v[2]/l];
  }
  function dot(a,b) { return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]; }

  /* ============== Sparse Optical Flow (Lucas-Kanade) ============== */

  function selectFlowFeatures(gray, w, h, edgeMap) {
    const pts = [];
    const step = Math.max(4, Math.floor(Math.min(w, h) / 20));
    for (let y = step; y < h - step; y += step) {
      for (let x = step; x < w - step; x += step) {
        if (edgeMap && edgeMap[y * w + x]) {
          pts.push({ x, y });
        } else {
          const gx = (gray[y * w + x + 1] || 0) - (gray[y * w + x - 1] || 0);
          const gy = (gray[(y + 1) * w + x] || 0) - (gray[(y - 1) * w + x] || 0);
          if (Math.abs(gx) + Math.abs(gy) > 30) pts.push({ x, y });
        }
        if (pts.length >= cfg.optFlowMaxPts) return pts;
      }
    }
    return pts;
  }

  function trackLK(prevG, curG, w, h, features) {
    const win = cfg.optFlowWin;
    const tracked = [];
    for (const f of features) {
      let px = f.x, py = f.y;
      let converged = false;
      for (let iter = 0; iter < 10; iter++) {
        let sxx = 0, syy = 0, sxy = 0, stx = 0, sty = 0;
        for (let dy = -win; dy <= win; dy++) {
          for (let dx = -win; dx <= win; dx++) {
            const ox = f.x + dx, oy = f.y + dy;
            const nx = Math.round(px + dx), ny = Math.round(py + dy);
            if (ox < 1 || ox >= w - 1 || oy < 1 || oy >= h - 1) continue;
            if (nx < 1 || nx >= w - 1 || ny < 1 || ny >= h - 1) continue;
            const ix = (prevG[oy * w + ox + 1] - prevG[oy * w + ox - 1]) * 0.5;
            const iy = (prevG[(oy + 1) * w + ox] - prevG[(oy - 1) * w + ox]) * 0.5;
            const it = curG[ny * w + nx] - prevG[oy * w + ox];
            sxx += ix * ix; syy += iy * iy; sxy += ix * iy;
            stx += ix * it; sty += iy * it;
          }
        }
        const det = sxx * syy - sxy * sxy;
        if (Math.abs(det) < 1e-4) break;
        const ddx = (syy * (-stx) - sxy * (-sty)) / det;
        const ddy = (sxx * (-sty) - sxy * (-stx)) / det;
        px += ddx; py += ddy;
        if (Math.abs(ddx) < 0.01 && Math.abs(ddy) < 0.01) { converged = true; break; }
      }
      if (converged && px >= 0 && px < w && py >= 0 && py < h) {
        tracked.push({ ox: f.x, oy: f.y, nx: px, ny: py });
      }
    }
    return tracked;
  }

  function computeOpticalFlowMotion(prevG, curG, w, h, edgeMap) {
    if (!cfg.useOpticalFlow || !prevG) {
      opticalFlowDelta = { dx: 0, dy: 0, count: 0 };
      return;
    }
    const features = prevFeatures || selectFlowFeatures(prevG, w, h, null);
    const tracked = trackLK(prevG, curG, w, h, features);
    let sumDx = 0, sumDy = 0, cnt = 0;
    for (const t of tracked) {
      const dx = t.nx - t.ox, dy = t.ny - t.oy;
      if (Math.sqrt(dx * dx + dy * dy) > cfg.optFlowMinDisp) {
        sumDx += dx; sumDy += dy; cnt++;
      }
    }
    opticalFlowDelta = cnt > 3 ? { dx: sumDx / cnt, dy: sumDy / cnt, count: cnt } : { dx: 0, dy: 0, count: 0 };
    prevFeatures = selectFlowFeatures(curG, w, h, edgeMap);
  }

  /* ============== Temporal Point Fusion ============== */

  function resetFusionBuffer() {
    fusionMap = new Map();
    fusionDirty = true;
  }

  function fusionKey(x, y, z) {
    const s = cfg.fusionVoxelSize;
    return (Math.floor(x / s) * 73856093 ^ Math.floor(y / s) * 19349663 ^ Math.floor(z / s) * 83492791) | 0;
  }

  function fusePoints(positions, colors, currentFrame) {
    if (!cfg.useTemporalFusion) return;
    if (!fusionMap) resetFusionBuffer();
    const n = positions.length / 3;
    for (let i = 0; i < n; i++) {
      const px = positions[i * 3], py = positions[i * 3 + 1], pz = positions[i * 3 + 2];
      const key = fusionKey(px, py, pz);
      if (fusionMap.has(key)) {
        const e = fusionMap.get(key);
        const w = 1 / (e.conf + 1);
        e.x += (px - e.x) * w;
        e.y += (py - e.y) * w;
        e.z += (pz - e.z) * w;
        e.r += (colors[i * 3] - e.r) * w;
        e.g += (colors[i * 3 + 1] - e.g) * w;
        e.b += (colors[i * 3 + 2] - e.b) * w;
        e.conf++;
        e.lastSeen = currentFrame;
      } else if (fusionMap.size < FUSION_MAX_POINTS) {
        fusionMap.set(key, {
          x: px, y: py, z: pz,
          r: colors[i * 3], g: colors[i * 3 + 1], b: colors[i * 3 + 2],
          conf: 1, lastSeen: currentFrame,
        });
      }
    }

    for (const [k, v] of fusionMap) {
      if (currentFrame - v.lastSeen > cfg.fusionMaxAge && v.conf < cfg.fusionMinConf) {
        fusionMap.delete(k);
      }
    }
    fusionDirty = true;
  }

  function getFusedPointCloud() {
    if (!fusionMap || fusionMap.size === 0) return null;
    const pos = [], col = [];
    for (const v of fusionMap.values()) {
      if (v.conf >= cfg.fusionMinConf) {
        pos.push(v.x, v.y, v.z);
        col.push(v.r, v.g, v.b);
      }
    }
    return pos.length ? { positions: new Float32Array(pos), colors: new Float32Array(col) } : null;
  }

  /* ============== Simple ICP Alignment ============== */

  function alignICP(srcPts, tgtPts, tgtN) {
    if (!cfg.useICP || srcPts.length < 9 || tgtN < 3) return srcPts;

    const ns = srcPts.length / 3;
    const out = new Float32Array(srcPts);
    let cx = 0, cy = 0, cz = 0;

    for (let iter = 0; iter < cfg.icpIterations; iter++) {
      cx = 0; cy = 0; cz = 0;
      let cnt = 0;
      for (let i = 0; i < ns; i++) {
        const sx = out[i * 3], sy = out[i * 3 + 1], sz = out[i * 3 + 2];
        let bestD = cfg.icpMaxCorrespDist * cfg.icpMaxCorrespDist, bx = 0, by = 0, bz = 0, found = false;
        for (let j = 0; j < tgtN; j++) {
          const dx = sx - tgtPts[j * 3], dy = sy - tgtPts[j * 3 + 1], dz = sz - tgtPts[j * 3 + 2];
          const d2 = dx * dx + dy * dy + dz * dz;
          if (d2 < bestD) { bestD = d2; bx = tgtPts[j * 3]; by = tgtPts[j * 3 + 1]; bz = tgtPts[j * 3 + 2]; found = true; }
        }
        if (found) { cx += bx - sx; cy += by - sy; cz += bz - sz; cnt++; }
      }
      if (cnt < 3) break;
      cx /= cnt; cy /= cnt; cz /= cnt;
      for (let i = 0; i < ns; i++) {
        out[i * 3] += cx; out[i * 3 + 1] += cy; out[i * 3 + 2] += cz;
      }
    }
    return out;
  }

  /* ============== Adaptive Frame Pacing ============== */

  function adaptProcSize(measuredFps) {
    if (!cfg.adaptiveFps) return;
    recentFpsSamples.push(measuredFps);
    if (recentFpsSamples.length > 8) recentFpsSamples.shift();
    if (recentFpsSamples.length < 4) return;

    const avgFps = recentFpsSamples.reduce((a, b) => a + b, 0) / recentFpsSamples.length;
    if (avgFps < cfg.targetFps * 0.7 && cfg.procSize > 120) {
      cfg.procSize = Math.max(120, cfg.procSize - 20);
    } else if (avgFps > cfg.targetFps * 1.4 && cfg.procSize < 400) {
      cfg.procSize = Math.min(400, cfg.procSize + 10);
    }
  }

  /* ============== Incremental GPU Buffer ============== */

  function ensureVoxelBuffer() {
    if (voxelBufGeo) return;
    voxelBufGeo = new THREE.BufferGeometry();
    const pos = new Float32Array(VOXEL_BUF_MAX * 3);
    const col = new Float32Array(VOXEL_BUF_MAX * 3);
    voxelBufGeo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    voxelBufGeo.setAttribute('color', new THREE.BufferAttribute(col, 3));
    voxelBufGeo.setDrawRange(0, 0);
    voxelBufMat = new THREE.PointsMaterial({ size: 0.03, vertexColors: true, sizeAttenuation: true });
  }

  function updateVoxelBuffer(positions, colors, count) {
    if (!voxelBufGeo) return;
    const posAttr = voxelBufGeo.attributes.position;
    const colAttr = voxelBufGeo.attributes.color;
    const n = Math.min(count, VOXEL_BUF_MAX);
    posAttr.array.set(positions.subarray(0, n * 3));
    colAttr.array.set(colors.subarray(0, n * 3));
    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
    voxelBufGeo.setDrawRange(0, n);
  }

  /* ============== Visual Hull ============== */

  function initVoxelGrid() {
    const R = cfg.voxelRes, n = R*R*R;
    voxels = new Uint32Array(n);
    voxels.fill(0x668899);
    totalVoxels = n;
    voxelsDirty = true;
  }

  function carveVisualHull(sil, sw, sh, yaw) {
    const R = cfg.voxelRes, half = cfg.voxelWorld/2, step = cfg.voxelWorld/R;
    const cosA = Math.cos(yaw), sinA = Math.sin(yaw);
    const fovRad = cfg.fovDeg * Math.PI / 180;
    const fx = sw / (2*Math.tan(fovRad/2)), cx = sw/2, cy = sh/2;
    let carved = 0;

    for (let i=0; i<R; i++) {
      const wx = i*step-half;
      for (let j=0; j<R; j++) {
        const wy = j*step-half;
        for (let k=0; k<R; k++) {
          const vi = i*R*R + j*R + k;
          if (!voxels[vi]) continue;
          const wz = k*step-half;
          const camX = wx*cosA + wz*sinA;
          const camZ = -wx*sinA + wz*cosA - cfg.cameraR;
          if (camZ >= -0.1) { voxels[vi]=0; carved++; continue; }
          const px = Math.round(fx*camX/(-camZ)+cx);
          const py = Math.round(fx*wy/(-camZ)+cy);
          if (px<0||px>=sw||py<0||py>=sh || !sil[py*sw+px]) { voxels[vi]=0; carved++; }
        }
      }
    }
    totalVoxels -= carved;
    if (carved>0) voxelsDirty = true;
  }

  function colorVoxels(imgData, w, h, yaw) {
    const R = cfg.voxelRes, half = cfg.voxelWorld/2, step = cfg.voxelWorld/R;
    const cosA = Math.cos(yaw), sinA = Math.sin(yaw);
    const fovRad = cfg.fovDeg * Math.PI / 180;
    const fx = w / (2*Math.tan(fovRad/2)), cx = w/2, cy = h/2;
    const d = imgData.data;
    for (let i=0; i<R; i++) {
      const wx = i*step-half;
      for (let j=0; j<R; j++) {
        const wy = j*step-half;
        for (let k=0; k<R; k++) {
          const vi = i*R*R+j*R+k;
          if (!voxels[vi]) continue;
          const wz = k*step-half;
          const cz = -wx*sinA + wz*cosA - cfg.cameraR;
          if (cz>=-0.1) continue;
          const px = Math.round(fx*(wx*cosA+wz*sinA)/(-cz)+cx);
          const py = Math.round(fx*wy/(-cz)+cy);
          if (px<0||px>=w||py<0||py>=h) continue;
          const pi = (py*w+px)*4;
          voxels[vi] = (d[pi]<<16)|(d[pi+1]<<8)|d[pi+2];
        }
      }
    }
  }

  /* ============== Quality Metric ============== */

  function computeQuality(edgeMap, sil, w, h) {
    const cx = Math.floor(w/2), cy = Math.floor(h/2);
    const r = Math.floor(Math.min(w,h)*0.3);
    let edgePixels = 0, silPixels = 0, total = 0;
    for (let y = cy-r; y <= cy+r; y++) {
      for (let x = cx-r; x <= cx+r; x++) {
        if (x<0||x>=w||y<0||y>=h) continue;
        total++;
        if (edgeMap[y*w+x]) edgePixels++;
        if (sil[y*w+x]) silPixels++;
      }
    }
    if (!total) return 0;
    const edgeDensity = edgePixels / total;
    const silCoverage = silPixels / total;
    return Math.min(100, Math.round(
      (Math.min(edgeDensity * 15, 1) * 50) + (silCoverage * 50)
    ));
  }

  /* ============== Overlay ============== */

  function drawOverlay(edgeMap, lines, corners, sil, pw, ph) {
    const ow = overlayCanvas.width, oh = overlayCanvas.height;
    const sx = ow/pw, sy = oh/ph;
    overlayCtx.clearRect(0, 0, ow, oh);

    if (cfg.showSilhouette) {
      overlayCtx.globalAlpha = cfg.overlayOpacity * 0.5;
      overlayCtx.fillStyle = '#34d399';
      for (let y=0; y<ph; y+=2)
        for (let x=0; x<pw; x+=2)
          if (sil[y*pw+x]) overlayCtx.fillRect(x*sx, y*sy, Math.ceil(sx*2), Math.ceil(sy*2));
    }

    if (cfg.showEdges) {
      overlayCtx.globalAlpha = cfg.overlayOpacity;
      overlayCtx.fillStyle = '#00d4ff';
      for (let y=0; y<ph; y++)
        for (let x=0; x<pw; x++)
          if (edgeMap[y*pw+x]) overlayCtx.fillRect(x*sx, y*sy, Math.ceil(sx), Math.ceil(sy));
    }

    overlayCtx.globalAlpha = 1.0;

    if (cfg.showLines) {
      overlayCtx.strokeStyle = '#facc15';
      overlayCtx.lineWidth = 1.5;
      for (const l of lines) {
        overlayCtx.beginPath();
        overlayCtx.moveTo(l.x1*sx, l.y1*sy);
        overlayCtx.lineTo(l.x2*sx, l.y2*sy);
        overlayCtx.stroke();
      }
    }

    if (cfg.showCorners || cfg.showAngles) {
      for (const c of corners) {
        if (cfg.showCorners) {
          overlayCtx.strokeStyle = '#f472b6';
          overlayCtx.lineWidth = 1.5;
          overlayCtx.beginPath();
          overlayCtx.arc(c.x*sx, c.y*sy, 5, 0, Math.PI*2);
          overlayCtx.stroke();
        }
        if (cfg.showAngles) {
          overlayCtx.strokeStyle = 'rgba(244,114,182,0.5)';
          overlayCtx.lineWidth = 1;
          overlayCtx.beginPath();
          overlayCtx.arc(c.x*sx, c.y*sy, 16, c.a1, c.a2);
          overlayCtx.stroke();
          overlayCtx.fillStyle = '#f472b6';
          overlayCtx.font = '600 9px Inter, sans-serif';
          overlayCtx.fillText(c.angle+'°', c.x*sx+8, c.y*sy-8);
        }
      }
    }
  }

  /* ============== 3D Preview ============== */

  function setup3D() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080c16);
    cam3d = new THREE.PerspectiveCamera(50,1,0.01,50);
    cam3d.position.set(0,0.4,2);
    renderer = new THREE.WebGLRenderer({ canvas: previewCanvas, antialias: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    scene.add(new THREE.GridHelper(2,10,0x162030,0x0e1520));
    scene.add(new THREE.AmbientLight(0xffffff,0.6));
    const dl = new THREE.DirectionalLight(0xffffff,0.5); dl.position.set(1,2,1); scene.add(dl);
  }

  function update3DPreview() {
    const showFused = cfg.useTemporalFusion && fusionDirty;
    if (!voxelsDirty && !showFused) return;

    if (showFused) {
      fusionDirty = false;
      const fused = getFusedPointCloud();
      if (fused && fused.positions.length > 0) {
        if (fusionPointObj) { scene.remove(fusionPointObj); fusionPointObj.geometry.dispose(); fusionPointObj.material.dispose(); fusionPointObj = null; }
        ensureVoxelBuffer();
        const n = fused.positions.length / 3;
        updateVoxelBuffer(fused.positions, fused.colors, n);
        if (!fusionPointObj) {
          fusionPointObj = new THREE.Points(voxelBufGeo, voxelBufMat);
          scene.add(fusionPointObj);
        }
      }
    }

    if (!voxelsDirty) return;
    voxelsDirty = false;

    if (!cfg.useTemporalFusion) {
      if (voxelMesh) { scene.remove(voxelMesh); voxelMesh.geometry.dispose(); voxelMesh.material.dispose(); voxelMesh = null; }
      const R = cfg.voxelRes, half = cfg.voxelWorld/2, step = cfg.voxelWorld/R;
      const pos = [], col = [];
      for (let i=0;i<R;i++) for (let j=0;j<R;j++) for (let k=0;k<R;k++) {
        const v = voxels[i*R*R+j*R+k]; if (!v) continue;
        pos.push(i*step-half, j*step-half, k*step-half);
        const r=(v>>16&0xff)/255, g=(v>>8&0xff)/255, b=(v&0xff)/255;
        col.push(r||0.4, g||0.7, b||0.9);
      }
      if (!pos.length) return;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(pos,3));
      geo.setAttribute('color', new THREE.Float32BufferAttribute(col,3));
      voxelMesh = new THREE.Points(geo, new THREE.PointsMaterial({ size: step*0.9, vertexColors:true, sizeAttenuation:true }));
      scene.add(voxelMesh);
    }
  }

  function startRenderLoop() {
    if (renderFrame) return;
    function loop() {
      renderFrame = requestAnimationFrame(loop);

      if (cfg.syncCamera && scanning) {
        const r = cfg.cameraR * 0.7;
        cam3d.position.set(
          r * Math.sin(liveOrientation.yaw),
          0.4 + Math.sin(liveOrientation.pitch) * 0.4,
          r * Math.cos(liveOrientation.yaw)
        );
      } else {
        prevAngle += 0.006;
        cam3d.position.set(Math.sin(prevAngle)*1.8, 0.6, Math.cos(prevAngle)*1.8);
      }
      cam3d.lookAt(0,0,0);

      const p = previewCanvas.parentElement;
      if (p && p.clientWidth>0) {
        renderer.setSize(p.clientWidth, p.clientHeight);
        cam3d.aspect = p.clientWidth/p.clientHeight;
        cam3d.updateProjectionMatrix();
      }
      if (cfg.showVoxels) update3DPreview();
      if (cfg.showWireframe) updateWireframePreview();
      renderer.render(scene, cam3d);
    }
    loop();
  }
  function stopRenderLoop() { if (renderFrame) { cancelAnimationFrame(renderFrame); renderFrame=null; } }
  function updateWireframePreview() {
    if (!wireframeDirty) return;
    wireframeDirty = false;

    if (wireVertexObj) { scene.remove(wireVertexObj); wireVertexObj.geometry.dispose(); wireVertexObj.material.dispose(); wireVertexObj=null; }
    if (wireEdgeObj)   { scene.remove(wireEdgeObj);   wireEdgeObj.geometry.dispose();   wireEdgeObj.material.dispose();   wireEdgeObj=null; }

    if (!vertices3D.length) return;

    const vPos=[], vCol=[];
    for (const v of vertices3D) {
      vPos.push(v.pos[0], v.pos[1], v.pos[2]);
      vCol.push(v.col[0], v.col[1], v.col[2]);
    }
    const vGeo = new THREE.BufferGeometry();
    vGeo.setAttribute('position', new THREE.Float32BufferAttribute(vPos,3));
    vGeo.setAttribute('color', new THREE.Float32BufferAttribute(vCol,3));
    wireVertexObj = new THREE.Points(vGeo, new THREE.PointsMaterial({ size:0.035, vertexColors:true, sizeAttenuation:true }));
    scene.add(wireVertexObj);

    if (edges3D.size) {
      const ePos=[];
      for (const key of edges3D) {
        const a = Math.floor(key/100000), b = key%100000;
        if (a<vertices3D.length && b<vertices3D.length) {
          ePos.push(vertices3D[a].pos[0],vertices3D[a].pos[1],vertices3D[a].pos[2]);
          ePos.push(vertices3D[b].pos[0],vertices3D[b].pos[1],vertices3D[b].pos[2]);
        }
      }
      if (ePos.length) {
        const eGeo = new THREE.BufferGeometry();
        eGeo.setAttribute('position', new THREE.Float32BufferAttribute(ePos,3));
        wireEdgeObj = new THREE.LineSegments(eGeo, new THREE.LineBasicMaterial({ color:0x00d4ff, transparent:true, opacity:0.7 }));
        scene.add(wireEdgeObj);
      }
    }
  }

  function clear3DScene() {
    if (voxelMesh)      { scene.remove(voxelMesh);      voxelMesh.geometry.dispose();      voxelMesh.material.dispose();      voxelMesh=null; }
    if (wireVertexObj)  { scene.remove(wireVertexObj);  wireVertexObj.geometry.dispose();  wireVertexObj.material.dispose();  wireVertexObj=null; }
    if (wireEdgeObj)    { scene.remove(wireEdgeObj);    wireEdgeObj.geometry.dispose();    wireEdgeObj.material.dispose();    wireEdgeObj=null; }
    if (fusionPointObj) { scene.remove(fusionPointObj); fusionPointObj.geometry.dispose(); fusionPointObj.material.dispose(); fusionPointObj=null; }
    if (voxelBufGeo)    { voxelBufGeo.dispose(); voxelBufGeo=null; }
    if (voxelBufMat)    { voxelBufMat.dispose(); voxelBufMat=null; }
  }

  /* ============== Frame Loop ============== */

  function scheduleFrame() {
    if (!scanning||paused||processing) return;
    const wait = Math.max(0, cfg.minFrameMs - (performance.now()-lastProcTime));
    setTimeout(() => { if (scanning&&!paused) processFrame(); }, wait);
  }

  function processFrame() {
    if (!scanning||paused||processing) return;
    if (!videoEl||videoEl.readyState<2) { setTimeout(scheduleFrame,200); return; }
    processing = true;
    const t0 = performance.now();

    try {
      const vw = videoEl.videoWidth, vh = videoEl.videoHeight;
      const scale = cfg.procSize / Math.max(vw, vh);
      const pw = Math.round(vw*scale), ph = Math.round(vh*scale);

      const c = document.createElement('canvas'); c.width=pw; c.height=ph;
      const ctx = c.getContext('2d'); ctx.drawImage(videoEl,0,0,pw,ph);
      const imgData = ctx.getImageData(0,0,pw,ph);

      let gray = toGray(imgData.data, pw*ph);
      if (cfg.sharpen) gray = unsharpMask(gray, pw, ph, cfg.sharpenAmount);

      const edgeMap = cannyEdges(gray, pw, ph);
      const lines = detectLines(edgeMap, pw, ph);
      const corners = detectCornersAndAngles(lines);
      const sil = extractSilhouette(gray, pw, ph);

      computeOpticalFlowMotion(prevGray, gray, pw, ph, edgeMap);
      prevGray = gray;

      edgeCount = lines.length;
      cornerCount = corners.length;
      quality = computeQuality(edgeMap, sil, pw, ph);

      const ori = getSmoothedOrientation();
      liveOrientation = ori;

      processVertexTracking(corners, lines, gray, pw, ph, ori.yaw, imgData);

      if (lastYaw !== null) {
        const delta = Math.abs(ori.yaw - lastYaw);
        if (delta > cfg.motionThreshold || frameCount < 4) {
          carveVisualHull(sil, pw, ph, ori.yaw);
          colorVoxels(imgData, pw, ph, ori.yaw);
        }
      } else {
        carveVisualHull(sil, pw, ph, ori.yaw);
        colorVoxels(imgData, pw, ph, ori.yaw);
      }
      lastYaw = ori.yaw;

      if (cfg.useTemporalFusion && voxelsDirty) {
        const cloud = getPointCloud();
        if (cloud.positions.length > 0) {
          const fused = getFusedPointCloud();
          let aligned = cloud.positions;
          if (cfg.useICP && fused && fused.positions.length > 9) {
            aligned = alignICP(cloud.positions, fused.positions, fused.positions.length / 3);
          }
          fusePoints(aligned, cloud.colors, frameCount);
        }
      }

      resizeOverlay();
      drawOverlay(edgeMap, lines, corners, sil, pw, ph);

      frameCount++;
      const elapsed = performance.now() - t0;
      fps = Math.round(1000 / elapsed);
      lastProcTime = performance.now();

      adaptProcSize(fps);

      emitStats();

    } catch (err) { console.error('Frame error:', err); }

    processing = false;
    scheduleFrame();
  }

  /* ============== Orientation ============== */

  function getSmoothedOrientation() {
    let yaw, pitch;
    if (typeof CameraModule !== 'undefined' && CameraModule.getOrientation) {
      const o = CameraModule.getOrientation();
      yaw   = (o.alpha||0) * Math.PI/180;
      pitch = ((o.beta||90)-90) * Math.PI/180;
    } else {
      yaw   = performance.now()/1000 * 0.3;
      pitch = 0;
    }

    if (cfg.useOpticalFlow && opticalFlowDelta.count > 5) {
      const flowYawCorrection = opticalFlowDelta.dx * 0.0003;
      yaw += flowYawCorrection;
    }

    if (cfg.useKalman) {
      if (!kfYaw) kfYaw = makeKalman1D(cfg.kalmanQ, cfg.kalmanR, yaw);
      if (!kfPitch) kfPitch = makeKalman1D(cfg.kalmanQ, cfg.kalmanR, pitch);
      kfYaw.q = cfg.kalmanQ; kfYaw.r = cfg.kalmanR;
      kfPitch.q = cfg.kalmanQ; kfPitch.r = cfg.kalmanR;
      yaw   = kalmanUpdate(kfYaw, yaw);
      pitch = kalmanUpdate(kfPitch, pitch);
      return { yaw, pitch };
    }

    if (cfg.smoothOrientation) {
      const f = cfg.smoothFactor;
      smoothYaw   = smoothYaw  * (1-f) + yaw   * f;
      smoothPitch = smoothPitch* (1-f) + pitch * f;
      return { yaw: smoothYaw, pitch: smoothPitch };
    }
    return { yaw, pitch };
  }

  function resizeOverlay() {
    if (!overlayCanvas||!videoEl) return;
    const r = videoEl.getBoundingClientRect();
    if (overlayCanvas.width!==r.width||overlayCanvas.height!==r.height) {
      overlayCanvas.width = r.width; overlayCanvas.height = r.height;
    }
  }

  function emitStats() { if (onStatsUpdate) onStatsUpdate(getStats()); }
  function emitStatus(s) { if (onStatusChange) onStatusChange(s); }

  return {
    init, startCamera, stopCamera,
    startScanning, pauseScanning, resumeScanning, stopScanning,
    reset, destroy, getPointCloud, getStats,
    isScanning, isPaused, isActive, stopRenderLoop,
    set, get, getAll, applyPreset,
  };
})();
