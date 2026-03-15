/**
 * RealtimeScanner — live 3D scanning from camera feed.
 * Captures frames continuously, runs AI depth estimation,
 * and accumulates a colored point cloud in real time.
 */
const RealtimeScanner = (() => {
  /* ===== Configuration ===== */
  const FRAME_SIZE = 256;
  const SAMPLE_STRIDE = 3;
  const VOXEL_SIZE = 0.012;
  const MAX_POINTS = 400000;
  const FOV_RAD = 60 * Math.PI / 180;
  const NEAR = 0.2;
  const FAR = 2.8;
  const MIN_FRAME_INTERVAL = 150;

  /* ===== State ===== */
  let videoEl = null;
  let previewCanvas = null;
  let stream = null;
  let scene, camera, renderer;
  let pointCloudObj = null;

  let scanning = false;
  let paused = false;
  let processing = false;

  let accumPositions = [];
  let accumColors = [];
  let voxelGrid = new Map();

  let frameCount = 0;
  let totalPoints = 0;
  let fps = 0;
  let lastProcessTime = 0;

  let renderFrame = null;
  let previewAngle = 0;

  let onStatsUpdate = null;
  let onStatusChange = null;

  /* ===== Public API ===== */

  function init(video, canvas3d, callbacks) {
    videoEl = video;
    previewCanvas = canvas3d;
    onStatsUpdate = callbacks?.onStats || null;
    onStatusChange = callbacks?.onStatus || null;
    setup3DPreview();
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
      console.error('Live camera failed:', e);
      return false;
    }
  }

  function stopCamera() {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    if (videoEl) videoEl.srcObject = null;
  }

  async function startScanning() {
    if (!AIEngine.isReady()) throw new Error('AI not initialized');
    scanning = true;
    paused = false;
    processing = false;
    frameCount = 0;
    totalPoints = 0;
    accumPositions = [];
    accumColors = [];
    voxelGrid = new Map();
    clearPreviewCloud();
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
    emitStatus('stopped');
  }

  function reset() {
    stopScanning();
    frameCount = 0;
    totalPoints = 0;
    accumPositions = [];
    accumColors = [];
    voxelGrid = new Map();
    clearPreviewCloud();
    emitStats();
  }

  function destroy() {
    stopScanning();
    stopCamera();
    stopRenderLoop();
    clearPreviewCloud();
    accumPositions = [];
    accumColors = [];
    voxelGrid = new Map();
  }

  function getPointCloud() {
    return {
      positions: new Float32Array(accumPositions),
      colors: new Float32Array(accumColors),
    };
  }

  function getStats() {
    return { frames: frameCount, points: totalPoints, fps, scanning, paused };
  }

  function isScanning() { return scanning; }
  function isPaused() { return paused; }
  function isActive() { return stream !== null; }

  /* ===== Frame Processing ===== */

  function scheduleFrame() {
    if (!scanning || paused || processing) return;
    const elapsed = performance.now() - lastProcessTime;
    const delay = Math.max(0, MIN_FRAME_INTERVAL - elapsed);
    setTimeout(() => { if (scanning && !paused) processFrame(); }, delay);
  }

  async function processFrame() {
    if (!scanning || paused || processing) return;
    if (!videoEl || videoEl.readyState < 2) {
      setTimeout(scheduleFrame, 300);
      return;
    }

    processing = true;
    const t0 = performance.now();

    try {
      const vw = videoEl.videoWidth;
      const vh = videoEl.videoHeight;
      const scale = FRAME_SIZE / Math.max(vw, vh);
      const w = Math.round(vw * scale);
      const h = Math.round(vh * scale);

      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(videoEl, 0, 0, w, h);
      const imageData = ctx.getImageData(0, 0, w, h);

      const blobUrl = await AIEngine.canvasToBlobUrl(canvas);
      const result = await AIEngine.estimateDepth(blobUrl);
      URL.revokeObjectURL(blobUrl);

      const depthMap = AIEngine.extractDepthMap(result);
      const orientation = getOrientation();

      addDepthPoints(imageData, w, h, depthMap, orientation);
      updatePreviewCloud();

      frameCount++;
      const dt = performance.now() - t0;
      fps = Math.round(1000 / dt);
      lastProcessTime = performance.now();
      emitStats();

    } catch (err) {
      console.error('Realtime frame error:', err);
    }

    processing = false;
    scheduleFrame();
  }

  function getOrientation() {
    if (typeof CameraModule !== 'undefined' && CameraModule.getOrientation) {
      const o = CameraModule.getOrientation();
      return {
        yaw: (o.alpha || 0) * Math.PI / 180,
        pitch: ((o.beta || 90) - 90) * Math.PI / 180,
        roll: (o.gamma || 0) * Math.PI / 180,
      };
    }
    const t = performance.now() / 1000;
    return { yaw: t * 0.3, pitch: 0, roll: 0 };
  }

  function addDepthPoints(imageData, iw, ih, depthMap, orientation) {
    const dw = depthMap.width;
    const dh = depthMap.height;
    const fx = dw / (2 * Math.tan(FOV_RAD / 2));
    const fy = fx;
    const cx = dw / 2;
    const cy = dh / 2;

    const cosY = Math.cos(orientation.yaw);
    const sinY = Math.sin(orientation.yaw);
    const cosP = Math.cos(orientation.pitch);
    const sinP = Math.sin(orientation.pitch);

    const bgThreshold = computeBgThreshold(depthMap.data);

    let added = 0;

    for (let y = 0; y < dh; y += SAMPLE_STRIDE) {
      for (let x = 0; x < dw; x += SAMPLE_STRIDE) {
        if (totalPoints + added >= MAX_POINTS) return;

        const d = depthMap.data[y * dw + x];
        if (d < bgThreshold) continue;

        const z = NEAR + (1.0 - d) * (FAR - NEAR);
        const camX = (x - cx) / fx * z;
        const camY = -(y - cy) / fy * z;
        const camZ = -z;

        const ty = camY * cosP - camZ * sinP;
        const tz = camY * sinP + camZ * cosP;

        const wx = camX * cosY - tz * sinY;
        const wz = camX * sinY + tz * cosY;

        const vx = Math.floor(wx / VOXEL_SIZE);
        const vy = Math.floor(ty / VOXEL_SIZE);
        const vz = Math.floor(wz / VOXEL_SIZE);
        const key = (vx * 73856093) ^ (vy * 19349663) ^ (vz * 83492791);
        if (voxelGrid.has(key)) continue;
        voxelGrid.set(key, 1);

        const imgX = Math.min(iw - 1, Math.max(0, Math.floor(x * (iw / dw))));
        const imgY = Math.min(ih - 1, Math.max(0, Math.floor(y * (ih / dh))));
        const pIdx = (imgY * iw + imgX) * 4;

        accumPositions.push(wx, ty, wz);
        accumColors.push(
          imageData.data[pIdx] / 255,
          imageData.data[pIdx + 1] / 255,
          imageData.data[pIdx + 2] / 255
        );
        added++;
      }
    }

    totalPoints += added;
  }

  function computeBgThreshold(data) {
    let sum = 0;
    for (let i = 0; i < data.length; i++) sum += data[i];
    const mean = sum / data.length;
    return Math.max(0.1, mean * 0.45);
  }

  /* ===== 3D Preview ===== */

  function setup3DPreview() {
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x080c16);

    camera = new THREE.PerspectiveCamera(50, 1, 0.01, 50);
    camera.position.set(0, 0.4, 2);

    renderer = new THREE.WebGLRenderer({
      canvas: previewCanvas,
      antialias: true,
      alpha: false,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const grid = new THREE.GridHelper(3, 15, 0x162030, 0x0e1520);
    scene.add(grid);

    const light = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(light);
  }

  function updatePreviewCloud() {
    clearPreviewCloud();
    if (accumPositions.length === 0) return;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(accumPositions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(accumColors, 3));

    const n = accumPositions.length / 3;
    const ptSize = n > 20000 ? 0.004 : n > 5000 ? 0.007 : 0.012;

    const material = new THREE.PointsMaterial({
      size: ptSize,
      vertexColors: true,
      sizeAttenuation: true,
    });

    pointCloudObj = new THREE.Points(geometry, material);
    scene.add(pointCloudObj);
  }

  function clearPreviewCloud() {
    if (pointCloudObj) {
      scene.remove(pointCloudObj);
      pointCloudObj.geometry.dispose();
      pointCloudObj.material.dispose();
      pointCloudObj = null;
    }
  }

  function startRenderLoop() {
    if (renderFrame) return;
    function loop() {
      renderFrame = requestAnimationFrame(loop);
      previewAngle += 0.008;
      camera.position.x = Math.sin(previewAngle) * 2;
      camera.position.z = Math.cos(previewAngle) * 2;
      camera.position.y = 0.5 + Math.sin(previewAngle * 0.5) * 0.2;
      camera.lookAt(0, 0, 0);

      const parent = previewCanvas.parentElement;
      if (parent) {
        const w = parent.clientWidth;
        const h = parent.clientHeight;
        if (w > 0 && h > 0) {
          renderer.setSize(w, h);
          camera.aspect = w / h;
          camera.updateProjectionMatrix();
        }
      }

      renderer.render(scene, camera);
    }
    loop();
  }

  function stopRenderLoop() {
    if (renderFrame) {
      cancelAnimationFrame(renderFrame);
      renderFrame = null;
    }
  }

  /* ===== Helpers ===== */

  function emitStats() {
    if (onStatsUpdate) onStatsUpdate(getStats());
  }

  function emitStatus(status) {
    if (onStatusChange) onStatusChange(status);
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
  };
})();
