/**
 * App controller — navigation, state management, event wiring.
 */
(function () {
  'use strict';

  /* ===== DOM References ===== */
  const $ = id => document.getElementById(id);

  const splash = $('splash');
  const app = $('app');
  const navItems = document.querySelectorAll('.nav-item');

  // Scan tab
  const scanIntro = $('scanIntro');
  const scanCamera = $('scanCamera');
  const cameraPreview = $('cameraPreview');
  const btnStartCamera = $('btnStartCamera');
  const btnCapture = $('btnCapture');
  const btnFlip = $('btnFlip');
  const btnStopCamera = $('btnStopCamera');
  const photoStrip = $('photoStrip');
  const hudAngle = $('hudAngle');
  const hudPhotos = $('hudPhotos');

  // Gallery tab
  const galleryEmpty = $('galleryEmpty');
  const galleryGrid = $('galleryGrid');
  const galleryActions = $('galleryActions');
  const galleryBadge = $('galleryBadge');
  const btnClearGallery = $('btnClearGallery');
  const btnProcess = $('btnProcess');

  // Process tab
  const processIdle = $('processIdle');
  const processActive = $('processActive');
  const processBar = $('processBar');
  const processPercent = $('processPercent');
  const processSteps = $('processSteps');
  const processStats = $('processStats');
  const statPoints = $('statPoints');
  const statTriangles = $('statTriangles');
  const statTime = $('statTime');
  const btnViewResult = $('btnViewResult');

  // Viewer tab
  const viewerEmpty = $('viewerEmpty');
  const viewerContainer = $('viewerContainer');
  const viewerCanvas = $('viewer3d');
  const viewerInfo = $('viewerInfo');
  const viewerActions = $('viewerActions');
  const btnResetView = $('btnResetView');
  const btnToggleMode = $('btnToggleMode');
  const btnExportOBJ = $('btnExportOBJ');
  const btnExportPLY = $('btnExportPLY');
  const btnNewScan = $('btnNewScan');

  // Modal
  const photoModal = $('photoModal');
  const modalPhoto = $('modalPhoto');
  const btnDeletePhoto = $('btnDeletePhoto');
  const btnCloseModal = $('btnCloseModal');

  let currentTab = 'tabScan';
  let processing = false;
  let modalPhotoId = null;
  let orientationInterval = null;
  let tipInterval = null;
  let aiMode = true;

  /* ===== Initialization ===== */
  window.addEventListener('DOMContentLoaded', () => {
    CameraModule.init(cameraPreview);

    setTimeout(() => {
      splash.classList.add('fade-out');
      app.classList.remove('hidden');
      setTimeout(() => splash.remove(), 600);
    }, 1800);

    bindEvents();
    initAIMode();
  });

  /* ===== Event Binding ===== */
  function bindEvents() {
    // Navigation
    navItems.forEach(btn => {
      btn.addEventListener('click', () => switchTab(btn.dataset.tab));
    });

    // Scan tab
    btnStartCamera.addEventListener('click', startCamera);
    btnCapture.addEventListener('click', capturePhoto);
    btnFlip.addEventListener('click', flipCamera);
    btnStopCamera.addEventListener('click', stopCamera);

    // File upload fallback
    $('btnLoadFiles').addEventListener('click', () => $('fileInput').click());
    $('fileInput').addEventListener('change', handleFileUpload);

    // AI mode toggle
    $('aiToggle').addEventListener('change', (e) => {
      aiMode = e.target.checked;
      updateAIStatus();
    });

    // Gallery tab
    btnClearGallery.addEventListener('click', clearGallery);
    btnProcess.addEventListener('click', startProcessing);

    // Process tab
    btnViewResult.addEventListener('click', () => {
      switchTab('tabViewer');
      showViewer();
    });

    // Viewer tab
    btnResetView.addEventListener('click', () => ViewerModule.resetView());
    btnToggleMode.addEventListener('click', toggleViewerMode);
    $('btnPointSizeUp').addEventListener('click', () => ViewerModule.adjustPointSize(1.4));
    $('btnPointSizeDown').addEventListener('click', () => ViewerModule.adjustPointSize(0.7));
    btnExportOBJ.addEventListener('click', exportOBJ);
    btnExportPLY.addEventListener('click', exportPLY);
    btnNewScan.addEventListener('click', newScan);

    // Modal
    btnCloseModal.addEventListener('click', closeModal);
    btnDeletePhoto.addEventListener('click', deleteModalPhoto);
    photoModal.querySelector('.modal-backdrop').addEventListener('click', closeModal);

    // Resize
    window.addEventListener('resize', () => {
      if (ViewerModule.isInitialized()) ViewerModule.resize();
    });
  }

  /* ===== Tab Navigation ===== */
  function switchTab(tabId) {
    if (currentTab === tabId) return;
    currentTab = tabId;

    document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
    const target = $(tabId);
    if (target) target.classList.add('active');

    navItems.forEach(btn => {
      btn.classList.toggle('active', btn.dataset.tab === tabId);
    });

    if (tabId === 'tabGallery') refreshGallery();

    if (tabId === 'tabViewer') {
      if (ViewerModule.isInitialized()) {
        ViewerModule.resize();
        ViewerModule.startLoop();
      }
    } else if (ViewerModule.isInitialized()) {
      ViewerModule.stopLoop();
    }
  }

  /* ===== AI Mode ===== */
  async function initAIMode() {
    const toggle = $('aiToggle');
    aiMode = toggle.checked;

    try {
      const supported = await AIEngine.checkSupport();
      if (!supported) {
        toggle.checked = false;
        toggle.disabled = true;
        aiMode = false;
        updateAIStatusText('AI niedostępne w tej przeglądarce', 'off');
      } else {
        updateAIStatusText('Gotowy do pobrania (~27 MB, cache po 1. uruchomieniu)', 'waiting');
      }
    } catch {
      toggle.checked = false;
      toggle.disabled = true;
      aiMode = false;
      updateAIStatusText('AI niedostępne', 'off');
    }
  }

  function updateAIStatus() {
    if (aiMode) {
      if (AIEngine.isReady()) {
        updateAIStatusText('Model AI załadowany i gotowy', 'ready');
      } else {
        updateAIStatusText('Gotowy do pobrania (~27 MB, cache po 1. uruchomieniu)', 'waiting');
      }
    } else {
      updateAIStatusText('Tryb klasyczny (bez AI)', 'off');
    }
  }

  function updateAIStatusText(text, state) {
    const statusEl = $('aiModelStatus');
    if (!statusEl) return;
    statusEl.className = 'ai-model-status';
    if (state === 'ready') statusEl.classList.add('ready');
    if (state === 'loading') statusEl.classList.add('loading');
    statusEl.querySelector('span:last-child').textContent = text;
  }

  function showAIOverlay(show) {
    const overlay = $('aiOverlay');
    if (show) {
      overlay.classList.remove('hidden');
    } else {
      overlay.classList.add('hidden');
    }
  }

  function updateAIOverlay(message, progress) {
    const msg = $('aiOverlayMessage');
    const bar = $('aiOverlayBar');
    if (msg) msg.textContent = message;
    if (bar) bar.style.width = progress + '%';
  }

  /* ===== Camera ===== */
  async function startCamera() {
    const ok = await CameraModule.start();
    if (!ok) {
      toast('Nie udało się uruchomić kamery', 'error');
      return;
    }
    scanIntro.classList.add('hidden');
    scanCamera.classList.remove('hidden');
    updatePhotoHud();
    startOrientationHud();
    startTipsCarousel();
    toast('Kamera aktywna — rób zdjęcia dookoła obiektu', 'info');
  }

  function stopCamera() {
    CameraModule.stop();
    scanCamera.classList.add('hidden');
    scanIntro.classList.remove('hidden');
    stopOrientationHud();
    stopTipsCarousel();
  }

  async function flipCamera() {
    const ok = await CameraModule.flip();
    if (!ok) toast('Nie udało się przełączyć kamery', 'error');
  }

  function capturePhoto() {
    const photo = CameraModule.capture();
    if (!photo) {
      toast('Osiągnięto limit zdjęć', 'warning');
      return;
    }

    if (navigator.vibrate) navigator.vibrate(30);

    btnCapture.classList.add('flash');
    setTimeout(() => btnCapture.classList.remove('flash'), 350);

    addPhotoToStrip(photo);
    updatePhotoHud();
    updateGalleryBadge();
    updateCoverageRing();

    const count = CameraModule.getPhotos().length;
    if (count === 3) {
      toast('Dobra robota! Kontynuuj obrót dookoła obiektu', 'success');
    } else if (count === 12) {
      toast('Świetnie! Masz 12 zdjęć — możesz przetwarzać lub robić więcej', 'success');
    }
  }

  function addPhotoToStrip(photo) {
    const div = document.createElement('div');
    div.className = 'photo-strip-item';
    div.innerHTML = `<img src="${photo.thumbUrl}" alt=""><div class="strip-num">${CameraModule.getPhotos().length}</div>`;
    photoStrip.appendChild(div);
    photoStrip.scrollLeft = photoStrip.scrollWidth;
  }

  function updatePhotoHud() {
    const count = CameraModule.getPhotos().length;
    hudPhotos.querySelector('span').textContent = `${count} / ${CameraModule.MAX_PHOTOS}`;
  }

  function startOrientationHud() {
    orientationInterval = setInterval(() => {
      const o = CameraModule.getOrientation();
      hudAngle.querySelector('span').textContent = `${Math.round(o.alpha)}°`;
      updateCoverageRing();
    }, 200);
  }

  function updateCoverageRing() {
    const progress = CameraModule.getAngleProgress();
    const arc = document.getElementById('coverageArc');
    const text = document.getElementById('coverageText');
    if (arc && text) {
      const circumference = 2 * Math.PI * 19;
      const offset = circumference - (progress.coverage / 100) * circumference;
      arc.style.strokeDashoffset = offset;
      text.textContent = progress.coverage + '%';
    }
  }

  function stopOrientationHud() {
    if (orientationInterval) {
      clearInterval(orientationInterval);
      orientationInterval = null;
    }
  }

  async function handleFileUpload(e) {
    const files = Array.from(e.target.files);
    if (!files.length) return;

    toast(`Wczytywanie ${files.length} zdjęć...`, 'info');

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const dataUrl = await readFileAsDataUrl(file);
      const img = await loadImageEl(dataUrl);

      const canvas = document.createElement('canvas');
      canvas.width = img.width;
      canvas.height = img.height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0);
      const fullDataUrl = canvas.toDataURL('image/jpeg', 0.85);

      const thumbCanvas = document.createElement('canvas');
      thumbCanvas.width = 128;
      thumbCanvas.height = 128;
      const tCtx = thumbCanvas.getContext('2d');
      const size = Math.min(img.width, img.height);
      const sx = (img.width - size) / 2;
      const sy = (img.height - size) / 2;
      tCtx.drawImage(canvas, sx, sy, size, size, 0, 0, 128, 128);
      const thumbUrl = thumbCanvas.toDataURL('image/jpeg', 0.7);

      const angle = (i / files.length) * 360;
      const photo = {
        id: Date.now() + '-file-' + i,
        dataUrl: fullDataUrl,
        thumbUrl,
        width: img.width,
        height: img.height,
        orientation: { alpha: angle, beta: 0, gamma: 0 },
        timestamp: Date.now(),
      };
      CameraModule.getPhotos().push(photo);
    }

    updateGalleryBadge();
    toast(`Wczytano ${files.length} zdjęć — przejdź do Galerii`, 'success');
    e.target.value = '';
  }

  function readFileAsDataUrl(file) {
    return new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (ev) => resolve(ev.target.result);
      reader.readAsDataURL(file);
    });
  }

  function loadImageEl(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }

  function startTipsCarousel() {
    const tips = document.querySelectorAll('#scanTips .tip-item');
    if (!tips.length) return;
    let idx = 0;
    tipInterval = setInterval(() => {
      tips[idx].classList.remove('active');
      idx = (idx + 1) % tips.length;
      tips[idx].classList.add('active');
    }, 4000);
  }

  function stopTipsCarousel() {
    if (tipInterval) {
      clearInterval(tipInterval);
      tipInterval = null;
    }
  }

  /* ===== Gallery ===== */
  function refreshGallery() {
    const photos = CameraModule.getPhotos();
    if (photos.length === 0) {
      galleryEmpty.classList.remove('hidden');
      galleryGrid.classList.add('hidden');
      galleryActions.classList.add('hidden');
      return;
    }

    galleryEmpty.classList.add('hidden');
    galleryGrid.classList.remove('hidden');
    galleryActions.classList.remove('hidden');

    galleryGrid.innerHTML = '';
    photos.forEach((photo, i) => {
      const div = document.createElement('div');
      div.className = 'gallery-item';
      const angle = Math.round(photo.orientation.alpha || 0);
      div.innerHTML = `
        <img src="${photo.thumbUrl}" alt="Zdjęcie ${i + 1}">
        <div class="gallery-num">${i + 1}</div>
        <div class="gallery-angle">${angle}°</div>
      `;
      div.addEventListener('click', () => openModal(photo));
      galleryGrid.appendChild(div);
    });
  }

  function updateGalleryBadge() {
    const count = CameraModule.getPhotos().length;
    if (count > 0) {
      galleryBadge.textContent = count;
      galleryBadge.classList.remove('hidden');
    } else {
      galleryBadge.classList.add('hidden');
    }
  }

  function clearGallery() {
    const count = CameraModule.getPhotos().length;
    if (count === 0) return;
    if (!confirm(`Usunąć wszystkie ${count} zdjęć?`)) return;

    CameraModule.clearPhotos();
    photoStrip.innerHTML = '';
    updatePhotoHud();
    updateGalleryBadge();
    refreshGallery();
    toast('Galeria wyczyszczona', 'info');
  }

  /* ===== Photo Modal ===== */
  function openModal(photo) {
    modalPhotoId = photo.id;
    modalPhoto.src = photo.dataUrl;
    photoModal.classList.remove('hidden');
  }

  function closeModal() {
    photoModal.classList.add('hidden');
    modalPhotoId = null;
  }

  function deleteModalPhoto() {
    if (!modalPhotoId) return;
    CameraModule.removePhoto(modalPhotoId);
    closeModal();
    refreshGallery();
    rebuildPhotoStrip();
    updatePhotoHud();
    updateGalleryBadge();
    toast('Zdjęcie usunięte', 'info');
  }

  function rebuildPhotoStrip() {
    photoStrip.innerHTML = '';
    CameraModule.getPhotos().forEach((photo, i) => {
      const div = document.createElement('div');
      div.className = 'photo-strip-item';
      div.innerHTML = `<img src="${photo.thumbUrl}" alt=""><div class="strip-num">${i + 1}</div>`;
      photoStrip.appendChild(div);
    });
  }

  /* ===== Processing ===== */
  async function startProcessing() {
    const photos = CameraModule.getPhotos();
    if (photos.length < 3) {
      toast('Potrzeba min. 3 zdjęć do rekonstrukcji', 'error');
      return;
    }

    processing = true;
    switchTab('tabProcess');
    processIdle.classList.add('hidden');
    processActive.classList.remove('hidden');
    processStats.classList.add('hidden');
    btnViewResult.classList.add('hidden');
    processSteps.innerHTML = '';

    let result;

    if (aiMode && AIEngine.isSupported() !== false) {
      if (!AIEngine.isReady()) {
        showAIOverlay(true);
        updateAIOverlay('Przygotowanie modelu AI...', 0);
        updateAIStatusText('Pobieranie modelu...', 'loading');
        try {
          await AIEngine.initialize((info) => {
            updateAIOverlay(info.message, info.progress);
          });
          showAIOverlay(false);
          updateAIStatusText('Model AI załadowany i gotowy', 'ready');
          toast('Model AI załadowany — rozpoczynam rekonstrukcję', 'success');
        } catch (err) {
          showAIOverlay(false);
          aiMode = false;
          $('aiToggle').checked = false;
          updateAIStatusText('Błąd ładowania AI — tryb klasyczny', 'off');
          toast('AI niedostępne — używam trybu klasycznego', 'error');
        }
      }

      if (aiMode && AIEngine.isReady()) {
        result = await ProcessingModule.runAI(photos, onProcessProgress);
      } else {
        result = await ProcessingModule.run(photos, onProcessProgress);
      }
    } else {
      result = await ProcessingModule.run(photos, onProcessProgress);
    }

    processing = false;

    if (!result) {
      toast('Przetwarzanie anulowane', 'error');
      return;
    }

    processStats.classList.remove('hidden');
    statPoints.textContent = result.stats.points.toLocaleString();
    statTriangles.textContent = result.stats.triangles.toLocaleString();
    statTime.textContent = result.stats.time + 's';
    btnViewResult.classList.remove('hidden');

    const mode = aiMode && AIEngine.isReady() ? 'AI' : 'klasyczny';
    toast(`Model 3D gotowy! (tryb ${mode})`, 'success');
  }

  function onProcessProgress({ stepIdx, stepName, steps, percent }) {
    processPercent.textContent = percent + '%';
    processBar.style.width = percent + '%';

    processSteps.innerHTML = '';
    steps.forEach((name, i) => {
      const div = document.createElement('div');
      div.className = 'process-step';
      if (i < stepIdx) div.classList.add('done');
      else if (i === stepIdx) div.classList.add('active');

      const icon = i < stepIdx ? '✓' : i === stepIdx ? '◌' : '·';
      div.innerHTML = `<div class="step-icon">${icon}</div><span>${name}</span>`;
      processSteps.appendChild(div);
    });
  }

  /* ===== 3D Viewer ===== */
  function showViewer() {
    const result = ProcessingModule.getResult();
    if (!result) return;

    viewerEmpty.classList.add('hidden');
    viewerContainer.classList.remove('hidden');
    viewerActions.classList.remove('hidden');

    if (!ViewerModule.isInitialized()) {
      ViewerModule.init(viewerCanvas);
    }

    ViewerModule.loadPointCloud(result.pointCloud.positions, result.pointCloud.colors);
    ViewerModule.loadMesh(
      result.pointCloud.positions,
      result.pointCloud.colors,
      result.mesh.indices
    );

    ViewerModule.resize();
    ViewerModule.startLoop();
    updateViewerInfo();
  }

  function toggleViewerMode() {
    ViewerModule.toggleMode();
    updateViewerInfo();
  }

  function updateViewerInfo() {
    const info = ViewerModule.getInfo();
    viewerInfo.textContent =
      `${info.mode} · ${info.points.toLocaleString()} pkt · ${info.triangles.toLocaleString()} trój.`;
  }

  /* ===== Export ===== */
  function exportOBJ() {
    const result = ProcessingModule.getResult();
    if (!result) return;
    const data = ProcessingModule.exportOBJ(result.pointCloud, result.mesh);
    downloadFile(data, 'scan3d-model.obj', 'text/plain');
    toast('Wyeksportowano OBJ', 'success');
  }

  function exportPLY() {
    const result = ProcessingModule.getResult();
    if (!result) return;
    const data = ProcessingModule.exportPLY(result.pointCloud, result.mesh);
    downloadFile(data, 'scan3d-model.ply', 'text/plain');
    toast('Wyeksportowano PLY', 'success');
  }

  function downloadFile(content, filename, mimeType) {
    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  }

  /* ===== New Scan ===== */
  function newScan() {
    ViewerModule.clearScene();
    ViewerModule.stopLoop();
    CameraModule.clearPhotos();
    photoStrip.innerHTML = '';
    updateGalleryBadge();

    viewerContainer.classList.add('hidden');
    viewerActions.classList.add('hidden');
    viewerEmpty.classList.remove('hidden');
    processActive.classList.add('hidden');
    processIdle.classList.remove('hidden');

    switchTab('tabScan');
    toast('Nowy skan — zrób zdjęcia obiektu', 'info');
  }

  /* ===== Toast Notifications ===== */
  function toast(message, type = 'info') {
    const container = $('toastContainer');
    const el = document.createElement('div');
    el.className = `toast toast-${type}`;
    el.textContent = message;
    container.appendChild(el);

    setTimeout(() => {
      el.classList.add('toast-out');
      setTimeout(() => el.remove(), 300);
    }, 2800);
  }
})();
