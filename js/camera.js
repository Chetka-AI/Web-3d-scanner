/**
 * Camera module — handles MediaStream, photo capture,
 * device orientation tracking, and thumbnail management.
 */
const CameraModule = (() => {
  let stream = null;
  let videoEl = null;
  let facingMode = 'environment';
  let photos = [];
  let orientation = { alpha: 0, beta: 0, gamma: 0 };
  let orientationEnabled = false;

  const MAX_PHOTOS = 36;
  const CAPTURE_WIDTH = 1280;
  const CAPTURE_HEIGHT = 960;

  function init(videoElement) {
    videoEl = videoElement;
  }

  async function start() {
    if (stream) stop();

    const constraints = {
      video: {
        facingMode,
        width: { ideal: CAPTURE_WIDTH },
        height: { ideal: CAPTURE_HEIGHT },
      },
      audio: false,
    };

    try {
      stream = await navigator.mediaDevices.getUserMedia(constraints);
      videoEl.srcObject = stream;
      await videoEl.play();
      startOrientation();
      return true;
    } catch (err) {
      console.error('Camera access failed:', err);
      return false;
    }
  }

  function stop() {
    if (stream) {
      stream.getTracks().forEach(t => t.stop());
      stream = null;
    }
    if (videoEl) {
      videoEl.srcObject = null;
    }
    stopOrientation();
  }

  async function flip() {
    facingMode = facingMode === 'environment' ? 'user' : 'environment';
    return start();
  }

  function capture() {
    if (!videoEl || !stream) return null;
    if (photos.length >= MAX_PHOTOS) return null;

    const canvas = document.createElement('canvas');
    const w = videoEl.videoWidth;
    const h = videoEl.videoHeight;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(videoEl, 0, 0, w, h);

    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);

    const thumbCanvas = document.createElement('canvas');
    thumbCanvas.width = 128;
    thumbCanvas.height = 128;
    const tCtx = thumbCanvas.getContext('2d');
    const size = Math.min(w, h);
    const sx = (w - size) / 2;
    const sy = (h - size) / 2;
    tCtx.drawImage(canvas, sx, sy, size, size, 0, 0, 128, 128);
    const thumbUrl = thumbCanvas.toDataURL('image/jpeg', 0.7);

    const photo = {
      id: Date.now() + '-' + photos.length,
      dataUrl,
      thumbUrl,
      width: w,
      height: h,
      orientation: { ...orientation },
      timestamp: Date.now(),
    };

    photos.push(photo);
    return photo;
  }

  function getPhotos() {
    return photos;
  }

  function removePhoto(id) {
    photos = photos.filter(p => p.id !== id);
  }

  function clearPhotos() {
    photos = [];
  }

  function getOrientation() {
    return { ...orientation };
  }

  function startOrientation() {
    if (orientationEnabled) return;

    if (typeof DeviceOrientationEvent !== 'undefined' &&
        typeof DeviceOrientationEvent.requestPermission === 'function') {
      DeviceOrientationEvent.requestPermission()
        .then(state => {
          if (state === 'granted') {
            window.addEventListener('deviceorientation', onOrientation, true);
            orientationEnabled = true;
          }
        })
        .catch(() => {});
    } else {
      window.addEventListener('deviceorientation', onOrientation, true);
      orientationEnabled = true;
    }
  }

  function stopOrientation() {
    window.removeEventListener('deviceorientation', onOrientation, true);
    orientationEnabled = false;
  }

  function onOrientation(e) {
    orientation.alpha = e.alpha || 0;
    orientation.beta = e.beta || 0;
    orientation.gamma = e.gamma || 0;
  }

  function isActive() {
    return stream !== null;
  }

  return {
    init,
    start,
    stop,
    flip,
    capture,
    getPhotos,
    removePhoto,
    clearPhotos,
    getOrientation,
    isActive,
    MAX_PHOTOS,
  };
})();
