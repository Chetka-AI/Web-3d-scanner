/**
 * AI Engine — Monocular depth estimation via Transformers.js
 * Uses Depth Anything V2 (Small, int8 quantized ~27MB) for
 * state-of-the-art dense depth prediction from single images.
 *
 * Loaded lazily — Transformers.js is fetched on demand via dynamic import().
 */
const AIEngine = (() => {
  const DEPTH_MODEL = 'onnx-community/depth-anything-v2-small';
  const TRANSFORMERS_CDN = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3';

  let T = null;
  let depthPipeline = null;
  let ready = false;
  let loading = false;
  let supported = null;

  async function checkSupport() {
    if (supported !== null) return supported;
    try {
      T = await import(TRANSFORMERS_CDN);
      supported = true;
    } catch (e) {
      console.warn('Transformers.js not supported:', e);
      supported = false;
    }
    return supported;
  }

  async function initialize(onProgress) {
    if (ready) return true;
    if (loading) return false;
    loading = true;

    try {
      if (!T) {
        onProgress?.({ phase: 'lib', progress: 0, message: 'Ładowanie Transformers.js...' });
        T = await import(TRANSFORMERS_CDN);
      }

      T.env.allowLocalModels = false;

      onProgress?.({ phase: 'model', progress: 0, message: 'Pobieranie modelu Depth Anything V2...' });

      depthPipeline = await T.pipeline('depth-estimation', DEPTH_MODEL, {
        dtype: 'q8',
        device: 'wasm',
        progress_callback: (info) => {
          if (info.status === 'initiate') {
            onProgress?.({
              phase: 'model',
              progress: 0,
              message: `Pobieranie: ${shortenFilename(info.file)}`
            });
          }
          if (info.status === 'progress' && info.progress != null) {
            onProgress?.({
              phase: 'model',
              progress: Math.round(info.progress),
              message: `Pobieranie: ${shortenFilename(info.file)} (${Math.round(info.progress)}%)`
            });
          }
          if (info.status === 'done') {
            onProgress?.({
              phase: 'model',
              progress: 100,
              message: `Gotowy: ${shortenFilename(info.file)}`
            });
          }
        }
      });

      ready = true;
      onProgress?.({ phase: 'ready', progress: 100, message: 'Model AI gotowy!' });
      return true;
    } catch (err) {
      console.error('AI Engine initialization failed:', err);
      supported = false;
      throw err;
    } finally {
      loading = false;
    }
  }

  /**
   * Run depth estimation on an image source.
   * Accepts: URL string, data URL, Blob URL, HTMLImageElement
   * Returns: { predicted_depth: Tensor, depth: RawImage }
   */
  async function estimateDepth(imageSource) {
    if (!depthPipeline) throw new Error('AI not initialized');
    return await depthPipeline(imageSource);
  }

  /**
   * Convert a canvas element to a blob URL suitable for the pipeline.
   */
  function canvasToBlobUrl(canvas) {
    return new Promise((resolve) => {
      canvas.toBlob((blob) => {
        resolve(URL.createObjectURL(blob));
      }, 'image/jpeg', 0.9);
    });
  }

  /**
   * Extract normalized depth map from pipeline result.
   * Returns Float32Array where 0.0 = far, 1.0 = close.
   */
  function extractDepthMap(result) {
    const tensor = result.predicted_depth;
    const data = tensor.data;
    const dims = tensor.dims;
    const h = dims[dims.length - 2];
    const w = dims[dims.length - 1];

    let dMin = Infinity, dMax = -Infinity;
    for (let i = 0; i < data.length; i++) {
      if (data[i] < dMin) dMin = data[i];
      if (data[i] > dMax) dMax = data[i];
    }

    const range = dMax - dMin || 1;
    const normalized = new Float32Array(data.length);
    for (let i = 0; i < data.length; i++) {
      normalized[i] = (data[i] - dMin) / range;
    }

    return { data: normalized, width: w, height: h };
  }

  function shortenFilename(f) {
    if (!f) return 'model';
    const parts = f.split('/');
    return parts[parts.length - 1] || f;
  }

  function isReady() { return ready; }
  function isLoading() { return loading; }
  function isSupported() { return supported; }

  return {
    checkSupport,
    initialize,
    estimateDepth,
    canvasToBlobUrl,
    extractDepthMap,
    isReady,
    isLoading,
    isSupported,
  };
})();
