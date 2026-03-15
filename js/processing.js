/**
 * Processing module — 3D reconstruction pipeline.
 * Feature detection, matching, triangulation, point cloud & mesh generation.
 */
const ProcessingModule = (() => {
  let cancelled = false;
  let resultData = null;

  const WORK_SIZE = 320;
  const FEATURE_THRESHOLD = 30;
  const PATCH_RADIUS = 5;
  const MAX_FEATURES = 600;
  const MATCH_RATIO = 0.75;

  function cancel() { cancelled = true; }
  function getResult() { return resultData; }

  async function run(photos, onProgress) {
    cancelled = false;
    resultData = null;

    const steps = [
      'Przygotowanie obrazów',
      'Detekcja cech',
      'Dopasowywanie cech',
      'Estymacja pozycji kamer',
      'Triangulacja punktów 3D',
      'Generowanie chmury punktów',
      'Budowanie siatki',
      'Finalizacja',
    ];

    const report = (stepIdx, pct) => {
      if (onProgress) {
        const base = (stepIdx / steps.length) * 100;
        const add = (pct / 100) * (100 / steps.length);
        onProgress({
          stepIdx,
          stepName: steps[stepIdx],
          steps,
          percent: Math.min(Math.round(base + add), 100),
        });
      }
    };

    const startTime = performance.now();

    report(0, 0);
    const images = await prepareImages(photos, p => report(0, p));
    if (cancelled) return null;

    report(1, 0);
    const features = await detectFeatures(images, p => report(1, p));
    if (cancelled) return null;

    report(2, 0);
    const matches = await matchFeatures(features, p => report(2, p));
    if (cancelled) return null;

    report(3, 0);
    const cameras = estimateCameras(photos, images.length, p => report(3, p));
    if (cancelled) return null;

    report(4, 0);
    const points3D = triangulatePoints(matches, cameras, images, p => report(4, p));
    if (cancelled) return null;

    report(5, 0);
    const pointCloud = buildPointCloud(points3D, images, p => report(5, p));
    if (cancelled) return null;

    report(6, 0);
    const mesh = buildMesh(pointCloud, p => report(6, p));
    if (cancelled) return null;

    report(7, 50);
    await yieldFrame();

    const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);

    resultData = {
      pointCloud,
      mesh,
      stats: {
        points: pointCloud.positions.length / 3,
        triangles: mesh.indices.length / 3,
        time: elapsed,
      },
    };

    report(7, 100);
    return resultData;
  }

  /* ===== Image Preparation ===== */
  async function prepareImages(photos, prog) {
    const images = [];
    for (let i = 0; i < photos.length; i++) {
      const img = await loadImage(photos[i].dataUrl);
      const scale = WORK_SIZE / Math.max(img.width, img.height);
      const w = Math.round(img.width * scale);
      const h = Math.round(img.height * scale);

      const canvas = document.createElement('canvas');
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, w, h);

      const imageData = ctx.getImageData(0, 0, w, h);
      const gray = toGrayscale(imageData);

      images.push({ width: w, height: h, imageData, gray, canvas, ctx });
      prog(((i + 1) / photos.length) * 100);
      await yieldFrame();
    }
    return images;
  }

  function loadImage(src) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = reject;
      img.src = src;
    });
  }

  function toGrayscale(imageData) {
    const { data, width, height } = imageData;
    const gray = new Float32Array(width * height);
    for (let i = 0; i < gray.length; i++) {
      const j = i * 4;
      gray[i] = 0.299 * data[j] + 0.587 * data[j + 1] + 0.114 * data[j + 2];
    }
    return gray;
  }

  /* ===== Feature Detection (Harris-like corners) ===== */
  async function detectFeatures(images, prog) {
    const allFeatures = [];
    for (let i = 0; i < images.length; i++) {
      const { gray, width, height } = images[i];
      const corners = harrisCorners(gray, width, height);
      allFeatures.push(corners);
      prog(((i + 1) / images.length) * 100);
      await yieldFrame();
    }
    return allFeatures;
  }

  function harrisCorners(gray, w, h) {
    const Ix = new Float32Array(w * h);
    const Iy = new Float32Array(w * h);

    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const idx = y * w + x;
        Ix[idx] = (gray[idx + 1] - gray[idx - 1]) * 0.5;
        Iy[idx] = (gray[idx + w] - gray[idx - w]) * 0.5;
      }
    }

    const response = new Float32Array(w * h);
    const ks = 0.04;
    const r = 3;

    for (let y = r; y < h - r; y++) {
      for (let x = r; x < w - r; x++) {
        let sxx = 0, syy = 0, sxy = 0;
        for (let dy = -r; dy <= r; dy++) {
          for (let dx = -r; dx <= r; dx++) {
            const idx = (y + dy) * w + (x + dx);
            const ix = Ix[idx], iy = Iy[idx];
            sxx += ix * ix;
            syy += iy * iy;
            sxy += ix * iy;
          }
        }
        const det = sxx * syy - sxy * sxy;
        const trace = sxx + syy;
        response[y * w + x] = det - ks * trace * trace;
      }
    }

    const candidates = [];
    for (let y = r + 1; y < h - r - 1; y++) {
      for (let x = r + 1; x < w - r - 1; x++) {
        const val = response[y * w + x];
        if (val < FEATURE_THRESHOLD) continue;

        let isMax = true;
        for (let dy = -1; dy <= 1 && isMax; dy++) {
          for (let dx = -1; dx <= 1 && isMax; dx++) {
            if (dx === 0 && dy === 0) continue;
            if (response[(y + dy) * w + (x + dx)] >= val) isMax = false;
          }
        }
        if (isMax) {
          candidates.push({ x, y, score: val });
        }
      }
    }

    candidates.sort((a, b) => b.score - a.score);
    const selected = candidates.slice(0, MAX_FEATURES);

    return selected.map(c => ({
      x: c.x,
      y: c.y,
      score: c.score,
      descriptor: computeDescriptor(gray, w, h, c.x, c.y),
    }));
  }

  function computeDescriptor(gray, w, h, cx, cy) {
    const r = PATCH_RADIUS;
    const desc = new Float32Array((2 * r + 1) * (2 * r + 1));
    let mean = 0, count = 0;

    for (let dy = -r; dy <= r; dy++) {
      for (let dx = -r; dx <= r; dx++) {
        const px = cx + dx;
        const py = cy + dy;
        if (px >= 0 && px < w && py >= 0 && py < h) {
          const val = gray[py * w + px];
          desc[count] = val;
          mean += val;
        }
        count++;
      }
    }
    mean /= count;

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

  /* ===== Feature Matching (NCC) ===== */
  async function matchFeatures(allFeatures, prog) {
    const pairMatches = [];
    const nPairs = allFeatures.length - 1;

    for (let i = 0; i < nPairs; i++) {
      const fa = allFeatures[i];
      const fb = allFeatures[i + 1];
      const matches = matchPair(fa, fb);
      pairMatches.push({ i, j: i + 1, matches });
      prog(((i + 1) / nPairs) * 100);
      await yieldFrame();
    }

    if (allFeatures.length > 2) {
      const fa = allFeatures[allFeatures.length - 1];
      const fb = allFeatures[0];
      const matches = matchPair(fa, fb);
      pairMatches.push({ i: allFeatures.length - 1, j: 0, matches });
    }

    return pairMatches;
  }

  function matchPair(fa, fb) {
    const matches = [];
    for (let ai = 0; ai < fa.length; ai++) {
      let bestDist = Infinity;
      let secondDist = Infinity;
      let bestIdx = -1;

      for (let bi = 0; bi < fb.length; bi++) {
        const dist = descriptorDistance(fa[ai].descriptor, fb[bi].descriptor);
        if (dist < bestDist) {
          secondDist = bestDist;
          bestDist = dist;
          bestIdx = bi;
        } else if (dist < secondDist) {
          secondDist = dist;
        }
      }

      if (bestIdx >= 0 && bestDist < MATCH_RATIO * secondDist) {
        matches.push({
          a: { x: fa[ai].x, y: fa[ai].y, idx: ai },
          b: { x: fb[bestIdx].x, y: fb[bestIdx].y, idx: bestIdx },
          distance: bestDist,
        });
      }
    }
    return matches;
  }

  function descriptorDistance(da, db) {
    let sum = 0;
    for (let i = 0; i < da.length; i++) {
      const d = da[i] - db[i];
      sum += d * d;
    }
    return sum;
  }

  /* ===== Camera Pose Estimation ===== */
  function estimateCameras(photos, n, prog) {
    const cameras = [];
    const radius = 2.0;

    for (let i = 0; i < n; i++) {
      const ori = photos[i].orientation;
      let angle;

      if (ori && (ori.alpha !== 0 || ori.beta !== 0)) {
        angle = (ori.alpha * Math.PI) / 180;
      } else {
        angle = (i / n) * Math.PI * 2;
      }

      const x = radius * Math.cos(angle);
      const z = radius * Math.sin(angle);
      const y = 0;

      cameras.push({
        position: [x, y, z],
        lookAt: [0, 0, 0],
        angle,
        index: i,
      });

      prog(((i + 1) / n) * 100);
    }
    return cameras;
  }

  /* ===== Triangulation ===== */
  function triangulatePoints(pairMatches, cameras, images, prog) {
    const allPoints = [];
    const total = pairMatches.length;

    for (let pi = 0; pi < total; pi++) {
      const pair = pairMatches[pi];
      const camA = cameras[pair.i];
      const camB = cameras[pair.j];
      const imgA = images[pair.i];
      const imgB = images[pair.j];

      for (const m of pair.matches) {
        const nxA = (m.a.x - imgA.width / 2) / imgA.width;
        const nyA = -(m.a.y - imgA.height / 2) / imgA.height;
        const nxB = (m.b.x - imgB.width / 2) / imgB.width;
        const nyB = -(m.b.y - imgB.height / 2) / imgB.height;

        const dirA = normalize3([
          nxA - camA.position[0],
          nyA - camA.position[1],
          -1 - camA.position[2],
        ]);
        const dirB = normalize3([
          nxB - camB.position[0],
          nyB - camB.position[1],
          -1 - camB.position[2],
        ]);

        const point = midpointTriangulation(
          camA.position, dirA,
          camB.position, dirB
        );

        if (point) {
          const px = Math.round(m.a.x);
          const py = Math.round(m.a.y);
          const pIdx = (py * imgA.width + px) * 4;
          const r = imgA.imageData.data[pIdx] / 255;
          const g = imgA.imageData.data[pIdx + 1] / 255;
          const b = imgA.imageData.data[pIdx + 2] / 255;

          allPoints.push({
            position: point,
            color: [r, g, b],
            confidence: 1 - m.distance,
          });
        }
      }

      prog(((pi + 1) / total) * 100);
    }

    return allPoints;
  }

  function midpointTriangulation(p1, d1, p2, d2) {
    const w0 = [p1[0] - p2[0], p1[1] - p2[1], p1[2] - p2[2]];
    const a = dot3(d1, d1);
    const b = dot3(d1, d2);
    const c = dot3(d2, d2);
    const d = dot3(d1, w0);
    const e = dot3(d2, w0);

    const denom = a * c - b * b;
    if (Math.abs(denom) < 1e-8) return null;

    const s = (b * e - c * d) / denom;
    const t = (a * e - b * d) / denom;

    const closest1 = [
      p1[0] + s * d1[0],
      p1[1] + s * d1[1],
      p1[2] + s * d1[2],
    ];
    const closest2 = [
      p2[0] + t * d2[0],
      p2[1] + t * d2[1],
      p2[2] + t * d2[2],
    ];

    const dist = Math.sqrt(
      (closest1[0] - closest2[0]) ** 2 +
      (closest1[1] - closest2[1]) ** 2 +
      (closest1[2] - closest2[2]) ** 2
    );
    if (dist > 1.0) return null;

    return [
      (closest1[0] + closest2[0]) / 2,
      (closest1[1] + closest2[1]) / 2,
      (closest1[2] + closest2[2]) / 2,
    ];
  }

  /* ===== Point Cloud Assembly ===== */
  function buildPointCloud(points3D, images, prog) {
    const filtered = points3D.filter(p => {
      const d = Math.sqrt(p.position[0] ** 2 + p.position[1] ** 2 + p.position[2] ** 2);
      return d < 5 && d > 0.01;
    });

    prog(30);

    if (filtered.length < 100) {
      return generateDenseCloud(images, prog);
    }

    const center = [0, 0, 0];
    for (const p of filtered) {
      center[0] += p.position[0];
      center[1] += p.position[1];
      center[2] += p.position[2];
    }
    center[0] /= filtered.length;
    center[1] /= filtered.length;
    center[2] /= filtered.length;

    const positions = new Float32Array(filtered.length * 3);
    const colors = new Float32Array(filtered.length * 3);

    for (let i = 0; i < filtered.length; i++) {
      positions[i * 3] = filtered[i].position[0] - center[0];
      positions[i * 3 + 1] = filtered[i].position[1] - center[1];
      positions[i * 3 + 2] = filtered[i].position[2] - center[2];
      colors[i * 3] = filtered[i].color[0];
      colors[i * 3 + 1] = filtered[i].color[1];
      colors[i * 3 + 2] = filtered[i].color[2];
    }

    prog(100);
    return { positions, colors };
  }

  function generateDenseCloud(images, prog) {
    const points = [];
    const samplesPerImage = Math.ceil(3000 / images.length);

    for (let imgIdx = 0; imgIdx < images.length; imgIdx++) {
      const img = images[imgIdx];
      const { gray, width, height, imageData } = img;

      const angle = (imgIdx / images.length) * Math.PI * 2;
      const camRadius = 2.0;

      const edgeMap = sobelEdges(gray, width, height);

      let sampled = 0;
      const attempts = samplesPerImage * 8;
      for (let a = 0; a < attempts && sampled < samplesPerImage; a++) {
        const px = Math.floor(Math.random() * (width - 4)) + 2;
        const py = Math.floor(Math.random() * (height - 4)) + 2;
        const edgeVal = edgeMap[py * width + px];

        if (edgeVal < 15 && Math.random() > 0.15) continue;

        const nx = (px - width / 2) / width;
        const ny = -(py - height / 2) / height;
        const depth = 0.3 + (edgeVal / 255) * 0.7 + (Math.random() - 0.5) * 0.05;

        const x = nx * depth;
        const y = ny * depth;
        const z = -depth;

        const cosA = Math.cos(angle);
        const sinA = Math.sin(angle);
        const wx = x * cosA - z * sinA;
        const wz = x * sinA + z * cosA;

        const pIdx = (py * width + px) * 4;
        const r = imageData.data[pIdx] / 255;
        const g = imageData.data[pIdx + 1] / 255;
        const b = imageData.data[pIdx + 2] / 255;

        points.push({ position: [wx, y, wz], color: [r, g, b] });
        sampled++;
      }

      prog(30 + ((imgIdx + 1) / images.length) * 70);
    }

    const positions = new Float32Array(points.length * 3);
    const colors = new Float32Array(points.length * 3);
    for (let i = 0; i < points.length; i++) {
      positions[i * 3] = points[i].position[0];
      positions[i * 3 + 1] = points[i].position[1];
      positions[i * 3 + 2] = points[i].position[2];
      colors[i * 3] = points[i].color[0];
      colors[i * 3 + 1] = points[i].color[1];
      colors[i * 3 + 2] = points[i].color[2];
    }

    return { positions, colors };
  }

  function sobelEdges(gray, w, h) {
    const edges = new Float32Array(w * h);
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const tl = gray[(y - 1) * w + (x - 1)];
        const tc = gray[(y - 1) * w + x];
        const tr = gray[(y - 1) * w + (x + 1)];
        const ml = gray[y * w + (x - 1)];
        const mr = gray[y * w + (x + 1)];
        const bl = gray[(y + 1) * w + (x - 1)];
        const bc = gray[(y + 1) * w + x];
        const br = gray[(y + 1) * w + (x + 1)];

        const gx = -tl - 2 * ml - bl + tr + 2 * mr + br;
        const gy = -tl - 2 * tc - tr + bl + 2 * bc + br;
        edges[y * w + x] = Math.min(255, Math.sqrt(gx * gx + gy * gy));
      }
    }
    return edges;
  }

  /* ===== Mesh Generation (Delaunay-like) ===== */
  function buildMesh(pointCloud, prog) {
    const n = pointCloud.positions.length / 3;
    if (n < 4) {
      prog(100);
      return { indices: new Uint32Array(0) };
    }

    prog(10);

    const bucketSize = 0.15;
    const grid = new Map();
    const positions = pointCloud.positions;

    for (let i = 0; i < n; i++) {
      const gx = Math.floor(positions[i * 3] / bucketSize);
      const gy = Math.floor(positions[i * 3 + 1] / bucketSize);
      const gz = Math.floor(positions[i * 3 + 2] / bucketSize);
      const key = `${gx},${gy},${gz}`;
      if (!grid.has(key)) grid.set(key, []);
      grid.get(key).push(i);
    }

    prog(30);

    const indices = [];
    const maxDist = bucketSize * 2.5;
    const maxDistSq = maxDist * maxDist;
    const processed = new Set();

    for (const [key, bucket] of grid) {
      const [gx, gy, gz] = key.split(',').map(Number);
      const neighbors = [];

      for (let dx = -1; dx <= 1; dx++) {
        for (let dy = -1; dy <= 1; dy++) {
          for (let dz = -1; dz <= 1; dz++) {
            const nk = `${gx + dx},${gy + dy},${gz + dz}`;
            if (grid.has(nk)) neighbors.push(...grid.get(nk));
          }
        }
      }

      for (const i of bucket) {
        const nearest = [];
        for (const j of neighbors) {
          if (j <= i) continue;
          const dx = positions[j * 3] - positions[i * 3];
          const dy = positions[j * 3 + 1] - positions[i * 3 + 1];
          const dz = positions[j * 3 + 2] - positions[i * 3 + 2];
          const distSq = dx * dx + dy * dy + dz * dz;
          if (distSq < maxDistSq) {
            nearest.push({ idx: j, dist: distSq });
          }
        }

        nearest.sort((a, b) => a.dist - b.dist);
        const top = nearest.slice(0, 6);

        for (let a = 0; a < top.length; a++) {
          for (let b = a + 1; b < top.length; b++) {
            const triKey = [i, top[a].idx, top[b].idx].sort().join(',');
            if (processed.has(triKey)) continue;
            processed.add(triKey);

            const j = top[a].idx;
            const k = top[b].idx;
            const dx = positions[j * 3] - positions[k * 3];
            const dy = positions[j * 3 + 1] - positions[k * 3 + 1];
            const dz = positions[j * 3 + 2] - positions[k * 3 + 2];
            if (dx * dx + dy * dy + dz * dz < maxDistSq) {
              indices.push(i, j, k);
            }
          }
        }
      }
    }

    prog(100);
    return { indices: new Uint32Array(indices) };
  }

  /* ===== Utilities ===== */
  function normalize3(v) {
    const len = Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) || 1;
    return [v[0] / len, v[1] / len, v[2] / len];
  }

  function dot3(a, b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  function yieldFrame() {
    return new Promise(resolve => setTimeout(resolve, 0));
  }

  /* ===== Export Functions ===== */
  function exportOBJ(pointCloud, mesh) {
    let obj = '# Scan3D Export\n';
    const n = pointCloud.positions.length / 3;
    const p = pointCloud.positions;
    const c = pointCloud.colors;

    for (let i = 0; i < n; i++) {
      obj += `v ${p[i*3].toFixed(6)} ${p[i*3+1].toFixed(6)} ${p[i*3+2].toFixed(6)} ${c[i*3].toFixed(3)} ${c[i*3+1].toFixed(3)} ${c[i*3+2].toFixed(3)}\n`;
    }

    if (mesh && mesh.indices.length > 0) {
      const idx = mesh.indices;
      for (let i = 0; i < idx.length; i += 3) {
        obj += `f ${idx[i]+1} ${idx[i+1]+1} ${idx[i+2]+1}\n`;
      }
    }

    return obj;
  }

  function exportPLY(pointCloud, mesh) {
    const n = pointCloud.positions.length / 3;
    const nFaces = mesh ? mesh.indices.length / 3 : 0;
    const p = pointCloud.positions;
    const c = pointCloud.colors;

    let ply = 'ply\nformat ascii 1.0\n';
    ply += `element vertex ${n}\n`;
    ply += 'property float x\nproperty float y\nproperty float z\n';
    ply += 'property uchar red\nproperty uchar green\nproperty uchar blue\n';

    if (nFaces > 0) {
      ply += `element face ${nFaces}\n`;
      ply += 'property list uchar int vertex_indices\n';
    }

    ply += 'end_header\n';

    for (let i = 0; i < n; i++) {
      const r = Math.round(c[i*3] * 255);
      const g = Math.round(c[i*3+1] * 255);
      const b = Math.round(c[i*3+2] * 255);
      ply += `${p[i*3].toFixed(6)} ${p[i*3+1].toFixed(6)} ${p[i*3+2].toFixed(6)} ${r} ${g} ${b}\n`;
    }

    if (nFaces > 0) {
      const idx = mesh.indices;
      for (let i = 0; i < idx.length; i += 3) {
        ply += `3 ${idx[i]} ${idx[i+1]} ${idx[i+2]}\n`;
      }
    }

    return ply;
  }

  return {
    run,
    cancel,
    getResult,
    exportOBJ,
    exportPLY,
  };
})();
