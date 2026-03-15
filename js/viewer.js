/**
 * Viewer module — Three.js 3D visualization of point clouds and meshes.
 */
const ViewerModule = (() => {
  let scene, camera, renderer, controls;
  let pointCloudObj = null;
  let meshObj = null;
  let canvasEl = null;
  let animFrame = null;
  let showMesh = false;
  let initialized = false;
  let boundingSize = 1;
  let autoRotate = true;

  function init(canvas) {
    canvasEl = canvas;

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x0a0e1a);

    camera = new THREE.PerspectiveCamera(60, 1, 0.01, 100);
    camera.position.set(0, 0.5, 2);

    renderer = new THREE.WebGLRenderer({
      canvas: canvasEl,
      antialias: true,
      alpha: false,
    });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    controls = new THREE.OrbitControls(camera, canvasEl);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.rotateSpeed = 0.8;
    controls.zoomSpeed = 0.8;
    controls.panSpeed = 0.5;
    controls.minDistance = 0.2;
    controls.maxDistance = 10;
    controls.target.set(0, 0, 0);
    controls.autoRotate = true;
    controls.autoRotateSpeed = 1.5;

    controls.addEventListener('start', () => {
      autoRotate = false;
      controls.autoRotate = false;
    });

    const ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    scene.add(ambientLight);

    const dirLight = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight.position.set(2, 3, 2);
    scene.add(dirLight);

    scene.fog = new THREE.FogExp2(0x0a0e1a, 0.15);

    const gridHelper = new THREE.GridHelper(4, 20, 0x1a2035, 0x111827);
    scene.add(gridHelper);

    const axesHelper = new THREE.AxesHelper(0.3);
    axesHelper.position.set(-1.8, 0.01, -1.8);
    scene.add(axesHelper);

    resize();
    initialized = true;
  }

  function resize() {
    if (!canvasEl || !renderer) return;
    const parent = canvasEl.parentElement;
    if (!parent) return;
    const w = parent.clientWidth;
    const h = parent.clientHeight || 400;
    renderer.setSize(w, h);
    camera.aspect = w / h;
    camera.updateProjectionMatrix();
  }

  function startLoop() {
    if (animFrame) return;
    function loop() {
      animFrame = requestAnimationFrame(loop);
      controls.update();
      renderer.render(scene, camera);
    }
    loop();
  }

  function stopLoop() {
    if (animFrame) {
      cancelAnimationFrame(animFrame);
      animFrame = null;
    }
  }

  function loadPointCloud(positions, colors) {
    clearScene();

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.computeBoundingSphere();

    const numPoints = positions.length / 3;
    const adaptiveSize = numPoints > 5000 ? 0.008 : numPoints > 1000 ? 0.015 : 0.025;

    const material = new THREE.PointsMaterial({
      size: adaptiveSize,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: true,
      opacity: 0.9,
    });

    pointCloudObj = new THREE.Points(geometry, material);
    scene.add(pointCloudObj);

    if (geometry.boundingSphere) {
      boundingSize = geometry.boundingSphere.radius || 1;
      const center = geometry.boundingSphere.center;
      controls.target.copy(center);
      camera.position.set(
        center.x,
        center.y + boundingSize * 0.5,
        center.z + boundingSize * 2.5
      );
    }
  }

  function loadMesh(positions, colors, indices) {
    if (meshObj) {
      scene.remove(meshObj);
      meshObj.geometry.dispose();
      meshObj.material.dispose();
      meshObj = null;
    }

    if (!indices || indices.length === 0) return;

    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();

    const material = new THREE.MeshStandardMaterial({
      vertexColors: true,
      side: THREE.DoubleSide,
      metalness: 0.1,
      roughness: 0.7,
      transparent: true,
      opacity: 0.85,
      flatShading: true,
    });

    meshObj = new THREE.Mesh(geometry, material);
    meshObj.visible = showMesh;
    scene.add(meshObj);
  }

  function toggleMode() {
    showMesh = !showMesh;
    if (pointCloudObj) pointCloudObj.visible = !showMesh;
    if (meshObj) meshObj.visible = showMesh;
    return showMesh ? 'mesh' : 'points';
  }

  function resetView() {
    controls.target.set(0, 0, 0);
    camera.position.set(0, boundingSize * 0.5, boundingSize * 2.5);
    autoRotate = true;
    controls.autoRotate = true;
    controls.update();
  }

  function adjustPointSize(factor) {
    if (pointCloudObj && pointCloudObj.material) {
      pointCloudObj.material.size = Math.max(0.002, Math.min(0.1, pointCloudObj.material.size * factor));
    }
  }

  function clearScene() {
    if (pointCloudObj) {
      scene.remove(pointCloudObj);
      pointCloudObj.geometry.dispose();
      pointCloudObj.material.dispose();
      pointCloudObj = null;
    }
    if (meshObj) {
      scene.remove(meshObj);
      meshObj.geometry.dispose();
      meshObj.material.dispose();
      meshObj = null;
    }
    showMesh = false;
  }

  function getInfo() {
    const pts = pointCloudObj ? pointCloudObj.geometry.attributes.position.count : 0;
    const tri = meshObj ? meshObj.geometry.index ? meshObj.geometry.index.count / 3 : 0 : 0;
    return { points: pts, triangles: tri, mode: showMesh ? 'Siatka' : 'Chmura punktów' };
  }

  function isInitialized() {
    return initialized;
  }

  return {
    init,
    resize,
    startLoop,
    stopLoop,
    loadPointCloud,
    loadMesh,
    toggleMode,
    resetView,
    clearScene,
    adjustPointSize,
    getInfo,
    isInitialized,
  };
})();
