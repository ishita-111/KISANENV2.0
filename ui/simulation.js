// simulation.js
// KisanEnv 2.0 — Three.js 3D Farm Scene (Rebuilt)
// Crops update visually ONLY at Day ~45 and ~90. Weather/atmosphere is continuous.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Renderer ────────────────────────────────────────────────────────────────
const canvas = document.getElementById('farm-canvas');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false });
renderer.setSize(window.innerWidth * 0.65, window.innerHeight);
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;
renderer.outputColorSpace = THREE.SRGBColorSpace;

// ── Scene + Camera (zoomed out for larger frame) ─────────────────────────
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, (window.innerWidth * 0.65) / window.innerHeight, 0.1, 800);
camera.position.set(0, 18, 48);
camera.lookAt(0, 0, 0);

// ── Controls ────────────────────────────────────────────────────────────────
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.06;
controls.minDistance = 6;
controls.maxDistance = 50;
controls.maxPolarAngle = Math.PI / 2.1;
controls.target.set(0, 0, 0);

// ── Lighting ────────────────────────────────────────────────────────────────
const skyHemi = new THREE.HemisphereLight(0x87ceeb, 0xc8a030, 0.7);
scene.add(skyHemi);

const sunLight = new THREE.DirectionalLight(0xffeedd, 2.5);
sunLight.position.set(20, 40, 15);
sunLight.castShadow = true;
sunLight.shadow.mapSize.width = 1024;
sunLight.shadow.mapSize.height = 1024;
sunLight.shadow.camera.near = 0.1;
sunLight.shadow.camera.far = 200;
sunLight.shadow.camera.left = -30;
sunLight.shadow.camera.right = 30;
sunLight.shadow.camera.top = 30;
sunLight.shadow.camera.bottom = -30;
scene.add(sunLight);

const ambientFill = new THREE.AmbientLight(0x304060, 0.3);
scene.add(ambientFill);

// ── Skybox Gradient & Atmosphere ────────────────────────────────────────────
const skyGeo = new THREE.SphereGeometry(300, 16, 8);
const skyMat = new THREE.ShaderMaterial({
  side: THREE.BackSide,
  uniforms: {
    topColor: { value: new THREE.Color(0x5ca2e8) },
    bottomColor: { value: new THREE.Color(0xdcecf8) },
    offset: { value: 30 },
    exponent: { value: 0.6 },
  },
  vertexShader: `
    varying vec3 vWorldPosition;
    void main() {
      vec4 wp = modelMatrix * vec4(position, 1.0);
      vWorldPosition = wp.xyz;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `,
  fragmentShader: `
    uniform vec3 topColor;
    uniform vec3 bottomColor;
    uniform float offset;
    uniform float exponent;
    varying vec3 vWorldPosition;
    void main() {
      float h = normalize(vWorldPosition + offset).y;
      gl_FragColor = vec4(mix(bottomColor, topColor, max(pow(max(h,0.0), exponent), 0.0)), 1.0);
    }
  `,
});
scene.add(new THREE.Mesh(skyGeo, skyMat));

// ── Ground (Soil Shader & Ridges) ───────────────────────────────────────────
const groundGeo = new THREE.PlaneGeometry(160, 160, 120, 120);
const soilMat = new THREE.ShaderMaterial({
  uniforms: {
    soilMoisture: { value: 0.5 },
    soilHealth: { value: 0.7 },
    time: { value: 0 },
  },
  vertexShader: `
    uniform float time;
    varying vec2 vUv;
    void main() {
      vUv = uv;
      vec3 p = position;
      // Crop dirt ridges aligning with COLS (spacing is 3.5, frequency = 2.0*PI/3.5 = ~1.795)
      p.z += sin(p.x * 1.795) * 0.35 + sin(p.x * 0.5 + time * 0.05) * 0.05;
      gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
    }
  `,
  fragmentShader: `
    uniform float soilMoisture;
    uniform float soilHealth;
    varying vec2 vUv;
    void main() {
      vec3 dry = vec3(0.35, 0.22, 0.12);
      vec3 wet = vec3(0.18, 0.11, 0.05);
      vec3 dead = vec3(0.28, 0.23, 0.18);
      vec3 c = mix(dry, wet, soilMoisture);
      c = mix(dead, c, soilHealth);
      c += (fract(sin(dot(vUv*50.0, vec2(127.1,311.7)))*43758.5) - 0.5) * 0.04;
      gl_FragColor = vec4(c, 1.0);
    }
  `,
});
const ground = new THREE.Mesh(groundGeo, soilMat);
ground.rotation.x = -Math.PI / 2;
ground.receiveShadow = true;
scene.add(ground);

// ── Crop Field (built once, updated only at day 45 and 90) ──────────────────
const ROWS = 12;
const COLS = 16;
const cropMeshes = [];

const stemGeo = new THREE.CylinderGeometry(0.03, 0.05, 1, 5);
const leafGeo = new THREE.PlaneGeometry(0.4, 0.22);
const bollGeo = new THREE.SphereGeometry(0.1, 5, 5);

let currentCropPhase = 0; // 0 = seedling, 1 = mid-growth, 2 = harvest

function buildCropField(phase) {
  // Remove old crops
  cropMeshes.forEach(g => scene.remove(g));
  cropMeshes.length = 0;

  const stemScale = phase === 0 ? 0.7 : phase === 1 ? 1.4 : 1.9;
  const leafCount = phase === 0 ? 2 : phase === 1 ? 4 : 5;
  const healthHue = phase === 2 ? 0.22 : 0.28;
  const showBolls = phase === 2;

  for (let row = 0; row < ROWS; row++) {
    for (let col = 0; col < COLS; col++) {
      const group = new THREE.Group();

      // Stem
      const sMat = new THREE.MeshLambertMaterial({
        color: new THREE.Color().setHSL(0.25, 0.65, 0.22 + phase * 0.08)
      });
      const stem = new THREE.Mesh(stemGeo, sMat);
      stem.scale.y = stemScale;
      stem.position.y = stemScale * 0.5;
      stem.castShadow = true;
      group.add(stem);

      // Leaves
      for (let l = 0; l < leafCount; l++) {
        const lMat = new THREE.MeshLambertMaterial({
          color: new THREE.Color().setHSL(healthHue, 0.7, 0.25 + phase * 0.05),
          side: THREE.DoubleSide,
          transparent: true,
          opacity: 0.85,
        });
        const leaf = new THREE.Mesh(leafGeo, lMat);
        const angle = (l / leafCount) * Math.PI * 2 + (Math.random() - 0.5) * 0.3;
        leaf.position.set(
          Math.cos(angle) * (0.15 + phase * 0.06),
          stemScale * 0.45 + l * 0.1,
          Math.sin(angle) * (0.15 + phase * 0.06)
        );
        leaf.rotation.set(0.2, angle, 0.12);
        leaf.castShadow = true;
        group.add(leaf);
      }

      // Cotton bolls at harvest
      if (showBolls) {
        for (let b = 0; b < 3; b++) {
          const bMat = new THREE.MeshLambertMaterial({ color: 0xf5f0e0 });
          const boll = new THREE.Mesh(bollGeo, bMat);
          boll.position.set(
            (Math.random() - 0.5) * 0.25,
            stemScale * 0.7 + b * 0.08,
            (Math.random() - 0.5) * 0.25
          );
          group.add(boll);
        }
      }

      group.position.set(
        (col - COLS / 2) * 3.5 + 1.75 + (Math.random() - 0.5) * 0.3,
        0,
        (row - ROWS / 2) * 3.2 + 1.6 + (Math.random() - 0.5) * 0.3
      );
      scene.add(group);
      cropMeshes.push(group);
    }
  }
  currentCropPhase = phase;
}

// Initial seedling field
buildCropField(0);

// ── Fences ──────────────────────────────────────────────────────────────────
const fenceGroup = new THREE.Group();
const postGeo = new THREE.BoxGeometry(0.4, 3.5, 0.4);
const woodMat = new THREE.MeshLambertMaterial({ color: 0x4a3b2c });
const beamGeoX = new THREE.BoxGeometry(90, 0.25, 0.15);
const beamGeoZ = new THREE.BoxGeometry(0.15, 0.25, 90);

// Left fence (-45)
for(let z=-45; z<=45; z+=9) {
  const p = new THREE.Mesh(postGeo, woodMat); p.position.set(-45, 1.75, z); p.castShadow = true; fenceGroup.add(p);
}
const beamL1 = new THREE.Mesh(beamGeoZ, woodMat); beamL1.position.set(-45, 2.6, 0); beamL1.castShadow = true; fenceGroup.add(beamL1);
const beamL2 = new THREE.Mesh(beamGeoZ, woodMat); beamL2.position.set(-45, 1.3, 0); beamL2.castShadow = true; fenceGroup.add(beamL2);

// Right fence (45)
for(let z=-45; z<=45; z+=9) {
  const p = new THREE.Mesh(postGeo, woodMat); p.position.set(45, 1.75, z); p.castShadow = true; fenceGroup.add(p);
}
const beamR1 = new THREE.Mesh(beamGeoZ, woodMat); beamR1.position.set(45, 2.6, 0); beamR1.castShadow = true; fenceGroup.add(beamR1);
const beamR2 = new THREE.Mesh(beamGeoZ, woodMat); beamR2.position.set(45, 1.3, 0); beamR2.castShadow = true; fenceGroup.add(beamR2);

// Back fence (-45)
for(let x=-45; x<=45; x+=9) {
  const p = new THREE.Mesh(postGeo, woodMat); p.position.set(x, 1.75, -45); p.castShadow = true; fenceGroup.add(p);
}
const beamB1 = new THREE.Mesh(beamGeoX, woodMat); beamB1.position.set(0, 2.6, -45); beamB1.castShadow = true; fenceGroup.add(beamB1);
const beamB2 = new THREE.Mesh(beamGeoX, woodMat); beamB2.position.set(0, 1.3, -45); beamB2.castShadow = true; fenceGroup.add(beamB2);
scene.add(fenceGroup);

// ── Neighbor Farms ──────────────────────────────────────────────────────────
[[-95, 0, 0], [95, 0, 0], [0, 0, -95]].forEach(([x, y, z]) => {
  const fm = new THREE.Mesh(
    new THREE.PlaneGeometry(100, 100),
    new THREE.MeshLambertMaterial({ color: 0x244f24 }) // Darker lush green
  );
  fm.rotation.x = -Math.PI / 2;
  fm.position.set(x, 0.05, z);
  fm.receiveShadow = true;
  scene.add(fm);
});

// ── Weather Effects ─────────────────────────────────────────────────────────
let rainParticles = null;
let hazeParticles = null;

function clearWeatherEffects() {
  if (rainParticles) { scene.remove(rainParticles); rainParticles = null; }
  if (hazeParticles) { scene.remove(hazeParticles); hazeParticles = null; }
}

function createRain(intensity) {
  clearWeatherEffects();
  const count = Math.floor(800 * intensity);
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(count * 3);
  const vel = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    pos[i*3] = (Math.random()-0.5) * 60;
    pos[i*3+1] = Math.random() * 25 + 5;
    pos[i*3+2] = (Math.random()-0.5) * 60;
    vel[i*3] = (Math.random()-0.5) * 0.4;
    vel[i*3+1] = -(2.5 + Math.random()*2) * intensity;
    vel[i*3+2] = (Math.random()-0.5) * 0.4;
  }
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  geo.setAttribute('velocity', new THREE.BufferAttribute(vel, 3));
  const mat = new THREE.PointsMaterial({ color: 0xaaccff, size: 0.04, transparent: true, opacity: 0.55 });
  rainParticles = new THREE.Points(geo, mat);
  scene.add(rainParticles);
}

function createDroughtHaze() {
  clearWeatherEffects();
  const count = 350;
  const geo = new THREE.BufferGeometry();
  const pos = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    pos[i*3] = (Math.random()-0.5) * 80;
    pos[i*3+1] = Math.random() * 7;
    pos[i*3+2] = (Math.random()-0.5) * 80;
  }
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  const mat = new THREE.PointsMaterial({ color: 0xc8a882, size: 0.22, transparent: true, opacity: 0.25 });
  hazeParticles = new THREE.Points(geo, mat);
  scene.add(hazeParticles);
}

// ── Day/Night Cycle ─────────────────────────────────────────────────────────
function setDayTime(dayFrac) {
  const hour = 8 + dayFrac * 10;
  const angle = (hour / 24) * Math.PI * 2;
  sunLight.position.set(Math.cos(angle)*100, Math.abs(Math.sin(angle))*80 + 5, 25);
  sunLight.intensity = 1.5 + Math.sin(angle) * 1.2;
  const warmth = 0.3 + (Math.sin(angle)*0.5+0.5) * 0.2;
  sunLight.color.setHSL(0.07, 0.9, warmth + 0.3);
}

// ── UPDATE FROM FARM STATE ──────────────────────────────────────────────────
export function updateFromFarmState(farmState, weatherCondition) {
  // Soil shader — updates every step
  soilMat.uniforms.soilMoisture.value = farmState.soil_moisture;
  soilMat.uniforms.soilHealth.value = farmState.soil_health;

  // Crops — ONLY update at phase boundaries
  const day = farmState.day || 1;
  let targetPhase = 0;
  if (day >= 80) targetPhase = 2;
  else if (day >= 40) targetPhase = 1;

  if (targetPhase !== currentCropPhase) {
    buildCropField(targetPhase);
  }

  // Weather — updates every step
  clearWeatherEffects();
  if (weatherCondition === 'rain' || weatherCondition === 'storm') {
    createRain(weatherCondition === 'storm' ? 2.0 : 1.0);
  } else if (weatherCondition === 'drought') {
    createDroughtHaze();
  }

  // Sky/sun
  setDayTime(day / 90);
}

// ── Animation Loop ──────────────────────────────────────────────────────────
const clock = new THREE.Clock();
function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();

  soilMat.uniforms.time.value = t;

  // Rain animation
  if (rainParticles) {
    const p = rainParticles.geometry.attributes.position.array;
    const v = rainParticles.geometry.attributes.velocity.array;
    for (let i = 0; i < p.length / 3; i++) {
      p[i*3+1] += v[i*3+1] * 0.016;
      p[i*3] += v[i*3] * 0.016;
      if (p[i*3+1] < 0) {
        p[i*3+1] = 25;
        p[i*3] = (Math.random()-0.5) * 60;
      }
    }
    rainParticles.geometry.attributes.position.needsUpdate = true;
  }

  // Gentle crop sway
  for (let i = 0; i < cropMeshes.length; i++) {
    cropMeshes[i].rotation.z = Math.sin(t * 0.7 + i * 0.3) * 0.025;
    cropMeshes[i].rotation.x = Math.cos(t * 0.5 + i * 0.2) * 0.015;
  }

  controls.update();
  renderer.render(scene, camera);
}
animate();

// ── Resize ──────────────────────────────────────────────────────────────────
window.addEventListener('resize', () => {
  const w = window.innerWidth * 0.65;
  const h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
});
