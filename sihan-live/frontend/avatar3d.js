/**
 * Three.js r168：SkinnedMesh + HDR + Bloom/SSAO + AnimationMixer
 * 换装、动作、表情；皮肤用 MeshPhysicalMaterial 的 transmission + attenuation 近似次表面（实时可达范围内尽量「像真的」）
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RGBELoader } from 'three/addons/loaders/RGBELoader.js';
import { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
import { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
import { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
import { SSAOPass } from 'three/addons/postprocessing/SSAOPass.js';
import { OutputPass } from 'three/addons/postprocessing/OutputPass.js';

// ---------------------------------------------------------------------------
// URL 解析
// ---------------------------------------------------------------------------

function resolveModelUrl() {
  try {
    const u = new URL(window.location.href);
    const q = u.searchParams.get('gltf') || u.searchParams.get('model');
    if (q && q.trim()) return decodeURIComponent(q.trim());
  } catch (e) { /* ignore */ }
  try {
    const ls = localStorage.getItem('sihan_gltf_url');
    if (ls && ls.trim()) return ls.trim();
  } catch (e) { /* ignore */ }
  return 'https://threejs.org/examples/models/gltf/Xbot.glb';
}

function resolveHdrUrl() {
  try {
    const u = new URL(window.location.href);
    const q = u.searchParams.get('hdr');
    if (q && q.trim()) return decodeURIComponent(q.trim());
  } catch (e) { /* ignore */ }
  try {
    const ls = localStorage.getItem('sihan_hdr_url');
    if (ls && ls.trim()) return ls.trim();
  } catch (e) { /* ignore */ }
  return 'https://threejs.org/examples/textures/equirectangular/royal_esplanade_1k.hdr';
}

const CONFIG = {
  get modelUrl() {
    return resolveModelUrl();
  },
  get hdrUrl() {
    return resolveHdrUrl();
  },
  clothingNameHints: {
    outer: ['coat', 'outer', 'jacket', 'clothes', 'dress', 'shirt', 'armor', 'cape'],
    underwear: ['underwear', 'under', 'lingerie', 'inner', 'bra', 'pants_inner'],
    body: ['body', 'skin', 'naked', 'base', 'torso', 'leg', 'arm', 'hand', 'foot', 'head', 'face'],
  },
  /** 动画 clip.name 模糊匹配 */
  animAlias: {
    idle: ['idle', 'stand', 'breathing', 'rest'],
    talking: ['talk', 'speak', 'lip', 'conversation', 'gesture', 'chat'],
    pose1: ['pose1', 'pose_1', 'salute', 'idle_pose', 'stance1'],
    pose2: ['pose2', 'pose_2', 'wave', 'stance2'],
    pose3: ['pose3', 'pose_3', 'dance', 'stance3'],
    pose4: ['pose4', 'pose_4', 'stretch', 'stance4'],
    pose5: ['pose5', 'pose_5', 'hips', 'stance5', 'sexy'],
    /** 循环层：优先用 clip，无则走程序化 */
    stroke: ['stroke', 'caress', 'touch_loop', 'pet'],
    breathing: ['breathe', 'breathing_loop', 'inhale'],
    'self-touch': ['self', 'touch_self', 'adjust', 'fidget'],
  },
  bloomStrength: 0.28,
  bloomRadius: 0.36,
  bloomThreshold: 0.85,
  ssaoKernelRadius: 7,
  ssaoMinDistance: 0.001,
  ssaoMaxDistance: 0.1,
  /** 皮肤：Physical 透射 + 体积衰减（非离线 PathTracing / Jensen BSSRDF，但在浏览器里是最接近物理的一层） */
  skinTransmission: 0.26,
  skinThickness: 0.52,
  skinAttenuationColor: 0xff5c48,
  skinAttenuationDistance: 0.78,
  skinIor: 1.4,
};

/**
 * 换装 4 级（与用户约定一致）
 * 3 = 全套（外衣+内衣+身体）
 * 2 = 仅外衣？ → 按渐进脱衣：去掉外衣，剩内衣+身体
 * 1 = 内衣+身体 → 再去内衣，仅身体
 * 0 = 完全裸体（仅身体网格；尽量隐藏非皮肤 other）
 */
const CLOTHING_LEVEL = {
  FULL: 3,
  OUTER_OFF: 2,
  UNDERWEAR_OFF: 1,
  NUDE: 0,
};

const state = {
  container: null,
  scene: null,
  camera: null,
  renderer: null,
  controls: null,
  clock: new THREE.Clock(),
  /** 主角色根节点 */
  characterRoot: null,
  mixer: null,
  skinnedMeshes: [],
  /** @type {THREE.Bone[]} */
  bonesForTalk: [],
  clothingBuckets: { outer: [], underwear: [], body: [], other: [] },
  clipBySemantic: {},
  /** 脸部 morph（优先） */
  faceMesh: null,
  /** 全部含 morph 的 mesh */
  morphMeshes: [],
  composer: null,
  /** 主姿态（idle / pose / talking 对应的 clip） */
  baseAction: null,
  /** 循环叠加动作 */
  loopAction: null,
  loopName: null,
  running: false,
  raf: 0,
  /** talking 程序化相位 */
  talkPhase: 0,
  isTalkingProcedural: false,
  /** rim / fill */
  rimLight: null,
  fillLight: null,
  currentClothingLevel: CLOTHING_LEVEL.FULL,
};

// ---------------------------------------------------------------------------
// 服装分层：traverse 收集 mesh，按 name 归类
// ---------------------------------------------------------------------------

function classifyClothingMeshes(root) {
  const hints = CONFIG.clothingNameHints;
  const buckets = state.clothingBuckets;
  Object.keys(buckets).forEach((k) => (buckets[k] = []));

  root.traverse((obj) => {
    if (!obj.isMesh) return;
    const n = (obj.name || '').toLowerCase();
    let placed = false;
    for (const key of ['outer', 'underwear', 'body']) {
      for (const h of hints[key] || []) {
        if (n.includes(h.toLowerCase())) {
          buckets[key].push(obj);
          placed = true;
          break;
        }
      }
      if (placed) break;
    }
    if (!placed) buckets.other.push(obj);
  });

  // 若 body 桶为空，把所有 SkinnedMesh 视作身体（避免全进 other 导致 0 级无效）
  if (state.clothingBuckets.body.length === 0) {
    root.traverse((o) => {
      if (o.isSkinnedMesh && !state.clothingBuckets.outer.includes(o) && !state.clothingBuckets.underwear.includes(o)) {
        state.clothingBuckets.body.push(o);
      }
    });
  }
}

/**
 * level: 0-3
 * 3 全套 | 2 脱外衣剩内衣 | 1 仅皮肤身体 | 0 裸体（身体 + 尽量关掉 other 里非皮肤）
 */
function changeClothes(level) {
  const lv = Math.max(0, Math.min(3, Number(level) | 0));
  state.currentClothingLevel = lv;
  const b = state.clothingBuckets;
  const setVis = (arr, v) => arr.forEach((m) => { m.visible = v; });

  const showOuter = lv >= CLOTHING_LEVEL.OUTER_OFF + 1;
  const showUnder = lv >= CLOTHING_LEVEL.UNDERWEAR_OFF + 1;

  setVis(b.outer, showOuter);
  setVis(b.underwear, showUnder);
  setVis(b.body, true);

  if (lv === CLOTHING_LEVEL.NUDE) {
    setVis(b.other, false);
    setVis(b.body, true);
  } else {
    setVis(b.other, true);
  }
}

// ---------------------------------------------------------------------------
// 皮肤材质 + Rim（近似 SSS：略高 sheen、低 roughness、暖色 emissive）
// ---------------------------------------------------------------------------

const _skinPhysicalMaterialCache = new WeakMap();

/**
 * 将皮肤 Standard 升级为 Physical：transmission + thickness + attenuation 模拟光在表皮下的散射前缘。
 * 与电影级随机 walk BSSRDF / 路径追踪仍不同，但是在 WebGL 实时里更「真」的一档。
 */
function enhanceSkinAndHairMaterials(root) {
  const skinLike = new Set([
    ...state.clothingBuckets.body.map((m) => m.uuid),
    ...state.skinnedMeshes.map((m) => m.uuid),
  ]);

  function upgradeToSkinPhysical(stdMat) {
    if (!stdMat || stdMat.isMeshStandardMaterial !== true) return stdMat;
    if (stdMat.isMeshPhysicalMaterial === true) {
      applySkinPhysicalParams(stdMat);
      return stdMat;
    }
    let phys = _skinPhysicalMaterialCache.get(stdMat);
    if (!phys) {
      phys = new THREE.MeshPhysicalMaterial();
      if (stdMat.map) phys.map = stdMat.map;
      if (stdMat.normalMap) phys.normalMap = stdMat.normalMap;
      if (stdMat.roughnessMap) phys.roughnessMap = stdMat.roughnessMap;
      if (stdMat.metalnessMap) phys.metalnessMap = stdMat.metalnessMap;
      if (stdMat.aoMap) phys.aoMap = stdMat.aoMap;
      if (stdMat.alphaMap) phys.alphaMap = stdMat.alphaMap;
      phys.color.copy(stdMat.color);
      phys.roughness = stdMat.roughness;
      phys.metalness = stdMat.metalness;
      phys.opacity = stdMat.opacity;
      phys.transparent = stdMat.transparent;
      phys.alphaTest = stdMat.alphaTest;
      phys.normalScale = stdMat.normalScale?.clone?.() || new THREE.Vector2(1, 1);
      _skinPhysicalMaterialCache.set(stdMat, phys);
    }
    applySkinPhysicalParams(phys);
    return phys;
  }

  function applySkinPhysicalParams(phys) {
    const base = phys.color?.clone?.() || new THREE.Color(0xffdbc4);
    phys.roughness = THREE.MathUtils.clamp(phys.roughness * 0.82, 0.32, 0.68);
    phys.metalness = THREE.MathUtils.clamp(phys.metalness * 0.35, 0, 0.08);
    phys.transmission = CONFIG.skinTransmission;
    phys.thickness = CONFIG.skinThickness;
    phys.ior = CONFIG.skinIor;
    phys.attenuationColor = new THREE.Color(CONFIG.skinAttenuationColor);
    phys.attenuationDistance = CONFIG.skinAttenuationDistance;
    phys.specularIntensity = 0.65;
    phys.specularColor = new THREE.Color(0xfff0eb);
    phys.sheen = 0.28;
    phys.sheenRoughness = 0.52;
    phys.sheenColor = new THREE.Color(0xffc8c4);
    phys.clearcoat = 0.12;
    phys.clearcoatRoughness = 0.38;
    phys.emissive = base.clone().multiplyScalar(0.018);
    phys.emissiveIntensity = 1;
    phys.envMapIntensity = 1.05;
  }

  root.traverse((obj) => {
    if (!obj.isMesh || !obj.material) return;
    const isSkinBucket = state.clothingBuckets.body.includes(obj);
    const isSkin = isSkinBucket || skinLike.has(obj.uuid);
    if (!isSkin) return;

    const mats = Array.isArray(obj.material) ? obj.material : [obj.material];
    const next = mats.map((mat) => upgradeToSkinPhysical(mat));
    obj.material = Array.isArray(obj.material) ? next : next[0];
  });
}

function addPortraitLights() {
  const rim = new THREE.DirectionalLight(0xaaccff, 0.55);
  rim.position.set(-2.2, 3.5, -3.8);
  rim.name = 'RimLight';
  state.scene.add(rim);
  state.rimLight = rim;

  const fill = new THREE.DirectionalLight(0xffc8e0, 0.28);
  fill.position.set(3, 1.8, 2);
  fill.name = 'FillLight';
  state.scene.add(fill);
  state.fillLight = fill;
}

// ---------------------------------------------------------------------------
// 骨骼：说话时轻微头 + 胸呼吸（程序化，叠加在 clip 之上）
// ---------------------------------------------------------------------------

const boneTalkQuat = new THREE.Quaternion();
const boneTalkEuler = new THREE.Euler();
const _boneRest = new Map();

function cacheBoneRestIfNeeded(bone) {
  if (!_boneRest.has(bone.uuid)) {
    _boneRest.set(bone.uuid, {
      q: bone.quaternion.clone(),
      p: bone.position.clone(),
    });
  }
}

function findTalkBones(root) {
  state.bonesForTalk = [];
  root.traverse((o) => {
    if (!o.isSkinnedMesh || !o.skeleton) return;
    o.skeleton.bones.forEach((bone) => {
      const n = (bone.name || '').toLowerCase();
      if (
        /head|neck|cervical|spine2|chest|spine1|upperchest|clavicle|spine/.test(n)
      ) {
        state.bonesForTalk.push(bone);
        cacheBoneRestIfNeeded(bone);
      }
    });
  });
}

function resetTalkBones() {
  state.bonesForTalk.forEach((bone) => {
    const r = _boneRest.get(bone.uuid);
    if (r) {
      bone.quaternion.copy(r.q);
      bone.position.copy(r.p);
    }
  });
}

function updateProceduralTalking(dt) {
  if (!state.isTalkingProcedural || state.bonesForTalk.length === 0) return;
  state.talkPhase += dt * 4.2;
  const headWag = Math.sin(state.talkPhase * 1.7) * 0.022;
  const nod = Math.sin(state.talkPhase * 2.8) * 0.014;
  const chest = Math.sin(state.talkPhase * 1.15) * 0.012;

  state.bonesForTalk.forEach((bone) => {
    const n = (bone.name || '').toLowerCase();
    if (/head|neck/.test(n)) {
      boneTalkEuler.set(nod * 0.55, headWag, 0, 'YXZ');
      boneTalkQuat.setFromEuler(boneTalkEuler);
      bone.quaternion.multiply(boneTalkQuat);
    }
    if (/chest|spine2|upperchest|spine1/.test(n)) {
      boneTalkEuler.set(chest, 0, 0, 'XYZ');
      boneTalkQuat.setFromEuler(boneTalkEuler);
      bone.quaternion.multiply(boneTalkQuat);
    }
  });
}

// ---------------------------------------------------------------------------
// 程序化循环（无 clip 时）
// ---------------------------------------------------------------------------

const procLoop = { name: null, phase: 0 };

function proceduralLoopTick(name, dt) {
  procLoop.phase += dt;
  const t = procLoop.phase;
  if (name === 'breathing') {
    const s = Math.sin(t * 1.2) * 0.022;
    state.bonesForTalk.forEach((bone) => {
      const n = (bone.name || '').toLowerCase();
      if (!/chest|spine2|upperchest|spine1|spine/.test(n)) return;
      boneTalkEuler.set(s, 0, 0, 'XYZ');
      boneTalkQuat.setFromEuler(boneTalkEuler);
      bone.quaternion.multiply(boneTalkQuat);
    });
    return;
  }
  if (name === 'stroke' || name === 'self-touch') {
    const yaw = Math.sin(t * 0.9) * (name === 'stroke' ? 0.05 : 0.035);
    state.bonesForTalk.forEach((bone) => {
      const n = (bone.name || '').toLowerCase();
      if (/head|neck/.test(n)) {
        boneTalkEuler.set(Math.sin(t * 0.7) * 0.018, yaw * 0.28, 0, 'YXZ');
        boneTalkQuat.setFromEuler(boneTalkEuler);
        bone.quaternion.multiply(boneTalkQuat);
      }
      if (name === 'self-touch' && /spine|chest|upperchest|clavicle|arm|hand/.test(n)) {
        boneTalkEuler.set(0, yaw * 0.12, Math.sin(t * 1.1) * 0.025, 'XYZ');
        boneTalkQuat.setFromEuler(boneTalkEuler);
        bone.quaternion.multiply(boneTalkQuat);
      }
    });
  }
}

// ---------------------------------------------------------------------------
// Clip 解析
// ---------------------------------------------------------------------------

function resolveClipsFromGltf(gltf) {
  const clips = gltf.animations || [];
  const semantic = {};

  function findClip(keywords) {
    const lower = (keywords || []).map((k) => k.toLowerCase());
    for (const clip of clips) {
      const name = (clip.name || '').toLowerCase();
      if (lower.some((kw) => name.includes(kw))) return clip;
    }
    return null;
  }

  for (const sem of Object.keys(CONFIG.animAlias)) {
    const clip = findClip(CONFIG.animAlias[sem]);
    if (clip) semantic[sem] = clip;
  }
  if (!semantic.idle && clips[0]) semantic.idle = clips[0];
  if (!semantic.talking && semantic.idle) semantic.talking = semantic.idle;
  for (let i = 1; i <= 5; i++) {
    const key = `pose${i}`;
    if (!semantic[key] && semantic.idle) semantic[key] = semantic.idle;
  }

  state.clipBySemantic = semantic;
}

function stopLoop() {
  if (state.loopAction) {
    state.loopAction.fadeOut(0.25);
    state.loopAction.stop();
    state.loopAction = null;
  }
  state.loopName = null;
  procLoop.name = null;
}

function startLoop(name) {
  stopLoop();
  if (!state.mixer || !name) return;
  const key = String(name).toLowerCase();
  const clip = state.clipBySemantic[key];
  state.loopName = key;
  procLoop.name = null;

  if (clip) {
    const act = state.mixer.clipAction(clip);
    act.reset().fadeIn(0.3).setLoop(THREE.LoopRepeat, Infinity);
    /** 与 base 叠加：权重低一些 */
    act.setEffectiveWeight(0.55);
    act.play();
    state.loopAction = act;
    return;
  }

  state.bonesForTalk.forEach((b) => cacheBoneRestIfNeeded(b));
  procLoop.name = key;
  procLoop.phase = 0;
}

function playPose(name) {
  if (!state.mixer) return;
  const raw = String(name || 'idle').toLowerCase();
  stopLoop();

  const isTalking = raw === 'talking' || raw === 'talk';
  state.isTalkingProcedural = isTalking;
  if (!isTalking) {
    state.talkPhase = 0;
    resetTalkBones();
  }

  let clipKey = raw;
  if (isTalking) clipKey = 'talking';
  if (raw.startsWith('pose')) clipKey = raw;

  const clip =
    state.clipBySemantic[clipKey] ||
    state.clipBySemantic.idle;

  if (!clip) return;

  const next = state.mixer.clipAction(clip);
  next.reset().fadeIn(0.3).setLoop(THREE.LoopRepeat, Infinity).play();
  next.setEffectiveWeight(1);

  if (state.baseAction && state.baseAction !== next) {
    state.baseAction.fadeOut(0.3);
  }
  state.baseAction = next;
}

// ---------------------------------------------------------------------------
// Morph：表情预设（微笑 / 害羞 / 诱惑）
// ---------------------------------------------------------------------------

const EXPRESSION_PRESETS = {
  微笑: ['smile', 'mouthsmile', 'mouth_smile', 'happy', 'joy'],
  smile: ['smile', 'mouthsmile', 'mouth_smile', 'happy', 'joy'],
  害羞: ['shy', 'blush', 'brow', 'embarrass', 'sad'],
  shy: ['shy', 'blush', 'brow', 'embarrass'],
  诱惑: ['seductive', 'wink', 'lips', 'sexy', 'kiss', 'smirk'],
  seductive: ['seductive', 'wink', 'lips', 'sexy', 'kiss', 'smirk'],
  neutral: [],
};

function fuzzyMorphIndex(mesh, hints) {
  const md = mesh.morphTargetDictionary;
  if (!md) return null;
  const lowerHints = hints.map((h) => h.toLowerCase());
  for (const key of Object.keys(md)) {
    const lk = key.toLowerCase();
    if (lowerHints.some((h) => lk.includes(h))) return { key, idx: md[key] };
  }
  return null;
}

function clearMorphs(mesh) {
  if (!mesh || !mesh.morphTargetInfluences) return;
  for (let i = 0; i < mesh.morphTargetInfluences.length; i++) {
    mesh.morphTargetInfluences[i] = 0;
  }
}

/**
 * @param {string|Record<string, number>} nameOrDict 预设名或多 morph 字典
 * @param {number} [strength] 0-1
 */
function setExpression(nameOrDict, strength = 1) {
  const str = THREE.MathUtils.clamp(strength, 0, 1);
  if (nameOrDict && typeof nameOrDict === 'object' && !Array.isArray(nameOrDict)) {
    const mesh = state.faceMesh || state.morphMeshes[0];
    if (!mesh || !mesh.morphTargetDictionary) return;
    const infl = mesh.morphTargetInfluences;
    const md = mesh.morphTargetDictionary;
    for (const [key, val] of Object.entries(nameOrDict)) {
      const idx = md[key];
      if (idx !== undefined) infl[idx] = THREE.MathUtils.clamp(val, 0, 1);
    }
    return;
  }

  const presetKey = String(nameOrDict || 'neutral').toLowerCase();
  const hints = EXPRESSION_PRESETS[presetKey] || EXPRESSION_PRESETS.微笑;

  state.morphMeshes.forEach((mesh) => {
    if (!mesh.morphTargetDictionary) return;
    clearMorphs(mesh);
    if (presetKey === 'neutral' || hints.length === 0) return;
    const hit = fuzzyMorphIndex(mesh, hints);
    if (hit) mesh.morphTargetInfluences[hit.idx] = str;
  });
}

function collectMorphMeshes(root) {
  state.morphMeshes = [];
  state.faceMesh = null;
  root.traverse((o) => {
    if (!o.isSkinnedMesh || !o.morphTargetDictionary) return;
    state.morphMeshes.push(o);
    const n = (o.name || '').toLowerCase();
    if (/head|face|mesh/.test(n) || !state.faceMesh) state.faceMesh = o;
  });
}

// ---------------------------------------------------------------------------
// Three 初始化
// ---------------------------------------------------------------------------

function initThreeBasics() {
  state.container = document.getElementById('avatar3d-canvas-wrap');
  if (!state.container) throw new Error('#avatar3d-canvas-wrap missing');

  state.scene = new THREE.Scene();
  state.scene.background = new THREE.Color(0x06060a);

  const r = state.container.getBoundingClientRect();
  const asp = r.width / Math.max(r.height, 1);

  state.camera = new THREE.PerspectiveCamera(42, asp, 0.1, 200);
  state.camera.position.set(0, 1.35, 3.2);
  state.camera.lookAt(0, 1, 0);

  state.renderer = new THREE.WebGLRenderer({
    antialias: true,
    alpha: true,
    powerPreference: 'high-performance',
  });
  state.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  state.renderer.setSize(r.width, r.height, false);
  state.renderer.setClearColor(0x000000, 0);
  state.renderer.outputColorSpace = THREE.SRGBColorSpace;
  state.renderer.toneMapping = THREE.ACESFilmicToneMapping;
  state.renderer.toneMappingExposure = 1.02;
  state.renderer.shadowMap.enabled = true;
  state.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
  if ('useLegacyLights' in state.renderer) {
    state.renderer.useLegacyLights = false;
  }
  state.container.appendChild(state.renderer.domElement);

  state.controls = new OrbitControls(state.camera, state.renderer.domElement);
  state.controls.target.set(0, 1, 0);
  state.controls.enableDamping = true;
  state.controls.dampingFactor = 0.06;
  state.controls.minDistance = 1.2;
  state.controls.maxDistance = 8;
  state.controls.maxPolarAngle = Math.PI * 0.49;
  state.controls.minAzimuthAngle = -Infinity;
  state.controls.maxAzimuthAngle = Infinity;
  // 手机单指旋转：ONE 用 ROTATE；桌面仍为一键旋转 + 右键平移 + 滚轮远近
  if (state.controls.touches) {
    state.controls.touches = {
      ONE: THREE.TOUCH.ROTATE,
      TWO: THREE.TOUCH.DOLLY_PAN,
    };
  }

  const hemi = new THREE.HemisphereLight(0x9078c8, 0x1a1a24, 0.38);
  state.scene.add(hemi);
  const dir = new THREE.DirectionalLight(0xfff5f0, 0.72);
  dir.position.set(2.2, 5.5, 2.8);
  dir.castShadow = true;
  dir.shadow.mapSize.set(2048, 2048);
  dir.shadow.bias = -0.0001;
  state.scene.add(dir);

  addPortraitLights();

  const ground = new THREE.Mesh(
    new THREE.CircleGeometry(4, 48),
    new THREE.MeshStandardMaterial({
      color: 0x12121a,
      roughness: 0.92,
      metalness: 0.04,
    })
  );
  ground.rotation.x = -Math.PI / 2;
  ground.receiveShadow = true;
  state.scene.add(ground);
}

async function loadEnvironmentHdr() {
  const pmrem = new THREE.PMREMGenerator(state.renderer);
  pmrem.compileEquirectangularShader();
  return new Promise((resolve) => {
    new RGBELoader().load(
      CONFIG.hdrUrl,
      (tex) => {
        tex.mapping = THREE.EquirectangularReflectionMapping;
        const env = pmrem.fromEquirectangular(tex).texture;
        state.scene.environment = env;
        tex.dispose();
        pmrem.dispose();
        resolve(true);
      },
      undefined,
      () => {
        pmrem.dispose();
        resolve(false);
      }
    );
  });
}

async function loadCharacter() {
  return new Promise((resolve, reject) => {
    new GLTFLoader().load(
      CONFIG.modelUrl,
      (gltf) => {
        const root = gltf.scene;
        state.characterRoot = root;
        state.skinnedMeshes = [];
        root.traverse((o) => {
          if (o.isSkinnedMesh) {
            o.frustumCulled = false;
            o.castShadow = true;
            o.receiveShadow = true;
            state.skinnedMeshes.push(o);
          } else if (o.isMesh) {
            o.castShadow = true;
            o.receiveShadow = true;
          }
        });

        const box = new THREE.Box3().setFromObject(root);
        const c = box.getCenter(new THREE.Vector3());
        const size = box.getSize(new THREE.Vector3());
        const maxDim = Math.max(size.x, size.y, size.z) || 1;
        const targetH = 1.85;
        const s = targetH / maxDim;
        root.scale.setScalar(s);
        root.position.sub(c.multiplyScalar(s));
        root.position.y = 0;

        state.scene.add(root);
        classifyClothingMeshes(root);
        collectMorphMeshes(root);
        findTalkBones(root);
        enhanceSkinAndHairMaterials(root);
        resolveClipsFromGltf(gltf);
        state.mixer = new THREE.AnimationMixer(root);

        changeClothes(CLOTHING_LEVEL.FULL);
        setExpression('neutral', 0);
        playPose('idle');

        resolve();
      },
      undefined,
      reject
    );
  });
}

function initPostProcessing() {
  const r = state.container.getBoundingClientRect();
  const w = r.width;
  const h = r.height;
  const composer = new EffectComposer(state.renderer);
  composer.addPass(new RenderPass(state.scene, state.camera));
  composer.addPass(
    new UnrealBloomPass(
      new THREE.Vector2(w, h),
      CONFIG.bloomStrength,
      CONFIG.bloomRadius,
      CONFIG.bloomThreshold
    )
  );
  const ssao = new SSAOPass(state.scene, state.camera, w, h);
  ssao.kernelRadius = CONFIG.ssaoKernelRadius;
  ssao.minDistance = CONFIG.ssaoMinDistance;
  ssao.maxDistance = CONFIG.ssaoMaxDistance;
  ssao.output = SSAOPass.OUTPUT.Default;
  composer.addPass(ssao);
  composer.addPass(new OutputPass());
  state.composer = composer;
}

function onResize() {
  if (!state.container || !state.camera) return;
  const rect = state.container.getBoundingClientRect();
  const w = rect.width;
  const h = Math.max(rect.height, 1);
  state.camera.aspect = w / h;
  state.camera.updateProjectionMatrix();
  state.renderer.setSize(w, h, false);
  if (state.composer) {
    state.composer.setSize(w, h);
    state.composer.passes.forEach((p) => {
      if (p.setSize) p.setSize(w, h);
    });
  }
}

function animateLoop() {
  if (!state.running) return;
  state.raf = requestAnimationFrame(animateLoop);
  const dt = state.clock.getDelta();

  if (state.controls) state.controls.update();
  if (state.mixer) state.mixer.update(dt);

  if (state.isTalkingProcedural) {
    state.bonesForTalk.forEach((b) => cacheBoneRestIfNeeded(b));
    updateProceduralTalking(dt);
  } else if (procLoop.name) {
    state.bonesForTalk.forEach((b) => cacheBoneRestIfNeeded(b));
    proceduralLoopTick(procLoop.name, dt);
  }

  if (state.composer) state.composer.render();
  else state.renderer.render(state.scene, state.camera);
}

export async function initSihanAvatar3D() {
  if (state.renderer) {
    onResize();
    return;
  }
  initThreeBasics();
  await loadEnvironmentHdr();
  initPostProcessing();
  window.addEventListener('resize', onResize);
  onResize();
  try {
    await loadCharacter();
  } catch (e) {
    console.warn('[avatar3d] glTF/HDR 加载失败', e);
  }
  state.running = true;
  animateLoop();
}

export function stopSihanAvatar3D() {
  state.running = false;
  if (state.raf) cancelAnimationFrame(state.raf);
  window.removeEventListener('resize', onResize);
  stopLoop();
  if (state.renderer) {
    state.container?.removeChild(state.renderer.domElement);
    state.renderer.dispose();
  }
  state.characterRoot = null;
  _boneRest.clear();
  Object.assign(state, {
    container: null,
    scene: null,
    camera: null,
    renderer: null,
    controls: null,
    mixer: null,
    composer: null,
    baseAction: null,
    loopAction: null,
    skinnedMeshes: [],
    clipBySemantic: {},
    faceMesh: null,
    morphMeshes: [],
    bonesForTalk: [],
    clothingBuckets: { outer: [], underwear: [], body: [], other: [] },
    rimLight: null,
    fillLight: null,
  });
}

const virtualAvatarApi = {
  changeClothes,
  playPose,
  setExpression,
  startLoop,
  stopLoop,
};

export const VirtualAvatar = {
  ...virtualAvatarApi,
  get CONFIG() {
    return CONFIG;
  },
};

if (typeof window !== 'undefined') {
  window.virtualAvatar = virtualAvatarApi;
  window.VirtualAvatar = VirtualAvatar;
  window.initSihanAvatar3D = initSihanAvatar3D;
}
