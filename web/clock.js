import * as THREE from "./vendor/three.module.min.js";
const D = await (await fetch("data.json")).json();
const COL = {en_days:"#4cc9f0", hi_days:"#ff9f1c", ta_days:"#f72585", en_months:"#4cc9f0", hi_months:"#ff9f1c"};
const KEYS = Object.keys(COL), LAYERS = Object.keys(D.proj).map(Number).sort((a,b)=>a-b);
const $ = s => document.querySelector(s);
const SCALE = 2.1;

/* ---------- three scene ---------- */
const cvs = $("#stage"), renderer = new THREE.WebGLRenderer({canvas:cvs, antialias:true, alpha:true});
renderer.setPixelRatio(Math.min(2, devicePixelRatio));
const scene = new THREE.Scene(), cam = new THREE.PerspectiveCamera(45, 1, .1, 100); cam.position.set(0, 0, 10.5);
const outer = new THREE.Group(), spin = new THREE.Group(); outer.add(spin); scene.add(outer);
outer.rotation.x = -1.0;
function resize(){ const w = innerWidth, h = innerHeight; renderer.setSize(w, h, false); cam.aspect = w/h; cam.updateProjectionMatrix(); outer.position.x = w > 900 ? 2.6 : 0; outer.position.y = w > 900 ? 0 : 1.6 }
resize(); addEventListener("resize", resize);

function glowTex(){ const c = document.createElement("canvas"); c.width = c.height = 128; const g = c.getContext("2d"), r = g.createRadialGradient(64,64,0,64,64,64);
  r.addColorStop(0,"rgba(255,255,255,1)"); r.addColorStop(.18,"rgba(255,255,255,.9)"); r.addColorStop(.4,"rgba(255,255,255,.25)"); r.addColorStop(1,"rgba(255,255,255,0)");
  g.fillStyle = r; g.fillRect(0,0,128,128); return new THREE.CanvasTexture(c) }
const TEX = glowTex();

const sets = {};
for (const k of KEYS) {
  const names = D.names[k], n = names.length, col = new THREE.Color(COL[k]);
  const pts = names.map(() => { const s = new THREE.Sprite(new THREE.SpriteMaterial({map:TEX, color:col, transparent:true, blending:THREE.AdditiveBlending, depthWrite:false})); s.scale.setScalar(.55); spin.add(s); return s });
  const lg = new THREE.BufferGeometry(); lg.setAttribute("position", new THREE.BufferAttribute(new Float32Array((n+1)*3), 3));
  const line = new THREE.Line(lg, new THREE.LineBasicMaterial({color:col, transparent:true, opacity:0})); spin.add(line);
  const labels = names.map(t => { const d = document.createElement("div"); d.className = "lb"; d.textContent = t; d.style.color = COL[k]; if (k.startsWith("ta")) d.style.fontSize = "11px"; $("#labels").appendChild(d); return d });
  sets[k] = {pts, line, labels, n, op:0, tgt:0, cur: names.map(() => new THREE.Vector3())};
}
const twin = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({color:0xffffff, transparent:true, opacity:0}));
twin.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(7*2*3), 3)); spin.add(twin); let twinTgt = 0;
const marker = new THREE.Sprite(new THREE.SpriteMaterial({map:TEX, color:0xffffff, transparent:true, blending:THREE.AdditiveBlending, depthWrite:false, opacity:0})); marker.scale.setScalar(1.1); spin.add(marker);
let markerTgt = 0;

function posAt(k, L){ // interpolate projections between the layers we exported
  L = Math.max(LAYERS[0], Math.min(LAYERS.at(-1), L)); let i = 0; while (i < LAYERS.length-2 && LAYERS[i+1] < L) i++;
  const a = LAYERS[i], b = LAYERS[i+1], f = (L-a)/(b-a), A = D.proj[a][k], B = D.proj[b][k];
  return A.map((p,j) => new THREE.Vector3(p[0]+(B[j][0]-p[0])*f, p[1]+(B[j][1]-p[1])*f, p[2]+(B[j][2]-p[2])*f).multiplyScalar(SCALE));
}

/* ---------- state per story step ---------- */
const state = {layer:20, layerTarget:20, step:0, spinSpeed:.0016, drag:false};
const STEPS = [
  {show:{en_days:1}, layer:20}, {show:{en_days:1}, layer:20}, {show:{en_days:.55, hi_days:1}, layer:null, anim:true, twin:true},
  {show:{en_days:.55, ta_days:1}, layer:20}, {show:{en_months:1, hi_months:1}, layer:20}, {show:{en_days:.8}, layer:20}, {show:{en_days:1}, layer:12, marker:true},
  {show:{en_days:.5}, layer:20}, {show:{en_days:.4}, layer:20}];
let animated = false;
function setStep(i){
  state.step = i; const S = STEPS[i];
  for (const k of KEYS) sets[k].tgt = S.show[k] || 0;
  twinTgt = S.twin ? .35 : 0; markerTgt = S.marker ? 1 : 0;
  if (S.layer != null) state.layerTarget = S.layer;
  if (S.anim && !animated) { animated = true; state.layer = 2; state.layerTarget = 20; $("#layer").value = 20; }
  if (i === 2 && animated) state.layerTarget = +$("#layer").value;
  drawHud(); dialUpdate();
}
new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) setStep(+e.target.dataset.step) }), {rootMargin:"-45% 0px -45% 0px"})
  .observe && document.querySelectorAll("section").forEach(s => new IntersectionObserver(es => es.forEach(e => { if (e.isIntersecting) setStep(+e.target.dataset.step) }), {rootMargin:"-45% 0px -45% 0px"}).observe(s));

/* ---------- drag to rotate ---------- */
let last = null;
addEventListener("pointerdown", e => { if (e.target.closest("main .box, #hud, nav")) return; last = [e.clientX, e.clientY]; state.drag = true });
addEventListener("pointermove", e => { if (!last) return; spin.rotation.z += (e.clientX-last[0])*.006; outer.rotation.x = Math.max(-1.5, Math.min(-.2, outer.rotation.x + (e.clientY-last[1])*.004)); last = [e.clientX, e.clientY] });
addEventListener("pointerup", () => { last = null; state.drag = false });

/* ---------- render loop ---------- */
const v = new THREE.Vector3();
function frame(){
  state.layer += (state.layerTarget - state.layer) * .06;
  if (!state.drag) spin.rotation.z += state.spinSpeed;
  for (const k of KEYS) {
    const s = sets[k]; s.op += (s.tgt - s.op) * .08;
    const P = posAt(k, state.layer);
    P.forEach((p, i) => { s.cur[i].lerp(p, .2); s.pts[i].position.copy(s.cur[i]); s.pts[i].material.opacity = s.op; s.pts[i].scale.setScalar(.5 + .12*Math.sin(performance.now()*.002 + i)) });
    const a = s.line.geometry.attributes.position; s.cur.forEach((p, i) => a.setXYZ(i, p.x, p.y, p.z)); a.setXYZ(s.n, s.cur[0].x, s.cur[0].y, s.cur[0].z); a.needsUpdate = true;
    s.line.material.opacity = s.op * .35; s.line.visible = s.op > .02;
    s.labels.forEach((el, i) => { v.copy(s.cur[i]); spin.localToWorld(v); v.project(cam); el.style.left = (v.x*.5+.5)*innerWidth + "px"; el.style.top = (-v.y*.5+.5)*innerHeight + "px"; el.style.opacity = Math.min(1, s.op) * (s.op > .05 ? 1 : 0) });
  }
  const ta = twin.geometry.attributes.position; for (let i = 0; i < 7; i++) { const a = sets.en_days.cur[i], b = sets.hi_days.cur[i]; ta.setXYZ(i*2, a.x, a.y, a.z); ta.setXYZ(i*2+1, b.x, b.y, b.z) } ta.needsUpdate = true;
  twin.material.opacity += (twinTgt - twin.material.opacity) * .08;
  marker.material.opacity += (markerTgt - marker.material.opacity) * .1; if (marker.userData.to) marker.position.lerp(marker.userData.to, .25);
  $("#layerNum").textContent = Math.round(state.layer);
  renderer.render(scene, cam); requestAnimationFrame(frame);
}
frame();

/* ---------- step 1: layer slider ---------- */
const ringAt = (k, L) => D.ring[k][Math.max(1, Math.min(28, Math.round(L))) - 1];
const alAt = L => D.align["en_days->hi_days"][Math.max(1, Math.min(28, Math.round(L))) - 1];
$("#layer").addEventListener("input", e => { state.layerTarget = +e.target.value; drawHud() });
setInterval(() => { if (state.step === 2) { $("#alNow").textContent = alAt(state.layer).toFixed(2); drawHud() } }, 150);

/* ---------- headline numbers ---------- */
const fmt = p => p < 0.0001 ? "<0.0001" : p.toFixed(4);
$("#pEn").textContent = fmt(D.ring_stats.en_days.p_layer_corrected); $("#pTa").textContent = D.ring_stats.ta_days.p_layer_corrected.toFixed(2);
const bestAl = key => Math.max(...D.align[key]);
$("#alHi").textContent = bestAl("en_days->hi_days").toFixed(2); $("#alHiM").textContent = bestAl("en_months->hi_months").toFixed(2);
$("#rMon").textContent = D.ring_stats.en_months.best_R.toFixed(2);
const rw = D.random_words_best_R; $("#rndMean").textContent = (rw.reduce((a,b)=>a+b)/rw.length).toFixed(2); $("#rndMax").textContent = Math.max(...rw).toFixed(2); $("#dayR").textContent = D.ring_stats.en_days.best_R.toFixed(2);

/* ---------- HUD charts ---------- */
const hud = $("#hud"), ch = $("#chart"), g = ch.getContext("2d");
function axes(title, note){ $("#hudTitle").textContent = title; $("#hudNote").textContent = note || ""; g.clearRect(0,0,800,260); g.strokeStyle = "rgba(255,255,255,.12)"; g.fillStyle = "#98a0b8"; g.font = "20px system-ui";
  for (let y = 0; y <= 1; y += .5) { const Y = 220 - y*190; g.beginPath(); g.moveTo(50, Y); g.lineTo(790, Y); g.stroke(); g.fillText(y.toFixed(1), 8, Y+6) } }
function lines(series, marker){ const X = L => 50 + (L-1)/27*700;
  series.forEach(s => { g.strokeStyle = s.color; g.lineWidth = 3; g.beginPath(); s.vals.forEach((y, i) => { const px = X(i+1), py = 220 - Math.max(0, y)*190; i ? g.lineTo(px, py) : g.moveTo(px, py) }); g.stroke() });
  g.fillStyle = "#98a0b8"; [1,7,14,21,28].forEach(L => g.fillText("L"+L, X(L)-12, 250));
  if (marker) { g.strokeStyle = "#fff"; g.lineWidth = 1.5; g.beginPath(); g.moveTo(X(marker), 20); g.lineTo(X(marker), 222); g.stroke() }
  let lx = 60; series.forEach(s => { g.fillStyle = s.color; g.fillRect(lx, 8, 16, 6); g.fillStyle = "#c4cade"; g.fillText(s.label, lx+22, 16); lx += 30 + g.measureText(s.label).width + 22 }) }
function drawHud(){
  const st = state.step; hud.style.display = [1,2,3,4,5,7].includes(st) ? "block" : "none"; if (hud.style.display === "none") return;
  if (st === 1) { axes("Weekday ring score by layer", "A perfect ring scores 1. It is high from layer 1 on; the dip at the final layer is where the model prepares its output."); lines([{label:"English days", color:COL.en_days, vals:D.ring.en_days}]) }
  if (st === 2) { axes("Hindi-to-English alignment by layer", "Low in the first layers, close to 1 from about layer 14."); lines([{label:"days", color:COL.hi_days, vals:D.align["en_days->hi_days"]},{label:"months", color:"#ffd6a0", vals:D.align["en_months->hi_months"]}], state.layer) }
  if (st === 4) { axes("Month ring score by layer", "The 12-month ring builds up over the first ~7 layers."); lines([{label:"English months", color:COL.en_months, vals:D.ring.en_months},{label:"Hindi months", color:COL.hi_months, vals:D.ring.hi_months}]) }
  if (st === 3) { axes("Ring score by layer", "Tamil names never reach the English level."); lines([{label:"English", color:COL.en_days, vals:D.ring.en_days},{label:"Tamil", color:COL.ta_days, vals:D.ring.ta_days}]) }
  if (st === 5) { axes("30 random-word sets vs weekdays", "Best-layer ring score, 7 words each. Vertical line = real weekdays."); const bins = new Array(10).fill(0); rw.forEach(x => bins[Math.min(9, Math.floor(x*10))]++); const mx = Math.max(...bins);
    bins.forEach((c, i) => { g.fillStyle = "rgba(152,160,184,.55)"; const h = c/mx*150; g.fillRect(60 + i*70, 220-h, 58, h); g.fillStyle = "#98a0b8"; g.fillText((i/10).toFixed(1), 60+i*70, 245) });
    const x = 60 + D.ring_stats.en_days.best_R*700; g.strokeStyle = COL.en_days; g.lineWidth = 3; g.beginPath(); g.moveTo(x, 30); g.lineTo(x, 222); g.stroke(); g.fillStyle = COL.en_days; g.fillText("weekdays", x-70, 24) }
  if (st === 7) { axes("Dial accuracy by number of harmonics", "Share of dial settings where the model gave the intended day. Chance = 0.14."); const m = D.measured_dial_spike, Ls = ["6","8","10","12","14"], C = ["#6c7aa8","#7fb7ff",COL.en_days];
    Ls.forEach((L, i) => [1,2,3].forEach((h, j) => { const val = m[L][h]; const x = 70 + i*145 + j*38, hh = val*190; g.fillStyle = C[j]; g.fillRect(x, 220-hh, 32, hh) })); Ls.forEach((L, i) => { g.fillStyle = "#98a0b8"; g.fillText("L"+L, 95 + i*145, 248) });
    let lx = 60; [["1 harmonic",C[0]],["2",C[1]],["3 (full ring)",C[2]]].forEach(([t,c]) => { g.fillStyle = c; g.fillRect(lx, 8, 16, 6); g.fillStyle = "#c4cade"; g.fillText(t, lx+22, 16); lx += 150 }) }
}

/* ---------- step 6: the dial ---------- */
const DAYS = D.names.en_days; let src = 4, mode = "ring", live = false, req = 0;
const dayBtns = $("#dayBtns"); DAYS.forEach((d, i) => { const b = document.createElement("button"); b.className = "btn" + (i === src ? " on" : ""); b.textContent = d; b.onclick = () => { src = i; [...dayBtns.children].forEach((c, j) => c.classList.toggle("on", j === i)); dialUpdate() }; dayBtns.appendChild(b) });
document.querySelectorAll("[data-mode]").forEach(b => b.onclick = () => { mode = b.dataset.mode; document.querySelectorAll("[data-mode]").forEach(c => c.classList.toggle("on", c === b)); dialUpdate() });
$("#dial").addEventListener("input", dialUpdate);
const bars = $("#bars"); DAYS.forEach(d => { bars.insertAdjacentHTML("beforeend", `<span>${d.slice(0,3)}</span><div class="bar"><i></i></div><span class="pv">0</span>`) });
const NOTE = {ring:"Replaying real runs of Sarvam-1: its internal 'today' is replaced by a point on the day ring (3 harmonics, layer 12).", chord:"Control: move along a straight line between neighbouring days instead of around the ring.", scrambled:"Control: the same machinery, but with the days wired in a scrambled order. The answers should land on the wrong days."};
async function dialUpdate(){
  const t = +$("#dial").value; $("#dialVal").textContent = t.toFixed(2); $("#srcName").textContent = DAYS[src];
  let probs = D.dial.sweep[mode][DAYS[src]][Math.round(t*4)];
  const id = ++req;
  try { const r = await fetch(`/api/dial?day=${src}&t=${t}&mode=${mode}`, {signal: AbortSignal.timeout(4000)}); if (r.ok) { const j = await r.json(); if (id === req) { probs = j.probs; live = true } } } catch (e) { if (!live) live = false }
  const badge = $("#liveBadge"); badge.textContent = live ? "live · model running on this laptop" : "replay of real runs"; badge.style.color = live ? "var(--ok)" : "var(--warn)"; badge.style.borderColor = badge.style.color;
  const mx = probs.indexOf(Math.max(...probs)); $("#ans").textContent = DAYS[mx];
  const pos = (src + t) , lo = Math.floor(pos), hi = Math.ceil(pos); $("#exp").textContent = lo === hi ? DAYS[(lo+1)%7] : DAYS[(lo+1)%7].slice(0,3) + "/" + DAYS[(hi+1)%7].slice(0,3);
  [...bars.querySelectorAll(".bar i")].forEach((el, i) => el.style.width = (probs[i]*100) + "%"); [...bars.querySelectorAll(".pv")].forEach((el, i) => el.textContent = (probs[i]*100).toFixed(0) + "%");
  $("#dialNote").textContent = NOTE[mode];
  const P = sets.en_days.cur, a = P[((lo%7)+7)%7], b = P[((hi%7)+7)%7], f = pos - lo; marker.userData.to = new THREE.Vector3().lerpVectors(a, b, f);
}
dialUpdate(); setStep(0);
