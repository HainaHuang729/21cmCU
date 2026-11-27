"""Interactive 3D WebGL HTML visualization and 2D publication plotting utilities."""

from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
import zlib

import numpy as np

if TYPE_CHECKING:
    from py21cmfast.drivers.coeval import Coeval
    from py21cmfast.drivers.lightcone import LightCone

_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>__TITLE__</title>
<style>
  :root {
    --bg: #0b1020;
    --panel: #131a2e;
    --border: #25314d;
    --text: #e6ecf8;
    --muted: #93a3c7;
    --accent: #38bdf8;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg); color: var(--text); display: flex; height: 100vh; overflow: hidden;
  }
  #sidebar {
    width: 310px; background: var(--panel); border-right: 1px solid var(--border);
    padding: 16px; display: flex; flex-direction: column; gap: 12px; overflow-y: auto;
  }
  h1 { font-size: 16px; margin: 0; color: #fff; }
  .meta { font-size: 12px; color: var(--muted); line-height: 1.45; }
  .ctrl { display: flex; flex-direction: column; gap: 4px; font-size: 12px; }
  .ctrl label { display: flex; justify-content: space-between; color: var(--muted); }
  .ctrl span.val { color: var(--accent); font-family: monospace; }
  input[type="range"] { width: 100%; accent-color: var(--accent); }
  select, button {
    background: #1b2440; color: var(--text); border: 1px solid var(--border);
    border-radius: 6px; padding: 6px 10px; font-size: 12px; cursor: pointer;
  }
  button:hover { border-color: var(--accent); }
  .row { display: flex; gap: 8px; flex-wrap: wrap; }
  .chk { display: flex; align-items: center; gap: 6px; font-size: 12px; color: var(--text); }
  #viewport { flex: 1; position: relative; background: radial-gradient(circle at center, #141d38 0%, #070a14 100%); }
  canvas { width: 100%; height: 100%; display: block; cursor: grab; }
  canvas:active { cursor: grabbing; }
  #hud {
    position: absolute; top: 12px; right: 14px; background: rgba(11,16,32,0.78);
    border: 1px solid var(--border); border-radius: 8px; padding: 8px 12px;
    font-size: 12px; color: var(--muted); pointer-events: none;
  }
  #cbar {
    height: 12px; border-radius: 4px; border: 1px solid var(--border); margin-top: 4px;
  }
</style>
</head>
<body>
<div id="sidebar">
  <h1>__TITLE__</h1>
  <div class="meta" id="stats"></div>
  <div class="ctrl">
    <label><span>Render Mode</span></label>
    <select id="modeSel">
      <option value="0">Volumetric + Slices</option>
      <option value="1">Slices Only</option>
      <option value="2">Volume Only</option>
    </select>
  </div>
  <div class="ctrl">
    <label><span>Colormap</span></label>
    <select id="cmapSel">
      <option value="0">EoR 21cm</option>
      <option value="1">Viridis</option>
      <option value="2">Inferno</option>
      <option value="3">Plasma</option>
      <option value="4">Coolwarm</option>
    </select>
    <div id="cbar"></div>
    <label><span id="vminTxt"></span><span id="vmaxTxt"></span></label>
  </div>
  <div class="ctrl">
    <label><span>Z / Redshift Slice</span><span class="val" id="zVal"></span></label>
    <input type="range" id="zSlice" min="0" max="1" step="0.005" value="0.50" />
  </div>
  <div class="ctrl">
    <label><span>Y Slice</span><span class="val" id="yVal"></span></label>
    <input type="range" id="ySlice" min="0" max="1" step="0.01" value="0.50" />
  </div>
  <div class="ctrl">
    <label><span>X Slice</span><span class="val" id="xSlice" min="0" max="1" step="0.01" value="0.50" />
  </div>
  <div class="ctrl">
    <label><span>Iso / Bubble Threshold</span><span class="val" id="thrVal"></span></label>
    <input type="range" id="thr" min="0" max="1" step="0.01" value="0.15" />
  </div>
  <div class="ctrl">
    <label><span>Volume Density</span><span class="val" id="densVal">0.90</span></label>
    <input type="range" id="dens" min="0.05" max="3.0" step="0.05" value="0.90" />
  </div>
  <div class="chk"><input type="checkbox" id="invertChk" /> Invert opacity mask</div>
  <div class="row">
    <button id="resetBtn">Reset View</button>
    <button id="snapBtn">Save PNG</button>
  </div>
</div>
<div id="viewport">
  <canvas id="glcanvas"></canvas>
  <div id="hud">Drag: Orbit · Scroll: Zoom · Double-click: Reset</div>
</div>
<script>
const META = __META_JSON__;
const PAYLOAD_B64 = "__PAYLOAD_B64__";

function b64ToBytes(b64) {
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

async function decodeVolume() {
  const compressed = b64ToBytes(PAYLOAD_B64);
  if (!META.compressed) return compressed;
  const ds = new DecompressionStream("deflate");
  const stream = new Response(compressed).body.pipeThrough(ds);
  const ab = await new Response(stream).arrayBuffer();
  return new Uint8Array(ab);
}

const VS = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() { v_uv = a_pos; gl_Position = vec4(a_pos, 0.0, 1.0); }`;

const FS = `#version 300 es
precision highp float;
precision highp sampler3D;
uniform sampler3D u_vol;
uniform vec3 u_ro;
uniform mat3 u_cam;
uniform float u_aspect;
uniform vec3 u_box;
uniform vec3 u_slice;
uniform float u_thr;
uniform float u_dens;
uniform int u_mode;
uniform int u_cmap;
uniform int u_invert;
in vec2 v_uv;
out vec4 fragColor;

vec3 cmap(float t) {
  t = clamp(t, 0.0, 1.0);
  if (u_cmap == 0) {
    // EoR divergence colormap: navy -> black/ionized -> amber -> white
    vec3 c0 = vec3(0.10, 0.42, 0.85);
    vec3 c1 = vec3(0.06, 0.08, 0.14);
    vec3 c2 = vec3(0.95, 0.42, 0.12);
    vec3 c3 = vec3(1.00, 0.94, 0.68);
    if (t < 0.42) return mix(c0, c1, t / 0.42);
    if (t < 0.78) return mix(c1, c2, (t - 0.42) / 0.36);
    return mix(c2, c3, (t - 0.78) / 0.22);
  }
  if (u_cmap == 1) {
    return mix(vec3(0.267,0.005,0.329), mix(vec3(0.128,0.567,0.551), vec3(0.993,0.906,0.144), max(0.0,t*2.0-1.0)), min(1.0,t*2.0));
  }
  if (u_cmap == 2) {
    return mix(vec3(0.03,0.02,0.08), mix(vec3(0.73,0.21,0.33), vec3(0.99,0.99,0.64), max(0.0,t*2.0-1.0)), min(1.0,t*2.0));
  }
  if (u_cmap == 3) {
    return mix(vec3(0.05,0.03,0.53), mix(vec3(0.80,0.28,0.47), vec3(0.94,0.98,0.13), max(0.0,t*2.0-1.0)), min(1.0,t*2.0));
  }
  return mix(vec3(0.23,0.30,0.75), mix(vec3(0.87,0.87,0.87), vec3(0.71,0.02,0.15), max(0.0,t*2.0-1.0)), min(1.0,t*2.0));
}

vec2 hitBox(vec3 ro, vec3 rd, vec3 b) {
  vec3 inv = 1.0 / rd;
  vec3 t0 = (-b - ro) * inv;
  vec3 t1 = (b - ro) * inv;
  vec3 tmin = min(t0, t1);
  vec3 tmax = max(t0, t1);
  return vec2(max(max(tmin.x, tmin.y), tmin.z), min(min(tmax.x, tmax.y), tmax.z));
}

void main() {
  vec3 rd = normalize(u_cam * normalize(vec3(v_uv.x * u_aspect, v_uv.y, -1.75)));
  vec2 hit = hitBox(u_ro, rd, u_box);
  if (hit.x > hit.y || hit.y < 0.0) {
    fragColor = vec4(0.035, 0.05, 0.10, 1.0);
    return;
  }
  float t0 = max(hit.x, 0.0);
  float t1 = hit.y;
  const int STEPS = 180;
  float dt = (t1 - t0) / float(STEPS);
  vec3 acc = vec3(0.0);
  float alpha = 0.0;

  if (u_mode != 2) {
    float bestT = 1e9;
    vec3 bestCol = vec3(0.0);
    vec3 slicePos = mix(-u_box, u_box, u_slice);
    vec3 axes[3] = vec3[3](vec3(1,0,0), vec3(0,1,0), vec3(0,0,1));
    for (int a = 0; a < 3; a++) {
      float denom = dot(rd, axes[a]);
      if (abs(denom) > 1e-4) {
        float ts = dot(slicePos - u_ro, axes[a]) / denom;
        if (ts >= t0 && ts <= t1 && ts < bestT) {
          vec3 p = u_ro + ts * rd;
          if (all(lessThanEqual(abs(p), u_box + 1e-4))) {
            vec3 tex = clamp((p + u_box) / (2.0 * u_box), 0.0, 1.0);
            float val = texture(u_vol, tex.zyx).r;
            bestT = ts;
            bestCol = cmap(val);
          }
        }
      }
    }
    if (bestT < 1e8 && u_mode == 1) {
      fragColor = vec4(bestCol, 1.0);
      return;
    }
    t1 = min(t1, bestT);
    dt = max((t1 - t0) / float(STEPS), 1e-4);
    if (u_mode == 0 && bestT < 1e8) {
      // Trace up to closest slice, then composite slice behind accumulated volume
      for (int i = 0; i < STEPS; i++) {
        float t = t0 + (float(i) + 0.5) * dt;
        vec3 p = u_ro + t * rd;
        vec3 tex = (p + u_box) / (2.0 * u_box);
        float val = texture(u_vol, tex.zyx).r;
        float m = u_invert == 1 ? (1.0 - val) : val;
        float w = smoothstep(u_thr, min(1.0, u_thr + 0.35), m) * u_dens * dt * 1.4;
        vec3 c = cmap(val);
        acc += (1.0 - alpha) * c * w;
        alpha += (1.0 - alpha) * w;
        if (alpha > 0.98) break;
      }
      vec3 finalCol = acc + (1.0 - alpha) * bestCol;
      fragColor = vec4(finalCol, 1.0);
      return;
    }
  }

  for (int i = 0; i < STEPS; i++) {
    float t = t0 + (float(i) + 0.5) * dt;
    vec3 p = u_ro + t * rd;
    vec3 tex = (p + u_box) / (2.0 * u_box);
    float val = texture(u_vol, tex.zyx).r;
    float m = u_invert == 1 ? (1.0 - val) : val;
    float w = smoothstep(u_thr, min(1.0, u_thr + 0.35), m) * u_dens * dt * 1.6;
    vec3 c = cmap(val);
    acc += (1.0 - alpha) * c * w;
    alpha += (1.0 - alpha) * w;
    if (alpha > 0.98) break;
  }
  vec3 bg = vec3(0.035, 0.05, 0.10);
  fragColor = vec4(acc + (1.0 - alpha) * bg, 1.0);
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(s));
  }
  return s;
}

(async function main() {
  const volBytes = await decodeVolume();
  const [nx, ny, nz] = META.shape;
  const maxDim = Math.max(META.box[0], META.box[1], META.box[2]);
  const boxHalf = [0.5 * META.box[0] / maxDim, 0.5 * META.box[1] / maxDim, 0.5 * META.box[2] / maxDim];

  document.getElementById("stats").innerHTML =
    `Quantity: <b>${META.quantity}</b><br>` +
    `Shape: ${nx} × ${ny} × ${nz} | Range: [${META.vmin.toFixed(2)}, ${META.vmax.toFixed(2)}]<br>` +
    `Size: ${META.box[0].toFixed(1)} × ${META.box[1].toFixed(1)} × ${META.box[2].toFixed(1)} Mpc`;
  document.getElementById("vminTxt").textContent = META.vmin.toFixed(2);
  document.getElementById("vmaxTxt").textContent = META.vmax.toFixed(2);
  document.getElementById("cmapSel").value = String(META.default_cmap || 0);

  const canvas = document.getElementById("glcanvas");
  const gl = canvas.getContext("webgl2", { preserveDrawingBuffer: true });
  if (!gl) { document.getElementById("hud").textContent = "WebGL2 not supported"; return; }

  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VS));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, FS));
  gl.linkProgram(prog);
  gl.useProgram(prog);

  const vbo = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "a_pos");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_3D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  // Note volume is stored C-order (nx, ny, nz) -> WebGL3D (width=nz, height=ny, depth=nx)
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, nz, ny, nx, 0, gl.RED, gl.UNSIGNED_BYTE, volBytes);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);

  let yaw = 0.65, pitch = 0.42, dist = 1.85, drag = false, px = 0, py = 0;
  function updateLabels() {
    const sx = +document.getElementById("xSlice").value;
    const sy = +document.getElementById("ySlice").value;
    const sz = +document.getElementById("zSlice").value;
    const thr = +document.getElementById("thr").value;
    const dens = +document.getElementById("dens").value;
    document.getElementById("xVal").textContent = `${(sx * META.box[0]).toFixed(1)} Mpc`;
    document.getElementById("yVal").textContent = `${(sy * META.box[1]).toFixed(1)} Mpc`;
    if (META.z_coords && META.z_coords.length >= 2) {
      const zmin = META.z_coords[0], zmax = META.z_coords[1];
      document.getElementById("zVal").textContent = `z ≈ ${(zmin + sz * (zmax - zmin)).toFixed(2)}`;
    } else {
      document.getElementById("zVal").textContent = `${(sz * META.box[2]).toFixed(1)} Mpc`;
    }
    const rawThr = META.vmin + thr * (META.vmax - META.vmin);
    document.getElementById("thrVal").textContent = rawThr.toFixed(2);
    document.getElementById("densVal").textContent = dens.toFixed(2);
    const cm = +document.getElementById("cmapSel").value;
    const grads = [
      "linear-gradient(90deg, #1a6bd9, #0f1424 42%, #f26b1f 78%, #fff0ad)",
      "linear-gradient(90deg, #440154, #21918c, #fde725)",
      "linear-gradient(90deg, #080514, #ba3655, #fcfdbf)",
      "linear-gradient(90deg, #0d0887, #cc4778, #f0f921)",
      "linear-gradient(90deg, #3b4cc0, #dddddd, #b40426)"
    ];
    document.getElementById("cbar").style.background = grads[cm];
  }

  function draw() {
    const r = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    canvas.width = Math.max(1, Math.floor(r.width * dpr));
    canvas.height = Math.max(1, Math.floor(r.height * dpr));
    gl.viewport(0, 0, canvas.width, canvas.height);

    const cp = Math.cos(pitch), sp = Math.sin(pitch), cy = Math.cos(yaw), sy = Math.sin(yaw);
    const ro = [dist * cp * sy, dist * sp, dist * cp * cy];
    const fwd = [-ro[0]/dist, -ro[1]/dist, -ro[2]/dist];
    const right = [cy, 0, -sy];
    const up = [
      right[1]*fwd[2] - right[2]*fwd[1],
      right[2]*fwd[0] - right[0]*fwd[2],
      right[0]*fwd[1] - right[1]*fwd[0]
    ];
    // GLSL mat3 columns are (right, up, -fwd)
    const camMat = [
      right[0], right[1], right[2],
      up[0], up[1], up[2],
      -fwd[0], -fwd[1], -fwd[2]
    ];

    gl.uniform3fv(gl.getUniformLocation(prog, "u_ro"), ro);
    gl.uniformMatrix3fv(gl.getUniformLocation(prog, "u_cam"), false, camMat);
    gl.uniform1f(gl.getUniformLocation(prog, "u_aspect"), canvas.width / canvas.height);
    gl.uniform3fv(gl.getUniformLocation(prog, "u_box"), boxHalf);
    gl.uniform3f(gl.getUniformLocation(prog, "u_slice"),
      +document.getElementById("xSlice").value,
      +document.getElementById("ySlice").value,
      +document.getElementById("zSlice").value
    );
    gl.uniform1f(gl.getUniformLocation(prog, "u_thr"), +document.getElementById("thr").value);
    gl.uniform1f(gl.getUniformLocation(prog, "u_dens"), +document.getElementById("dens").value);
    gl.uniform1i(gl.getUniformLocation(prog, "u_mode"), +document.getElementById("modeSel").value);
    gl.uniform1i(gl.getUniformLocation(prog, "u_cmap"), +document.getElementById("cmapSel").value);
    gl.uniform1i(gl.getUniformLocation(prog, "u_invert"), document.getElementById("invertChk").checked ? 1 : 0);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  }

  document.querySelectorAll("input,select").forEach(el => el.addEventListener("input", () => { updateLabels(); draw(); }));
  canvas.addEventListener("mousedown", e => { drag = true; px = e.clientX; py = e.clientY; });
  window.addEventListener("mouseup", () => { drag = false; });
  window.addEventListener("mousemove", e => {
    if (!drag) return;
    yaw -= (e.clientX - px) * 0.008;
    pitch = Math.max(-1.45, Math.min(1.45, pitch + (e.clientY - py) * 0.008));
    px = e.clientX; py = e.clientY;
    draw();
  });
  canvas.addEventListener("wheel", e => {
    e.preventDefault();
    dist = Math.max(0.7, Math.min(4.5, dist * (1.0 + e.deltaY * 0.001)));
    draw();
  }, { passive: false });
  function resetView() { yaw = 0.65; pitch = 0.42; dist = 1.85; draw(); }
  canvas.addEventListener("dblclick", resetView);
  document.getElementById("resetBtn").addEventListener("click", resetView);
  document.getElementById("snapBtn").addEventListener("click", () => {
    draw();
    const a = document.createElement("a");
    a.download = "21cmcu_3d_view.png";
    a.href = canvas.toDataURL("image/png");
    a.click();
  });
  window.addEventListener("resize", draw);
  updateLabels();
  draw();
})();
</script>
</body>
</html>
"""

_CMAP_IDS = {
    "eor_21cm": 0,
    "viridis": 1,
    "inferno": 2,
    "plasma": 3,
    "coolwarm": 4,
}


def _downsample_3d(arr: np.ndarray, max_xy: int, max_z: int) -> np.ndarray:
    nx, ny, nz = arr.shape
    sx = max(1, int(np.ceil(nx / max_xy)))
    sy = max(1, int(np.ceil(ny / max_xy)))
    sz = max(1, int(np.ceil(nz / max_z)))
    if sx == 1 and sy == 1 and sz == 1:
        return arr
    return arr[::sx, ::sy, ::sz]


def export_volume_3d_html(
    volume: np.ndarray,
    out_path: str | Path,
    *,
    box_length: tuple[float, float, float] | float | None = None,
    z_coords: tuple[float, float] | None = None,
    title: str = "21cmCU Interactive 3D Volume",
    quantity_name: str = "brightness_temp [mK]",
    cmap: Literal["eor_21cm", "viridis", "inferno", "plasma", "coolwarm"] = "eor_21cm",
    vmin: float | None = None,
    vmax: float | None = None,
    max_xy: int = 96,
    max_z: int = 160,
    compress: bool = True,
) -> Path:
    """Export any 3D simulation field to a standalone interactive WebGL2 HTML5 viewer."""
    arr = np.asarray(volume, dtype=np.float32)
    if arr.ndim != 3 or arr.size == 0:
        raise ValueError(f"Expected non-empty 3D volume, got shape {arr.shape}")

    sub = _downsample_3d(arr, max_xy=int(max_xy), max_z=int(max_z))
    finite = sub[np.isfinite(sub)]
    if finite.size == 0:
        lo, hi = 0.0, 1.0
    else:
        lo = float(np.percentile(finite, 1.0)) if vmin is None else float(vmin)
        hi = float(np.percentile(finite, 99.0)) if vmax is None else float(vmax)
        if hi <= lo:
            lo = float(np.min(finite))
            hi = float(np.max(finite))
            if hi <= lo:
                hi = lo + 1.0

    scaled = np.clip((np.nan_to_num(sub, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    quant = np.ascontiguousarray(np.round(scaled * 255.0), dtype=np.uint8)
    raw_bytes = quant.tobytes(order="C")
    payload = zlib.compress(raw_bytes, level=6) if compress else raw_bytes
    b64 = base64.b64encode(payload).decode("ascii")

    if box_length is None:
        box = [float(s) for s in sub.shape]
    elif np.ndim(box_length) == 0:
        b = float(box_length)
        box = [b, b, b]
    else:
        box = [float(x) for x in box_length]

    meta = {
        "shape": [int(s) for s in sub.shape],
        "box": box,
        "vmin": lo,
        "vmax": hi,
        "quantity": str(quantity_name),
        "default_cmap": int(_CMAP_IDS.get(cmap, 0)),
        "compressed": bool(compress),
        "z_coords": [float(z_coords[0]), float(z_coords[1])] if z_coords is not None else None,
    }
    doc = (
        _HTML_TEMPLATE.replace("__TITLE__", html.escape(title))
        .replace("__META_JSON__", json.dumps(meta))
        .replace("__PAYLOAD_B64__", b64)
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")
    return out


def export_lightcone_3d_html(
    lightcone: LightCone,
    out_path: str | Path,
    *,
    quantity: str = "brightness_temp",
    cmap: Literal["eor_21cm", "viridis", "inferno", "plasma", "coolwarm"] | None = None,
    max_xy: int = 96,
    max_z: int = 192,
    title: str | None = None,
) -> Path:
    """Export a ``LightCone`` field to an interactive 3D WebGL2 HTML5 lightcone viewer."""
    if quantity not in lightcone.lightcones:
        avail = ", ".join(sorted(lightcone.lightcones))
        raise KeyError(f"Quantity {quantity!r} not in lightcone.lightcones ({avail})")
    zs = np.asarray(lightcone.lightcone_redshifts, dtype=np.float64)
    eff_cmap = cmap or ("viridis" if "fraction" in quantity else "eor_21cm")
    eff_title = title or f"21cmCU 3D Lightcone — {quantity} (z={zs.min():.2f}–{zs.max():.2f})"
    return export_volume_3d_html(
        lightcone.lightcones[quantity],
        out_path,
        box_length=lightcone.lightcone_dimensions,
        z_coords=(float(zs[0]), float(zs[-1])),
        title=eff_title,
        quantity_name=quantity,
        cmap=eff_cmap,
        max_xy=max_xy,
        max_z=max_z,
    )


def _require_matplotlib():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for 2D plotting helpers; install matplotlib or use export_lightcone_3d_html()."
        ) from exc
    return plt


def plot_lightcone_summary(
    lightcone: LightCone,
    *,
    quantity: str = "brightness_temp",
    slice_idx: int | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
) -> Any:
    """Plot a publication-ready 3-panel lightcone summary (slice, redshift history, chunk 1D PS)."""
    from .analysis import lightcone_chunk_power_spectra

    plt = _require_matplotlib()
    arr = np.asarray(lightcone.lightcones[quantity])
    ix = arr.shape[0] // 2 if slice_idx is None else int(slice_idx)
    zs = np.asarray(lightcone.lightcone_redshifts, dtype=np.float64)
    box_x, _, box_z = lightcone.lightcone_dimensions

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), constrained_layout=True)

    im = axes[0].imshow(
        arr[ix],
        origin="lower",
        aspect="auto",
        extent=[float(zs[0]), float(zs[-1]), 0.0, float(box_x)],
        cmap="RdBu_r" if "brightness" in quantity else "viridis",
    )
    axes[0].set_xlabel("Redshift z")
    axes[0].set_ylabel("Transverse Y [Mpc]")
    axes[0].set_title(f"Lightcone Slice ({quantity})")
    fig.colorbar(im, ax=axes[0], shrink=0.85)

    gq = lightcone.global_quantities or {}
    if gq and lightcone.node_redshifts:
        nz = np.asarray(lightcone.node_redshifts, dtype=np.float64)
        for key, label in (("neutral_fraction", "x_HI"), ("brightness_temp", "T_b [mK]")):
            if key in gq:
                axes[1].plot(nz, np.asarray(gq[key]), marker="o", ms=3, label=label)
        axes[1].legend()
    else:
        axes[1].plot(zs, np.mean(arr, axis=(0, 1)), label=f"<{quantity}>")
        axes[1].legend()
    axes[1].set_xlabel("Redshift z")
    axes[1].set_title("Global History")
    axes[1].grid(alpha=0.25)

    ps_res = lightcone_chunk_power_spectra(
        lightcone, quantity=quantity, n_chunks=min(3, max(1, arr.shape[2] // 16)), n_bins=10
    )
    for z_mid, ch in zip(ps_res["chunk_redshifts"], ps_res["chunks"], strict=True):
        axes[2].loglog(ch["k"], ch["delta_sq"], label=f"z={float(z_mid):.2f}")
    axes[2].set_xlabel(r"$k$ [Mpc$^{-1}$]")
    axes[2].set_ylabel(r"$\Delta^2(k)$ [mK$^2$]")
    axes[2].set_title("Lightcone Chunk Power Spectra")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.25, which="both")

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_coeval_slices(
    coeval: Coeval,
    *,
    quantities: tuple[str, ...] = ("brightness_temp", "neutral_fraction", "density"),
    slice_idx: int | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
) -> Any:
    """Plot side-by-side 2D cross-section slices from a ``Coeval`` simulation box."""
    plt = _require_matplotlib()
    avail = [q for q in quantities if hasattr(coeval, q)]
    if not avail:
        raise ValueError(f"None of {quantities} found on coeval box")

    fig, axes = plt.subplots(1, len(avail), figsize=(4.5 * len(avail), 4.0), constrained_layout=True)
    if len(avail) == 1:
        axes = [axes]

    box_len = float(coeval.simulation_options.BOX_LEN)
    for ax, q in zip(axes, avail, strict=True):
        arr = np.asarray(getattr(coeval, q))
        iz = arr.shape[2] // 2 if slice_idx is None else int(slice_idx)
        cmap = "RdBu_r" if "brightness" in q else "viridis"
        im = ax.imshow(arr[:, :, iz].T, origin="lower", extent=[0, box_len, 0, box_len], cmap=cmap)
        ax.set_title(f"{q} (z={coeval.redshift:.2f})")
        ax.set_xlabel("X [Mpc]")
        ax.set_ylabel("Y [Mpc]")
        fig.colorbar(im, ax=ax, shrink=0.85)

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_mock_comparison(
    clean_tb: np.ndarray,
    mock_tb: np.ndarray,
    box_length: float | tuple[float, float, float],
    redshift: float,
    *,
    instrument: str = "SKA1-Low",
    slice_idx: int | None = None,
    out_path: str | Path | None = None,
    show: bool = False,
) -> Any:
    """Plot clean vs interferometer mock 21cm slices alongside 2D cylindrical $P(k_\\perp, k_\\parallel)$."""
    from .analysis import power_spectrum_2d

    plt = _require_matplotlib()
    clean = np.asarray(clean_tb, dtype=np.float32)
    mock = np.asarray(mock_tb, dtype=np.float32)
    iz = clean.shape[2] // 2 if slice_idx is None else int(slice_idx)
    bx = float(box_length) if np.ndim(box_length) == 0 else float(box_length[0])

    ps2 = power_spectrum_2d(mock, box_length, n_kperp=8, n_kpara=8)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), constrained_layout=True)
    im0 = axes[0].imshow(clean[:, :, iz].T, origin="lower", extent=[0, bx, 0, bx], cmap="RdBu_r")
    axes[0].set_title(f"True Signal (z={redshift:.2f})")
    fig.colorbar(im0, ax=axes[0], shrink=0.85)

    im1 = axes[1].imshow(mock[:, :, iz].T, origin="lower", extent=[0, bx, 0, bx], cmap="RdBu_r")
    axes[1].set_title(f"{instrument} Mock Observation")
    fig.colorbar(im1, ax=axes[1], shrink=0.85)

    im2 = axes[2].imshow(
        np.log10(np.maximum(ps2["pk"].T, 1e-6)),
        origin="lower",
        aspect="auto",
        cmap="inferno",
    )
    axes[2].set_title(r"$\log_{10} P_{\mathrm{mock}}(k_\perp, k_\parallel)$")
    axes[2].set_xlabel(r"$k_\perp$ bin")
    axes[2].set_ylabel(r"$k_\parallel$ bin")
    fig.colorbar(im2, ax=axes[2], shrink=0.85)

    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=180, bbox_inches="tight")
    if show:
        plt.show()
    return fig


__all__ = [
    "export_lightcone_3d_html",
    "export_volume_3d_html",
    "plot_coeval_slices",
    "plot_lightcone_summary",
    "plot_mock_comparison",
]
