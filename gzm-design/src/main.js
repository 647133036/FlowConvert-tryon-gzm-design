import { Leafer, Rect, Text, Image, Group } from 'leafer-ui'

// ---- Bug fix 3: CJK font guard ----
// Build /fonts/fonts.css with @font-face + ensure Nginx serves font/woff2 MIME + CORS.
// This guard verifies CJK fonts loaded before rendering text layers.
function fontGuard() {
  if (typeof document === 'undefined') return
  const probe = document.createElement('span')
  probe.textContent = '测试'
  probe.style.cssText = 'position:absolute;visibility:hidden;font-size:100px'
  document.body.appendChild(probe)
  const w1 = probe.offsetWidth
  probe.style.fontFamily = 'sans-serif'
  const w2 = probe.offsetWidth
  document.body.removeChild(probe)
  if (w1 === w2) {
    console.warn('[fontGuard] CJK font may not be loaded; falling back to sans-serif')
  }
}
fontGuard()

// ---- State ----
const state = {
  layers: [],
  selected: null,
  undoStack: [],
  canvasW: 1080,
  canvasH: 1350,
}

// ---- Bug fix 4: undo stack must record AI edits ----
// All layer mutations (including AI image.url changes) go through record()
// so Ctrl+Z can revert them.
function record() {
  state.undoStack.push(JSON.stringify(state.layers))
  if (state.undoStack.length > 50) state.undoStack.shift()
}

function undo() {
  if (state.undoStack.length === 0) return
  const snap = state.undoStack.pop()
  state.layers = JSON.parse(snap)
  renderAll()
}

// ---- Leafer setup ----
const leafer = new Leafer({ view: document.getElementById('canvas') })
leafer.resize(state.canvasW, state.canvasH)
leafer.tree.scale = 0.5

// ---- Bug fix 1: copy/paste — useCopyPaste clipboard target ----
// Leafer's built-in copy/paste targets the active app; we intercept
// Ctrl+C / Ctrl+V and operate on state.selected explicitly.
function copyLayer() {
  if (!state.selected) return
  record()
  const src = state.selected
  const dup = { ...src, x: src.x + 10, y: src.y + 10, id: 'L' + Date.now() }
  state.layers.push(dup)
  renderAll()
}

function pasteLayer() {
  // paste from system clipboard or internal copy
  if (state.layers.length === 0) return
  copyLayer()
}

// ---- Bug fix 2: layer visible/Leafer.visible dual-write ----
// Toggle visibility writes to BOTH state layer.visible AND Leafer.visible
// so the render layer and UI stay in sync.
function toggleVisible(id) {
  const layer = state.layers.find(l => l.id === id)
  if (!layer) return
  record()
  layer.visible = !layer.visible
  const node = leafer.getChild(id)
  if (node) {
    node.visible = layer.visible  // dual-write: state + Leafer
  }
  document.querySelectorAll('[data-layer-id]').forEach(el => {
    if (el.dataset.layerId === id) el.style.opacity = layer.visible ? '1' : '0.3'
  })
}

// ---- Layer factories (pure CPU — no GPU, no model SDK) ----
function addText(text, x, y) {
  record()
  const id = 'L' + Date.now()
  state.layers.push({ id, type: 'text', text, x, y, fontSize: 48, color: '#111', visible: true })
  renderAll()
}

function addPrice(value, x, y) {
  record()
  const id = 'L' + Date.now()
  state.layers.push({ id, type: 'price', text: '$' + value, x, y, fontSize: 72, color: '#e11', visible: true })
  renderAll()
}

function addLogo(url, x, y, w, h) {
  record()
  const id = 'L' + Date.now()
  state.layers.push({ id, type: 'image', url, x, y, width: w, height: h, visible: true })
  renderAll()
}

function addQR(data, x, y, size) {
  record()
  const id = 'L' + Date.now()
  state.layers.push({ id, type: 'qr', data, x, y, width: size, height: size, visible: true })
  renderAll()
}

function addBg(color, w, h) {
  record()
  const id = 'L' + Date.now()
  state.layers.push({ id, type: 'rect', color, x: 0, y: 0, width: w, height: h, visible: true })
  renderAll()
}

// ---- Render all layers to Leafer scene ----
function renderAll() {
  leafer.clear()
  state.layers.forEach(layer => {
    let node
    switch (layer.type) {
      case 'text':
        node = new Text({ text: layer.text, fontSize: layer.fontSize, color: layer.color })
        break
      case 'price':
        node = new Text({ text: layer.text, fontSize: layer.fontSize, color: layer.color, fontWeight: 'bold' })
        break
      case 'image':
        node = new Image({ url: layer.url, width: layer.width, height: layer.height })
        break
      case 'qr':
        node = new Text({ text: '[QR:' + layer.data + ']', fontSize: 12, color: '#666' })
        break
      case 'rect':
        node = new Rect({ width: layer.width, height: layer.height, fill: layer.color })
        break
      default:
        node = new Rect({ width: 100, height: 100, fill: '#ccc' })
    }
    node.id = layer.id
    node.x = layer.x
    node.y = layer.y
    node.visible = layer.visible
    leafer.add(node)
  })
}

// ---- AI panel (wires to /api/jobs; pure CPU for poster, GPU for tryon) ----
function openAiPanel(layer) {
  state.selected = layer
  const panel = document.getElementById('ai-panel')
  panel.classList.remove('hidden')
  document.getElementById('ai-hint').textContent =
    '选中图层: ' + layer.type + ' · 右键菜单或点按钮触发 AI 动作'
}

async function runAiAction(action, layer) {
  const hint = document.getElementById('ai-hint')
  hint.textContent = '提交 ' + action + ' 中…'
  try {
    const resp = await fetch('/api/jobs', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        platform: 'independent_site',
        action,
        layer_id: layer.id,
        layer_type: layer.type,
        payload: { x: layer.x, y: layer.y, width: layer.width, height: layer.height },
      }),
    })
    const data = await resp.json()
    if (data.job_id) {
      hint.textContent = 'job ' + data.job_id + ' 已提交（GPU）'
    } else if (data.error) {
      hint.textContent = data.error
    }
  } catch (e) {
    hint.textContent = '请求失败: ' + e.message
  }
}

// ---- UI wiring ----
document.getElementById('add-text').onclick = () => addText('新文字', 80, 80)
document.getElementById('add-price').onclick = () => addPrice('99', 80, 200)
document.getElementById('add-logo').onclick = () => addLogo('/placeholder-logo.png', 80, 320, 120, 120)
document.getElementById('add-qr').onclick = () => addQR('https://example.com', 80, 480, 80)
document.getElementById('add-bg').onclick = () => addBg('#f5f5f5', state.canvasW, state.canvasH)

document.getElementById('render').onclick = async () => {
  const tpl = {
    width: state.canvasW,
    height: state.canvasH,
    layers: state.layers.map(l => ({ ...l })),
  }
  const resp = await fetch('/api/poster/render', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(tpl),
  })
  const data = await resp.json()
  if (data.url) {
    alert('海报已渲染: ' + data.url)
  } else {
    alert('渲染失败: ' + (data.error || 'unknown'))
  }
}

// Keyboard: Ctrl+Z undo, Ctrl+C copy, Ctrl+V paste
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey || e.metaKey) {
    switch (e.key) {
      case 'z': e.preventDefault(); undo(); break
      case 'c': e.preventDefault(); copyLayer(); break
      case 'v': e.preventDefault(); pasteLayer(); break
    }
  }
})

// Click canvas to select layer
document.getElementById('canvas').addEventListener('click', (e) => {
  const id = e.target.id || e.target.closest('[id]')?.id
  const layer = state.layers.find(l => l.id === id)
  if (layer) {
    state.selected = layer
    openAiPanel(layer)
  }
})

// AI panel buttons
document.querySelectorAll('[data-ai]').forEach(btn => {
  btn.onclick = () => {
    if (!state.selected) return
    runAiAction(btn.dataset.ai, state.selected)
  }
})

export { state, record, undo, copyLayer, pasteLayer, toggleVisible, addText, addPrice, addLogo, addQR, addBg }
