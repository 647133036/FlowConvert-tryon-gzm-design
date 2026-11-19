// 唯一发起口：先 POST /api/jobs 拿 jobId，再挂 SSE。同步返回，不 await 结果。
// SSE 用 fetch+ReadableStream（要带 Authorization），不是 EventSource。
import { getAi, setAi, isProtected } from '../types/layerMeta'

export interface SseEvent {
  type: string
  job_id: string
  data: Record<string, unknown>
}

export interface TryonReviewPayload {
  jobId: number
  finalUrl: string
  assetVer: number
  seedNs: string
  bbox?: [number, number, number, number]
  qcFlags?: string[]
}

// SSE 重连: 1s→2s→4s, 上限 30s, 最多 10 次, 带 Last-Event-ID
const RECONNECT_BASE_MS = 1000
const RECONNECT_MAX_MS = 30_000
const RECONNECT_MAX_ATTEMPTS = 10

export async function submitJob(opts: {
  idemKey: string
  personId: string
  garmentSku: string
  category: string
  mode: string
  seed?: number
  segFree?: boolean
  batchId?: string
  costCenter?: string
}): Promise<{ jobId: string; cacheHit: boolean; estCent: number; pos: number }> {
  const resp = await fetch('/api/jobs', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(opts),
  })
  if (resp.status === 402) {
    const body = await resp.json()
    throw new Error(`Budget exceeded: ${body.hint || 'check cost center'}`)
  }
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({}))
    throw new Error(body.detail || `submit failed ${resp.status}`)
  }
  return resp.json()
}

export async function estimateJob(opts: {
  layers: unknown[]
  rows: Record<string, unknown>[]
  policy?: string
  provider?: string
}): Promise<{
  total: number
  hits: number
  run: number
  cent: number
  etaSec: number
  provider: string
  blocked: { row: number; reason: string }[]
}> {
  const resp = await fetch('/api/jobs/estimate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(opts),
  })
  if (!resp.ok) throw new Error(`estimate failed ${resp.status}`)
  return resp.json()
}

export function subscribeEvents(
  jobId: string,
  onEvent: (ev: SseEvent) => void,
  lastEventId?: string,
): AbortController {
  const ctrl = new AbortController()
  let attempt = 0

  async function connect() {
    try {
      const headers: Record<string, string> = {
        'Accept': 'text/event-stream',
      }
      const token = localStorage.getItem('gzm_token')
      if (token) headers['Authorization'] = `Bearer ${token}`
      if (lastEventId) headers['Last-Event-ID'] = lastEventId

      const r = await fetch(`/api/jobs/${jobId}/events`, {
        headers,
        signal: ctrl.signal,
      })
      if (!r.ok || !r.body) {
        throw new Error(`SSE ${r.status}`)
      }

      const reader = r.body.getReader()
      const dec = new TextDecoder()
      let buf = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buf += dec.decode(value, { stream: true })
        const parts = buf.split('\n\n')
        buf = parts.pop() ?? ''
        for (const p of parts) {
          if (!p.trim()) continue
          const lines = p.split('\n')
          let event = 'message'
          let data = ''
          let id = ''
          for (const line of lines) {
            if (line.startsWith('event:')) event = line.slice(6).trim()
            if (line.startsWith('data:')) data += line.slice(5).trim()
            if (line.startsWith('id:')) id = line.slice(3).trim()
          }
          if (id) lastEventId = id
          if (data) {
            try {
              const parsed = JSON.parse(data)
              onEvent({ type: event, job_id: jobId, data: parsed })
            } catch {
              onEvent({ type: event, job_id: jobId, data: { raw: data } })
            }
          }
          if (event === 'ready' || event === 'failed' || event === 'canceled') {
            return
          }
        }
      }
    } catch (e: unknown) {
      if (ctrl.signal.aborted) return
      attempt++
      if (attempt >= RECONNECT_MAX_ATTEMPTS) {
        onEvent({
          type: 'failed',
          job_id: jobId,
          data: { reason: `SSE reconnect exhausted after ${attempt} attempts` },
        })
        return
      }
      const delay = Math.min(RECONNECT_BASE_MS * 2 ** (attempt - 1), RECONNECT_MAX_MS)
      setTimeout(connect, delay)
    }
  }

  void connect()
  return ctrl
}

// 角色保护: 提交前裁掉 price/logo/qr/aigc-stamp 图层
export function filterAiLayers(layers: any[]): any[] {
  return layers.filter((el) => {
    const ai = getAi(el)
    if (!ai) return true
    return !isProtected(ai.role)
  })
}

// 热替换: 只有 applyTryonResult 一个入口改 el.url
export async function applyTryonResult(
  app: any,
  el: any,
  r: TryonReviewPayload,
): Promise<void> {
  const url = `${r.finalUrl}?v=${r.assetVer}`
  app.tree.edit.start('ai-tryon')
  try {
    el.url = url
    el.pixelRatio = 2
    setAi(el, {
      jobId: r.jobId,
      seedNs: r.seedNs,
      bbox: r.bbox,
      postfx: 'banded',
      lockAi: false,
    })
    el.locked = false
  } finally {
    app.tree.edit.end()
    app.tree.requestRender()
  }
}

// 费用确认条: 批量前先 estimate
export function formatEstimate(est: {
  total: number
  hits: number
  run: number
  cent: number
  etaSec: number
  provider: string
}): string {
  const yuan = (est.cent / 100).toFixed(2)
  const min = Math.ceil(est.etaSec / 60)
  return `本次 ${est.total} 张 · ¥${yuan} · 缓存命中 ${est.hits} · 实跑 ${est.run} · 约 ${min} 分钟 · ${est.provider}`
}
