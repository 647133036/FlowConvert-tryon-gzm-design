// render/server.ts — @leafer-ui/node + sharp + GlobalFonts
// L4 出图层: 2x PNG/WebP + AIGC 双标识 + 热区 JSON 侧车

import { Leafer, Image, Text, Rect } from '@leafer-ui/node'
import { createCanvas } from '@napi-rs/canvas'
import { registerFonts } from '@napi-rs/canvas'
import sharp from 'sharp'
import fs from 'node:fs'
import path from 'node:path'
import http from 'node:http'

const PORT = parseInt(process.env.RENDERER_URL?.match(/:(\d+)$/)?.[1] || '3001')
const FONTS_DIR = process.env.RENDER_FONTS_DIR || './public/fonts'
const PIXEL_RATIO = parseInt(process.env.RENDER_PIXEL_RATIO || '2')

// 字体注册: 渲染机同款字体 (不然预览思源、导出宋体)
registerFonts({
  files: fs.readdirSync(FONTS_DIR).filter(f => f.endsWith('.woff2')).map(f => path.join(FONTS_DIR, f)),
})

interface RenderRequest {
  width: number
  height: number
  layers: Array<{
    id: string
    type: string
    x: number
    y: number
    width?: number
    height?: number
    text?: string
    fontSize?: number
    color?: string
    fill?: string
    url?: string
    data?: string
    visible?: boolean
    role?: string
  }>
  outputs: Array<{ board?: string; fmt: string; name?: string; q?: number }>
  aigcMark: { implicit: boolean; explicit?: string }
  hotspot: boolean
}

async function renderPoster(req: RenderRequest): Promise<{ png: Buffer; webp?: Buffer; hotspots?: object }> {
  const W = req.width * PIXEL_RATIO
  const H = req.height * PIXEL_RATIO

  // Build Leafer scene
  const leafer = new Leafer({
    width: W,
    height: H,
    view: createCanvas(W, H) as any,
  })

  for (const layer of req.layers) {
    if (!layer.visible) continue
    const lx = layer.x * PIXEL_RATIO
    const ly = layer.y * PIXEL_RATIO

    switch (layer.type) {
      case 'rect': {
        const r = new Rect({ width: (layer.width || W) * PIXEL_RATIO, height: (layer.height || H) * PIXEL_RATIO, fill: layer.fill || layer.color || '#fff' })
        r.x = lx; r.y = ly
        leafer.add(r)
        break
      }
      case 'text':
      case 'price': {
        const t = new Text({
          text: layer.text || '',
          fontSize: (layer.fontSize || 48) * PIXEL_RATIO,
          color: layer.color || '#000',
          fontFamily: 'SourceHanSansSC-sub',
          fontWeight: layer.type === 'price' ? 'bold' : 'normal',
        })
        t.x = lx; t.y = ly
        leafer.add(t)
        break
      }
      case 'image':
      case 'logo': {
        const w = (layer.width || 100) * PIXEL_RATIO
        const h = (layer.height || 100) * PIXEL_RATIO
        if (layer.url) {
          const img = new Image({ url: layer.url, width: w, height: h })
          img.x = lx; img.y = ly
          leafer.add(img)
        }
        break
      }
      case 'qr': {
        const s = (layer.width || 80) * PIXEL_RATIO
        const q = new Text({ text: `[QR:${layer.data || ''}]`, fontSize: 12 * PIXEL_RATIO, color: '#666' })
        q.x = lx; q.y = ly
        leafer.add(q)
        break
      }
    }
  }

  // AIGC 双标识
  if (req.aigcMark?.explicit) {
    const badge = new Text({
      text: req.aigcMark.explicit,
      fontSize: 14 * PIXEL_RATIO,
      color: '#666',
      fontFamily: 'SourceHanSansSC-sub',
    })
    badge.x = W - 200 * PIXEL_RATIO
    badge.y = H - 30 * PIXEL_RATIO
    leafer.add(badge)
  }

  // Render to PNG
  const canvas = (leafer as any).view
  const pngBuf: Buffer = await new Promise((resolve, reject) => {
    const out = sharp(Buffer.from(canvas.toBuffer('image/png')))
    out.toBuffer({ resolve, reject })
  })

  let webpBuf: Buffer | undefined
  const fmt = req.outputs?.find(o => o.fmt === 'webp')?.fmt
  if (fmt === 'webp') {
    const q = req.outputs?.find(o => o.fmt === 'webp')?.q || 86
    webpBuf = await sharp(pngBuf).webp({ quality: q }).toBuffer()
  }

  // Hotspots (PDP clickable areas)
  let hotspots: object | undefined
  if (req.hotspot) {
    hotspots = {
      hotspots: req.layers
        .filter(l => l.role === 'garment' || l.role === 'model')
        .map(l => ({
          layer_id: l.id,
          role: l.role,
          x: l.x,
          y: l.y,
          w: l.width || 0,
          h: l.height || 0,
        })),
    }
  }

  return { png: pngBuf, webp: webpBuf, hotspots }
}

// HTTP server
const server = http.createServer(async (req, res) => {
  if (req.method === 'POST' && req.url === '/render') {
    let body = ''
    req.on('data', (chunk: Buffer) => { body += chunk.toString() })
    req.on('end', async () => {
      try {
        const parsed = JSON.parse(body) as RenderRequest
        const { png, webp, hotspots } = await renderPoster(parsed)
        res.setHeader('Content-Type', 'application/json')
        res.end(JSON.stringify({
          png_b64: png.toString('base64'),
          webp_b64: webp?.toString('base64'),
          hotspots,
          size: { w: parsed.width, h: parsed.height, scale: PIXEL_RATIO },
        }))
      } catch (e: unknown) {
        res.statusCode = 500
        res.end(JSON.stringify({ error: String(e) }))
      }
    })
  } else if (req.url === '/healthz') {
    res.setHeader('Content-Type', 'application/json')
    res.end(JSON.stringify({ ok: true, fonts: fs.readdirSync(FONTS_DIR).length }))
  } else {
    res.statusCode = 404
    res.end('not found')
  }
})

server.listen(PORT, '127.0.0.1', () => {
  console.log(`[render] listening on 127.0.0.1:${PORT}, fonts=${FONTS_DIR}`)
})
