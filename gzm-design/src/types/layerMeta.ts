// L1 ↔ L2 唯一契约：图层 aiMeta 协议。
// 挂 Symbol.for('gzm.aiMeta')，不污染 Leafer 可枚举序列化。

export type LayerRole =
  | 'model'
  | 'garment'
  | 'price'
  | 'logo'
  | 'qr'
  | 'aigc-stamp'

export interface AiMeta {
  role: LayerRole
  sku?: string
  personPreset?: string
  category?: 'tops' | 'bottoms' | 'one-pieces'
  garmentPhotoType?: 'model' | 'flat-lay'
  lockAi?: boolean
  seedNs?: string
  jobId?: number
  bbox?: [number, number, number, number]
  postfx?: 'none' | 'anchor' | 'banded'
}

export const AI_KEY = Symbol.for('gzm.aiMeta')

export function getAi(el: any): AiMeta | undefined {
  return el?.[AI_KEY]
}

export function setAi(el: any, m: Partial<AiMeta>): void {
  el[AI_KEY] = { ...(el[AI_KEY] ?? { role: 'garment' as LayerRole }), ...m }
}

// 导出 JSON 时把 aiMeta 平铺进 userData，模板复用靠它
export function serializeAi(el: any): Record<string, unknown> {
  const ai = getAi(el)
  return ai ? { aiMeta: ai } : {}
}

export function deserializeAi(el: any, d: any): void {
  if (d?.aiMeta) setAi(el, d.aiMeta)
}

// 角色保护：这些角色永不进 AI 输入，提交时前端裁掉
export const PROTECTED_ROLES: ReadonlySet<LayerRole> = new Set([
  'price',
  'logo',
  'qr',
  'aigc-stamp',
])

export function isProtected(role: LayerRole): boolean {
  return PROTECTED_ROLES.has(role)
}
