<template>
  <div class="ai-panel" v-if="visible" @click.stop>
    <h3>AI 动作</h3>
    <div v-if="!selected" class="hint">选中一个 garment 或 model 图层</div>
    <template v-else>
      <div class="layer-info">
        图层: {{ selected.type }} · role: {{ aiMeta?.role || 'garment' }}
      </div>
      <button
        v-if="canTryOn"
        @click="runAction('tryon')"
        :disabled="busy"
      >
        AI 试衣
      </button>
      <button @click="runAction('swap-bg')" :disabled="busy">
        换背景
      </button>
      <button @click="runAction('upscale')" :disabled="busy">
        超清 2x
      </button>
      <button @click="runAction('inpaint')" :disabled="busy">
        局部重绘
      </button>
      <div v-if="busy" class="progress">
        <span>提交中…</span>
        <div class="bar" :style="{ width: progressPct + '%' }"></div>
      </div>
      <div v-if="estimate" class="estimate">
        <div class="estimate-text">{{ estimateText }}</div>
        <button @click="confirmSubmit" class="confirm">确认提交</button>
        <button @click="cancelEstimate" class="cancel">取消</button>
      </div>
      <div v-if="hint" class="hint" :class="{ warn: hintIsWarn }">{{ hint }}</div>
    </template>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from 'vue'
import { getAi, isProtected, type AiMeta } from '../types/layerMeta'
import {
  estimateJob,
  formatEstimate,
  submitJob,
  subscribeEvents,
} from './useGenerate'

const props = defineProps<{
  selected: any
  visible: boolean
}>()

const busy = ref(false)
const hint = ref('')
const hintIsWarn = ref(false)
const progressPct = ref(0)
const estimate = ref<Record<string, unknown> | null>(null)
const pendingAction = ref('')
const sseCtrl = ref<AbortController | null>(null)

const aiMeta = computed<AiMeta | undefined>(() => {
  if (!props.selected) return undefined
  return getAi(props.selected)
})

const canTryOn = computed(() => {
  const role = aiMeta.value?.role
  if (!role) return false
  return !isProtected(role)
})

const estimateText = computed(() => {
  if (!estimate.value) return ''
  const est = estimate.value as {
    total: number
    hits: number
    run: number
    cent: number
    etaSec: number
    provider: string
  }
  return formatEstimate(est)
})

function cancelEstimate() {
  estimate.value = null
  pendingAction.value = ''
}

async function runAction(action: string) {
  if (!props.selected || busy.value) return
  busy.value = true
  hint.value = ''
  hintIsWarn.value = false
  pendingAction.value = action

  // 提交前拦截: 调 estimate → 弹费用与库存确认条
  try {
    const est = await estimateJob({
      layers: [props.selected],
      rows: [
        {
          sku: aiMeta.value?.sku || '',
          img_top: '',
          img_bottom: '',
        },
      ],
    })
    estimate.value = est as unknown as Record<string, unknown>
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e)
    hint.value = msg
    hintIsWarn.value = true
    busy.value = false
  }
}

async function confirmSubmit() {
  if (!estimate.value || !pendingAction.value) return
  cancelEstimate()

  try {
    const { jobId, cacheHit } = await submitJob({
      idemKey: crypto.randomUUID(),
      personId: aiMeta.value?.personPreset || '',
      garmentSku: aiMeta.value?.sku || '',
      category: aiMeta.value?.category || 'tops',
      mode: 'Balanced',
      segFree: true,
    })

    if (cacheHit) {
      hint.value = '缓存命中，无需等待'
      busy.value = false
      return
    }

    // SSE 订阅: queued→running→preview→postfx→review→ready
    sseCtrl.value = subscribeEvents(jobId, (ev) => {
      switch (ev.type) {
        case 'queued':
          progressPct.value = 10
          hint.value = '排队中…'
          break
        case 'running':
          progressPct.value = 40
          hint.value = '生成中…'
          break
        case 'preview':
          progressPct.value = 70
          hint.value = '预览已出，等待 POSTFX'
          break
        case 'postfx':
          progressPct.value = 90
          hint.value = 'POSTFX 中…'
          break
        case 'ready':
          progressPct.value = 100
          hint.value = '完成'
          busy.value = false
          break
        case 'failed':
          hint.value = `失败: ${JSON.stringify(ev.data)}`
          hintIsWarn.value = true
          busy.value = false
          break
        case 'canceled':
          hint.value = '已取消'
          busy.value = false
          break
        default:
          break
      }
    })
  } catch (e: unknown) {
    const msg = e instanceof Error ? e.message : String(e)
    hint.value = msg
    hintIsWarn.value = true
    busy.value = false
  }
}

defineExpose({ close: () => (visible.value = false) })
const visible = computed({
  get: () => props.visible,
  set: (v) => v,
})
</script>

<style scoped>
.ai-panel {
  position: absolute;
  top: 12px;
  right: 12px;
  width: 260px;
  background: var(--panel, #1a1a2e);
  color: var(--tx, #eee);
  border-radius: 8px;
  padding: 14px;
  box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
  z-index: 100;
}
h3 {
  font-size: 14px;
  margin: 0 0 8px;
  color: var(--ac, #ffd24a);
}
.layer-info {
  font-size: 12px;
  color: var(--tx3, #6b7885);
  margin-bottom: 8px;
}
button {
  display: block;
  width: 100%;
  padding: 7px;
  margin: 4px 0;
  background: var(--panel2, #333);
  color: var(--tx, #fff);
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 13px;
}
button:hover:not(:disabled) {
  background: var(--panel2-hover, #444);
}
button:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}
button.confirm {
  background: var(--ac2, #4dd6a8);
  color: #000;
  font-weight: 600;
}
button.cancel {
  background: transparent;
  border: 1px solid var(--line, #26303c);
}
.hint {
  font-size: 11px;
  color: var(--tx3, #6b7885);
  margin-top: 6px;
}
.hint.warn {
  color: var(--bad, #ff6b6b);
}
.estimate {
  margin-top: 8px;
  padding: 8px;
  background: var(--panel2, #222);
  border-radius: 4px;
}
.estimate-text {
  font-size: 12px;
  color: var(--tx2, #9aa7b6);
  margin-bottom: 6px;
}
.progress {
  margin-top: 8px;
}
.progress .bar {
  height: 3px;
  background: var(--ac, #ffd24a);
  border-radius: 2px;
  transition: width 0.3s;
}
</style>
