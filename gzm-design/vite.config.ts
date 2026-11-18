import { defineConfig, loadEnv } from 'vite'
import { resolve } from 'path'

export default defineConfig(({ mode }) => {
  const e = loadEnv(mode, process.cwd())
  return {
    server: {
      port: 5173,
      allowedHosts: ['.monkeycode-ai.online'],
      proxy: {
        // 普通 REST
        '/api': {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
          timeout: 120_000,
          proxyTimeout: 120_000,
        },
        // SSE: 必须单独一条，关所有缓冲
        '/api/jobs/:id/events': {
          target: 'http://127.0.0.1:8000',
          changeOrigin: true,
          timeout: 0,
          proxyTimeout: 0,
          configure(proxy) {
            proxy.on('proxyReq', (pr) => {
              pr.setHeader('Cache-Control', 'no-cache')
              pr.setHeader('Connection', 'keep-alive')
              pr.setHeader('X-Accel-Buffering', 'no')
            })
            proxy.on('proxyRes', (res) => {
              res.headers['cache-control'] = 'no-cache, no-transform'
              res.headers['x-accel-buffering'] = 'no'
              res.headers['content-type'] = 'text/event-stream; charset=utf-8'
            })
          },
        },
        // 本地 OSS
        '/tryon': { target: 'http://127.0.0.1:9000', changeOrigin: true },
      },
    },
    preview: { port: 4173, allowedHosts: ['.monkeycode-ai.online'] },
    build: {
      target: 'es2020',
      sourcemap: false,
      chunkSizeWarningLimit: 1500,
      rollupOptions: {
        output: {
          manualChunks: {
            leafer: ['leafer-ui'],
          },
        },
      },
    },
  }
})
