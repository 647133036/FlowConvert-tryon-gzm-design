# AI 试衣出图台 · 开工设计稿（归档）

- 来源: 用户提供的《电商 AI 试衣出图台 开工设计稿》(2026-11-18), 原文见本目录 `开工设计稿.html`, 勿再粘贴全文, 直接读文件。
- 性质: 静态设计文档(无前端代码可复用)。L1 GzmDesign 前端 fork 与 L2 编排层为后续独立项目。
- 本仓库已落地部分(v0.2.0): Provider 层(`backend/tryon_provider.py`, fashn_local + aidge 双源)、幂等键(§03)、成本记账简化版(est/act/hit + provider 字段, §06)、prio 队列(§07)、402 熔断/429 退避(§05)。
- 未落地部分: §09 六端点契约、三队列独立 worker、Postgres 9 表、GzmDesign 5 处前端改造(14 人日)、合成人台库(§4.5 + R7)。
- 合成人台增量(HTML 版独有, 粘贴稿没有): §4.5 person_asset 退役 valid_until/agreement_url 换 lineage 血缘, seed/seed_ns NOT NULL 服务端强制注入, 合成人台边际成本 ¥0.0001/张(铸一次 ¥0.24 摊 2000+ 次)——见 §11 R7。
