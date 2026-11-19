-- 0001_core_tables.sql
-- Atelier 编排层核心表：job / garment_asset / template_binding / cost_ledger

CREATE TABLE IF NOT EXISTS garment_asset (
  id            bigserial PRIMARY KEY,
  tenant_id     bigint NOT NULL,
  sku_id        text NOT NULL,
  colorway      text NOT NULL DEFAULT 'STD',
  category      text NOT NULL CHECK (category IN ('tops','bottoms','dresses','outerwear')),
  flatlay_url   text NOT NULL,
  sha256        text NOT NULL,
  ver           int NOT NULL DEFAULT 1,
  risk_tags     text[] NOT NULL DEFAULT '{}',
  logo_bbox     jsonb,
  base_color_lab float4[3],
  master_data_at timestamptz,
  created_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, sha256)
);
CREATE INDEX IF NOT EXISTS gmt_risk ON garment_asset USING GIN (risk_tags);
CREATE INDEX IF NOT EXISTS gmt_sku  ON garment_asset (tenant_id, sku_id, colorway, ver DESC);

CREATE TABLE IF NOT EXISTS tryon_job (
  id              bigserial PRIMARY KEY,
  tenant_id       bigint NOT NULL,
  idem_key        text NOT NULL UNIQUE,
  idempotency_key text NOT NULL,
  state           text NOT NULL DEFAULT 'DRAFT',
  attempts        int NOT NULL DEFAULT 0,
  provider_req_id text,
  cost_est_cent   int,
  batch_id        text,
  prio            smallint NOT NULL DEFAULT 2,
  cb_ref          text,
  person_id       text NOT NULL,
  garment_sku     text NOT NULL,
  category        text NOT NULL,
  mode            text NOT NULL DEFAULT 'Balanced',
  seed            bigint NOT NULL,
  seed_ns         text NOT NULL,
  seg_free        boolean NOT NULL DEFAULT true,
  pipeline_ver    text NOT NULL,
  state_changed_at timestamptz NOT NULL DEFAULT now(),
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS gmt_job_state ON tryon_job (tenant_id, state, prio DESC, created_at);
CREATE INDEX IF NOT EXISTS gmt_job_batch ON tryon_job (batch_id) WHERE batch_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS template_binding (
  id           bigserial PRIMARY KEY,
  tenant_id    bigint NOT NULL,
  template_id  text NOT NULL,
  ver          int NOT NULL,
  layer_key    text NOT NULL,
  role         text NOT NULL,
  bind_level   smallint NOT NULL CHECK (bind_level BETWEEN 1 AND 3),
  sku_id       text,
  sku_expr     text,
  pool_key     text,
  lock_ai      bool NOT NULL DEFAULT false,
  visual_hash  text NOT NULL,
  published_at timestamptz,
  created_at   timestamptz NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, template_id, ver, layer_key)
);

CREATE TABLE IF NOT EXISTS cost_ledger (
  id            bigserial PRIMARY KEY,
  tenant_id     bigint NOT NULL,
  job_id        bigint NOT NULL,
  entry_type    text NOT NULL CHECK (entry_type IN ('EST','ACT','HIT','REF','OVR')),
  amount_cent   int NOT NULL,
  provider      text,
  created_at    timestamptz NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, job_id, entry_type, provider)
);
CREATE INDEX IF NOT EXISTS gmt_cost_day ON cost_ledger (tenant_id, created_at DESC);
