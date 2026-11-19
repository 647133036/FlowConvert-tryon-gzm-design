-- 0002_person_seed_ns.sql
-- 铸脸 person_asset + seed_ns 命名空间 + 四级选人路由支持表

CREATE TABLE IF NOT EXISTS person_asset (
  id              bigserial PRIMARY KEY,
  tenant_id       bigint NOT NULL,
  face_preset_id  text NOT NULL,
  tier            text NOT NULL DEFAULT 'sub' CHECK (tier IN ('hero','sub','test')),
  body_bucket     text NOT NULL,
  lock            jsonb NOT NULL,
  free            jsonb NOT NULL,
  ban             jsonb NOT NULL,
  seed_ns         text NOT NULL,
  seed            bigint NOT NULL,
  gen_model       text NOT NULL,
  gen_ver         text NOT NULL,
  prompt_hash     text NOT NULL,
  dedup_score     float4,
  nsfw_score      float4,
  probe_wm        text,
  lineage         jsonb NOT NULL,
  use_count       int NOT NULL DEFAULT 0,
  last_used_sku   text,
  status          text NOT NULL DEFAULT 'sandbox' CHECK (status IN ('sandbox','prod','frozen')),
  retire_policy   text NOT NULL DEFAULT 'freeze_not_delete',
  created_at      timestamptz NOT NULL DEFAULT now(),
  updated_at      timestamptz NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, face_preset_id, ver_placeholder)  -- placeholder, see below
);
-- face_preset_id includes version suffix (e.g. A01v2); no separate ver col
ALTER TABLE person_asset DROP CONSTRAINT IF EXISTS person_asset_face_preset_id_ver_placeholder_key;
ALTER TABLE person_asset ADD CONSTRAINT uq_person_preset UNIQUE (tenant_id, face_preset_id, gen_ver);
CREATE INDEX IF NOT EXISTS gmt_person_bucket ON person_asset (tenant_id, body_bucket, status);
CREATE INDEX IF NOT EXISTS gmt_person_seed ON person_asset (tenant_id, seed_ns);

CREATE TABLE IF NOT EXISTS seed_ns_registry (
  tenant_id   bigint NOT NULL,
  ns          text NOT NULL,
  engine      text NOT NULL,
  status      text NOT NULL DEFAULT 'active' CHECK (status IN ('active','frozen')),
  created_at  timestamptz NOT NULL DEFAULT now(),
  PRIMARY KEY (tenant_id, ns)
);
-- current_seed_ns(tenant): active ns, newest first
CREATE OR REPLACE FUNCTION current_seed_ns(tid bigint) RETURNS text AS $$
  SELECT ns FROM seed_ns_registry WHERE tenant_id = tid AND status = 'active'
  ORDER BY created_at DESC LIMIT 1;
$$ LANGUAGE SQL;
