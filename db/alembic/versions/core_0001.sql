CREATE SCHEMA IF NOT EXISTS agent;
REVOKE ALL ON SCHEMA agent FROM PUBLIC;
CREATE TABLE agent.schema_version (version text PRIMARY KEY CHECK (version = 'agent_core_0001'));
INSERT INTO agent.schema_version VALUES ('agent_core_0001');
CREATE TABLE agent.conversations (
 id uuid PRIMARY KEY, title text NOT NULL, created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 last_success_run_id uuid
);
CREATE TABLE agent.runs (
 id uuid PRIMARY KEY,
 task text NOT NULL,
 conversation_id uuid REFERENCES agent.conversations(id),
 turn_no integer,
 previous_run_id uuid REFERENCES agent.runs(id),
 idempotency_scope text NOT NULL,
 idempotency_key text,
 status text NOT NULL CHECK (status IN ('queued','running','cancelling','completed','failed','cancelled')),
 created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 queue_deadline timestamptz NOT NULL,
 started_at timestamptz, deadline_at timestamptz, finished_at timestamptz,
 worker_id uuid, lease_token uuid, lease_until timestamptz,
 answer text, error_code text, final_checkpoint jsonb,
 UNIQUE (idempotency_scope,idempotency_key), UNIQUE (conversation_id,turn_no),
 CHECK ((conversation_id IS NULL) = (turn_no IS NULL)),
 CHECK ((status = 'completed') = (answer IS NOT NULL AND final_checkpoint IS NOT NULL))
);
ALTER TABLE agent.conversations ADD FOREIGN KEY (last_success_run_id) REFERENCES agent.runs(id);
CREATE UNIQUE INDEX one_active_conversation_run ON agent.runs(conversation_id)
 WHERE conversation_id IS NOT NULL AND status IN ('queued','running','cancelling');
CREATE INDEX run_queue ON agent.runs(created_at,id) WHERE status='queued';
CREATE TABLE agent.events (
 id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
 run_id uuid NOT NULL REFERENCES agent.runs(id),
 status text NOT NULL, error_code text,
 created_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE TABLE agent.workers (
 id uuid PRIMARY KEY, started_at timestamptz NOT NULL DEFAULT clock_timestamp(),
 heartbeat_at timestamptz NOT NULL DEFAULT clock_timestamp(), stopped_at timestamptz
);
