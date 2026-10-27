-- Challenge tokens already redeemed, kept until they expire. The primary key is what
-- makes a token single use across restarts and replicas.
CREATE TABLE used_challenges (
    id TEXT PRIMARY KEY,
    expires_at BIGINT NOT NULL
);
CREATE INDEX used_challenges_expiry ON used_challenges (expires_at);
