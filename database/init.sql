CREATE TABLE IF NOT EXISTS entity_dictionary (
    id SERIAL PRIMARY KEY,
    prefix VARCHAR(100) NOT NULL,
    definition TEXT NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(prefix, definition)
);

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY,
    filename VARCHAR(500) NOT NULL,
    original_file_path VARCHAR(500) NOT NULL,
    extracted_text TEXT,
    status VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS extracted_entities (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    entity_type VARCHAR(100) NOT NULL,
    entity_value TEXT NOT NULL,
    start_position INTEGER NOT NULL,
    end_position INTEGER NOT NULL,
    confidence FLOAT,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS entity_pairs (
    id UUID PRIMARY KEY,
    document_id UUID REFERENCES documents(id) ON DELETE CASCADE,
    act_entity_id UUID REFERENCES extracted_entities(id),
    money_entity_id UUID REFERENCES extracted_entities(id),
    pair_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS masked_documents (
    id UUID PRIMARY KEY,
    original_document_id UUID REFERENCES documents(id),
    mask_strategy VARCHAR(50) NOT NULL,
    output_format VARCHAR(10) NOT NULL,
    masked_file_path VARCHAR(500) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_entities_document ON extracted_entities(document_id);
CREATE INDEX idx_entities_type ON extracted_entities(entity_type);
CREATE INDEX idx_pairs_document ON entity_pairs(document_id);
