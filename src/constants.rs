//! Common constants used across different backends
//!
//! This module provides const strings and values that are shared across
//! different LLM backends to reduce allocations and improve performance.

// Role constants
pub const ROLE_USER: &str = "user";
pub const ROLE_ASSISTANT: &str = "assistant";
pub const ROLE_SYSTEM: &str = "system";
pub const ROLE_TOOL: &str = "tool";
pub const ROLE_DEVELOPER: &str = "developer"; // Used by Cohere

// Message type constants
pub const MESSAGE_TYPE_TEXT: &str = "text";
pub const MESSAGE_TYPE_IMAGE_URL: &str = "image_url";
pub const MESSAGE_TYPE_FUNCTION: &str = "function";

// Default values
pub const DEFAULT_OPENAI_BASE_URL: &str = "https://api.openai.com/v1/";
pub const DEFAULT_COHERE_BASE_URL: &str = "https://api.cohere.ai/compatibility/v1/";
pub const DEFAULT_MISTRAL_BASE_URL: &str = "https://api.mistral.ai/v1/";
pub const DEFAULT_OPENAI_MODEL: &str = "gpt-4.1";
pub const DEFAULT_COHERE_MODEL: &str = "command-light";
pub const DEFAULT_MISTRAL_MODEL: &str = "mistral-small-latest";

// Model prefixes for compatibility checking
pub const REASONING_MODEL_PREFIXES: &[&str] = &[
    "o1", "o1-preview", "o1-mini",
    "o3", "o3-mini",
    "o4", "o4-mini"
];

// Error messages
pub const ERR_NO_RESPONSE_CHOICES: &str = "[No response choices]";
pub const ERR_IMAGE_NOT_IMPLEMENTED: &str = "Image messages not implemented";
pub const ERR_PDF_NOT_IMPLEMENTED: &str = "PDF messages not implemented";

// SSE constants
pub const SSE_DONE_MARKER: &str = "[DONE]";
pub const SSE_DATA_PREFIX: &str = "data: ";