//! Server-Sent Events (SSE) parsing utilities
//!
//! This module provides common SSE parsing functionality used by streaming
//! endpoints across different LLM providers.

use crate::{constants::*, error::LLMError};
use serde::de::DeserializeOwned;

/// A trait for extracting content from SSE response chunks
pub trait SSEContentExtractor {
    /// Extract the content string from the parsed response
    fn extract_content(&self) -> Option<&str>;
}

/// Generic SSE chunk parser that can work with different response types
///
/// # Arguments
///
/// * `chunk` - The raw SSE chunk text
/// * `parse_fn` - Function to parse the JSON data into the response type
///
/// # Type Parameters
///
/// * `T` - The response type that implements SSEContentExtractor
///
/// # Returns
///
/// * `Ok(Some(String))` - Collected content if found
/// * `Ok(None)` - If chunk should be skipped (e.g., ping, done signal)
/// * `Err(LLMError)` - If parsing fails
pub fn parse_sse_chunk<T, F>(chunk: &str, parse_fn: F) -> Result<Option<String>, LLMError>
where
    T: SSEContentExtractor,
    F: Fn(&str) -> Result<T, serde_json::Error>,
{
    let mut collected_content = String::new();

    for line in chunk.lines() {
        let line = line.trim();

        if let Some(data) = line.strip_prefix(SSE_DATA_PREFIX) {
            if data == SSE_DONE_MARKER {
                return if collected_content.is_empty() {
                    Ok(None)
                } else {
                    Ok(Some(collected_content))
                };
            }

            match parse_fn(data) {
                Ok(response) => {
                    if let Some(content) = response.extract_content() {
                        collected_content.push_str(content);
                    }
                }
                Err(_) => continue,
            }
        }
    }

    if collected_content.is_empty() {
        Ok(None)
    } else {
        Ok(Some(collected_content))
    }
}

/// Helper function for parsing standard SSE chunks with serde
///
/// This is a convenience function that automatically deserializes JSON
/// using serde and the SSEContentExtractor trait.
pub fn parse_sse_chunk_json<T>(chunk: &str) -> Result<Option<String>, LLMError>
where
    T: DeserializeOwned + SSEContentExtractor,
{
    parse_sse_chunk(chunk, |data| serde_json::from_str::<T>(data))
}