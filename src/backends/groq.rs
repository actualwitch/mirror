//! Groq API client implementation for chat functionality.
//!
//! This module provides integration with Groq's LLM models through their API.

use crate::{
    chat::{ChatMessage, ChatProvider, ChatResponse, ChatRole, Tool, Usage},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider, ToolCall,
};
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// Client for interacting with Groq's API.
pub struct Groq {
    pub api_key: String,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system: Option<String>,
    pub timeout_seconds: Option<u64>,
    pub stream: Option<bool>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    client: Client,
}

#[derive(Serialize)]
struct GroqChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Serialize)]
struct GroqChatRequest<'a> {
    model: &'a str,
    messages: Vec<GroqChatMessage<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
}

#[derive(Deserialize, Debug)]
struct GroqChatResponse {
    choices: Vec<GroqChatChoice>,
    usage: Option<GroqUsage>,
}

#[derive(Deserialize, Debug)]
struct GroqUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Deserialize, Debug)]
struct GroqChatChoice {
    message: GroqChatMsg,
}

#[derive(Deserialize, Debug)]
struct GroqChatMsg {
    content: String,
}

impl std::fmt::Display for GroqChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.text().unwrap_or_default())
    }
}

impl ChatResponse for GroqChatResponse {
    fn text(&self) -> Option<String> {
        self.choices.first().and_then(|c| {
            if c.message.content.is_empty() {
                None
            } else {
                Some(c.message.content.clone())
            }
        })
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        todo!()
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.as_ref().map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        })
    }
}

#[allow(clippy::too_many_arguments)]
impl Groq {
    /// Creates a new Groq client with the specified configuration.
    pub fn new(
        api_key: impl Into<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        stream: Option<bool>,
        top_p: Option<f32>,
        top_k: Option<u32>,
    ) -> Self {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        Self {
            api_key: api_key.into(),
            model: model.unwrap_or("llama-3.3-70b-versatile".to_string()),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            stream,
            top_p,
            top_k,
            client: builder.build().expect("Failed to build reqwest Client"),
        }
    }
}

#[async_trait]
impl ChatProvider for Groq {
    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Groq API key".to_string()));
        }

        let mut groq_msgs: Vec<GroqChatMessage> = messages
            .iter()
            .map(|m| GroqChatMessage {
                role: match m.role {
                    ChatRole::User => "user",
                    ChatRole::Assistant => "assistant",
                },
                content: &m.content,
            })
            .collect();

        if let Some(system) = &self.system {
            groq_msgs.insert(
                0,
                GroqChatMessage {
                    role: "system",
                    content: system,
                },
            );
        }

        let body = GroqChatRequest {
            model: &self.model,
            messages: groq_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: self.stream.unwrap_or(false),
            top_p: self.top_p,
            top_k: self.top_k,
        };

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("Groq request payload: {json}");
            }
        }

        let mut request = self
            .client
            .post("https://api.groq.com/openai/v1/chat/completions")
            .header("Content-Type", "application/json")
            .bearer_auth(&self.api_key)
            .json(&body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let resp = request.send().await?;

        log::debug!("Groq HTTP status: {}", resp.status());

        let resp = resp.error_for_status()?;
        let json_resp: GroqChatResponse = resp.json().await?;

        Ok(Box::new(json_resp))
    }

    async fn chat_with_tools(
        &self,
        _messages: &[ChatMessage],
        _tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        todo!()
    }
}

#[async_trait]
impl CompletionProvider for Groq {
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "Groq completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for Groq {
    async fn embed(&self, _text: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        Err(LLMError::ProviderError(
            "Embedding not supported".to_string(),
        ))
    }
}

#[async_trait]
impl SpeechToTextProvider for Groq {
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "Groq does not implement speech to text endpoint yet.".into(),
        ))
    }
}

#[async_trait]
impl TextToSpeechProvider for Groq {}

#[async_trait]
impl ModelsProvider for Groq {}

impl LLMProvider for Groq {}
