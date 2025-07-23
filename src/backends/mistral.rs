//! Mistral API client implementation for chat and completion functionality.
//!
//! This module provides integration with Mistral's models through their API.

use crate::constants::*;
use std::time::Duration;

#[cfg(feature = "mistral")]
use crate::{
    builder::LLMBackend,
    chat::Tool,
    chat::{ChatMessage, ChatProvider, ChatRole, MessageType, StructuredOutputFormat},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::{ModelListRawEntry, ModelListRequest, ModelListResponse, ModelsProvider},
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider,
};
use crate::{
    chat::{ChatResponse, ToolChoice, Usage},
    FunctionCall, ToolCall,
};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use either::*;
use futures::stream::Stream;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Client for interacting with Mistral's API.
///
/// Provides methods for chat and completion requests using Mistral's models.
pub struct Mistral {
    pub api_key: String,
    pub base_url: Url,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system: Option<String>,
    pub timeout_seconds: Option<u64>,
    pub stream: Option<bool>,
    pub top_p: Option<f32>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    /// Embedding parameters
    pub embedding_encoding_format: Option<String>,
    pub embedding_dimensions: Option<u32>,
    /// JSON schema for structured output
    pub json_schema: Option<StructuredOutputFormat>,
    client: Client,
}

/// Individual message in a Mistral chat conversation.
#[derive(Serialize, Debug)]
struct MistralChatMessage {
    #[allow(dead_code)]
    role: String,
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "either::serde_untagged_optional"
    )]
    content: Option<Either<Vec<MessageContent>, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<MistralFunctionCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Debug)]
struct MistralFunctionPayload {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug)]
struct MistralFunctionCall {
    id: String,
    #[serde(rename = "type")]
    content_type: String,
    function: MistralFunctionPayload,
}

#[derive(Serialize, Debug)]
struct MessageContent {
    #[serde(rename = "type", skip_serializing_if = "Option::is_none")]
    message_type: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    text: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_url: Option<ImageUrlContent>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "tool_call_id")]
    tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "content")]
    tool_output: Option<String>,
}

/// Individual image message in a Mistral chat conversation.
#[derive(Serialize, Debug)]
struct ImageUrlContent {
    url: String,
}

#[derive(Serialize)]
struct MistralEmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
}

/// Request payload for Mistral's chat API endpoint.
#[derive(Serialize, Debug)]
struct MistralChatRequest {
    model: String,
    messages: Vec<MistralChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<MistralResponseFormat>,
}

/// Response from Mistral's chat API endpoint.
#[derive(Deserialize, Debug)]
struct MistralChatResponse {
    choices: Vec<MistralChatChoice>,
    usage: Option<MistralUsage>,
}

/// Usage information from Mistral's API response
#[derive(Deserialize, Debug)]
struct MistralUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// Individual choice within a Mistral chat API response.
#[derive(Deserialize, Debug)]
struct MistralChatChoice {
    message: MistralChatMsg,
}

/// Message content within a Mistral chat API response.
#[derive(Deserialize, Debug)]
struct MistralChatMsg {
    #[allow(dead_code)]
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<MistralToolCall>>,
}

/// Mistral-specific tool call structure for deserialization
#[derive(Deserialize, Debug)]
struct MistralToolCall {
    id: String,
    #[serde(rename = "type", default = "default_tool_type")]
    call_type: Option<String>,
    function: MistralToolFunction,
}

/// Mistral-specific function structure for deserialization
#[derive(Deserialize, Debug)]
struct MistralToolFunction {
    name: String,
    arguments: String,
}

fn default_tool_type() -> Option<String> {
    Some("function".to_string())
}

#[derive(Deserialize, Debug)]
struct MistralEmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct MistralEmbeddingResponse {
    data: Vec<MistralEmbeddingData>,
}

/// Response from Mistral's streaming chat API endpoint.
#[derive(Deserialize, Debug)]
struct MistralChatStreamResponse {
    choices: Vec<MistralChatStreamChoice>,
}

/// Individual choice within a Mistral streaming chat API response.
#[derive(Deserialize, Debug)]
struct MistralChatStreamChoice {
    delta: MistralChatStreamDelta,
}

/// Delta content within a Mistral streaming chat API response.
#[derive(Deserialize, Debug)]
struct MistralChatStreamDelta {
    content: Option<String>,
}

impl crate::sse::SSEContentExtractor for MistralChatStreamResponse {
    fn extract_content(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.delta.content.as_deref())
    }
}

/// An object specifying the format that the model must output.
#[derive(Deserialize, Debug, Serialize)]
enum MistralResponseType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_object")]
    JsonObject,
}

#[derive(Deserialize, Debug, Serialize)]
struct MistralResponseFormat {
    #[serde(rename = "type")]
    response_type: MistralResponseType,
}

impl From<StructuredOutputFormat> for MistralResponseFormat {
    fn from(_structured_response_format: StructuredOutputFormat) -> Self {
        // Mistral currently only supports json_object format
        MistralResponseFormat {
            response_type: MistralResponseType::JsonObject,
        }
    }
}

impl ChatResponse for MistralChatResponse {
    fn text(&self) -> Option<String> {
        self.choices.first().and_then(|c| c.message.content.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.choices
            .first()
            .and_then(|c| c.message.tool_calls.as_ref())
            .map(|mistral_calls| {
                mistral_calls
                    .iter()
                    .map(|tc| ToolCall {
                        id: tc.id.clone(),
                        call_type: tc.call_type.clone().unwrap_or_else(|| "function".to_string()),
                        function: FunctionCall {
                            name: tc.function.name.clone(),
                            arguments: tc.function.arguments.clone(),
                        },
                    })
                    .collect()
            })
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.as_ref().map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        })
    }
}

impl std::fmt::Display for MistralChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let first_choice = match self.choices.first() {
            Some(choice) => choice,
            None => return write!(f, "{}", ERR_NO_RESPONSE_CHOICES),
        };
        
        // Convert Mistral tool calls to generic format for display
        let converted_tool_calls = first_choice.message.tool_calls.as_ref().map(|calls| {
            calls
                .iter()
                .map(|tc| ToolCall {
                    id: tc.id.clone(),
                    call_type: tc.call_type.clone().unwrap_or_else(|| "function".to_string()),
                    function: FunctionCall {
                        name: tc.function.name.clone(),
                        arguments: tc.function.arguments.clone(),
                    },
                })
                .collect::<Vec<_>>()
        });
        
        match (
            &first_choice.message.content,
            &converted_tool_calls,
        ) {
            (Some(content), Some(tool_calls)) => {
                for tool_call in tool_calls {
                    write!(f, "{tool_call}")?;
                }
                write!(f, "{content}")
            }
            (Some(content), None) => write!(f, "{content}"),
            (None, Some(tool_calls)) => {
                for tool_call in tool_calls {
                    write!(f, "{tool_call}")?;
                }
                Ok(())
            }
            (None, None) => write!(f, ""),
        }
    }
}

impl Mistral {
    /// Creates a new Mistral client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Mistral API key
    /// * `model` - Model to use (defaults to "mistral-small-latest")
    /// * `max_tokens` - Maximum tokens to generate
    /// * `temperature` - Sampling temperature
    /// * `timeout_seconds` - Request timeout in seconds
    /// * `system` - System prompt
    /// * `stream` - Whether to stream responses
    /// * `top_p` - Top-p sampling parameter
    /// * `embedding_encoding_format` - Format for embedding outputs
    /// * `embedding_dimensions` - Dimensions for embedding vectors
    /// * `tools` - Function tools that the model can use
    /// * `tool_choice` - Determines how the model uses tools
    /// * `json_schema` - JSON schema for structured output
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: impl Into<String>,
        base_url: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        temperature: Option<f32>,
        timeout_seconds: Option<u64>,
        system: Option<String>,
        stream: Option<bool>,
        top_p: Option<f32>,
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Self, LLMError> {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        
        let base_url_str = base_url.unwrap_or_else(|| DEFAULT_MISTRAL_BASE_URL.to_owned());
        let base_url = Url::parse(&base_url_str)
            .map_err(|e| LLMError::InvalidRequest(format!("Invalid base URL '{}': {}", base_url_str, e)))?;
        
        let client = builder.build()
            .map_err(|e| LLMError::InvalidRequest(format!("Failed to build HTTP client: {}", e)))?;
        
        Ok(Self {
            api_key: api_key.into(),
            base_url,
            model: model.unwrap_or_else(|| DEFAULT_MISTRAL_MODEL.to_string()),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            stream,
            top_p,
            tools,
            tool_choice,
            embedding_encoding_format,
            embedding_dimensions,
            client,
            json_schema,
        })
    }
}

#[async_trait]
impl ChatProvider for Mistral {
    /// Sends a chat request to Mistral's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    /// * `tools` - Optional slice of tools to use in the chat
    /// # Returns
    ///
    /// The model's response text or an error
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Mistral API key".to_string()));
        }

        // Clone the messages to have an owned mutable vector.
        let messages = messages.to_vec();

        let mut mistral_msgs: Vec<MistralChatMessage> = vec![];

        for msg in messages {
            if let MessageType::ToolResult(results) = &msg.message_type {
                for result in results {
                    mistral_msgs.push(
                        MistralChatMessage {
                            role: ROLE_TOOL.to_string(),
                            tool_call_id: Some(result.id.clone()),
                            tool_calls: None,
                            content: Some(Right(result.function.arguments.clone())),
                        },
                    );
                }
            } else {
                mistral_msgs.push(chat_message_to_api_message(msg))
            }
        }

        if let Some(system) = &self.system {
            mistral_msgs.insert(
                0,
                MistralChatMessage {
                    role: ROLE_SYSTEM.to_string(),
                    content: Some(Left(vec![MessageContent {
                        message_type: Some(MESSAGE_TYPE_TEXT.to_string()),
                        text: Some(system.clone()),
                        image_url: None,
                        tool_call_id: None,
                        tool_output: None,
                    }])),
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }

        let response_format: Option<MistralResponseFormat> =
            self.json_schema.clone().map(|s| s.into());

        let request_tools = tools.map(|t| t.to_vec()).or_else(|| self.tools.clone());

        let request_tool_choice = if request_tools.is_some() {
            self.tool_choice.clone()
        } else {
            None
        };

        let body = MistralChatRequest {
            model: self.model.clone(),
            messages: mistral_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: self.stream.unwrap_or(false),
            top_p: self.top_p,
            tools: request_tools,
            tool_choice: request_tool_choice,
            response_format,
        };

        let url = self
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("Mistral request payload: {json}");
            }
        }

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let response = request.send().await?;

        log::debug!("Mistral HTTP status: {}", response.status());

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Mistral API returned error status: {status}"),
                raw_response: error_text,
            });
        }

        // Parse the successful response
        let resp_text = response.text().await?;
        let json_resp: Result<MistralChatResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);

        match json_resp {
            Ok(response) => Ok(Box::new(response)),
            Err(e) => Err(LLMError::ResponseFormatError {
                message: format!("Failed to decode Mistral API response: {e}"),
                raw_response: resp_text,
            }),
        }
    }

    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None).await
    }

    /// Sends a streaming chat request to Mistral's API.
    ///
    /// # Arguments
    ///
    /// * `messages` - Slice of chat messages representing the conversation
    ///
    /// # Returns
    ///
    /// A stream of text tokens or an error
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Mistral API key".to_string()));
        }

        let messages = messages.to_vec();
        let mut mistral_msgs: Vec<MistralChatMessage> = vec![];

        for msg in messages {
            if let MessageType::ToolResult(results) = &msg.message_type {
                for result in results {
                    mistral_msgs.push(MistralChatMessage {
                        role: "tool".to_string(),
                        tool_call_id: Some(result.id.clone()),
                        tool_calls: None,
                        content: Some(Right(result.function.arguments.clone())),
                    });
                }
            } else {
                mistral_msgs.push(chat_message_to_api_message(msg))
            }
        }

        if let Some(system) = &self.system {
            mistral_msgs.insert(
                0,
                MistralChatMessage {
                    role: ROLE_SYSTEM.to_string(),
                    content: Some(Left(vec![MessageContent {
                        message_type: Some(MESSAGE_TYPE_TEXT.to_string()),
                        text: Some(system.clone()),
                        image_url: None,
                        tool_call_id: None,
                        tool_output: None,
                    }])),
                    tool_calls: None,
                    tool_call_id: None,
                },
            );
        }

        let body = MistralChatRequest {
            model: self.model.clone(),
            messages: mistral_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: true,
            top_p: self.top_p,
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            response_format: None,
        };

        let url = self
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Mistral API returned error status: {status}"),
                raw_response: error_text,
            });
        }

        Ok(crate::chat::create_sse_stream(response, |chunk| {
            crate::sse::parse_sse_chunk_json::<MistralChatStreamResponse>(chunk)
        }))
    }
}

// Create an owned MistralChatMessage
fn chat_message_to_api_message(chat_msg: ChatMessage) -> MistralChatMessage {
    MistralChatMessage {
        role: match chat_msg.role {
            ChatRole::User => ROLE_USER.to_string(),
            ChatRole::Assistant => ROLE_ASSISTANT.to_string(),
        },
        tool_call_id: None,
        content: match &chat_msg.message_type {
            MessageType::Text => Some(Right(chat_msg.content.clone())),
            MessageType::Image(_) => unimplemented!("{}", ERR_IMAGE_NOT_IMPLEMENTED),
            MessageType::Pdf(_) => unimplemented!("{}", ERR_PDF_NOT_IMPLEMENTED),
            MessageType::ImageURL(url) => {
                Some(Left(vec![MessageContent {
                    message_type: Some(MESSAGE_TYPE_IMAGE_URL.to_string()),
                    text: None,
                    image_url: Some(ImageUrlContent { url: url.clone() }),
                    tool_output: None,
                    tool_call_id: None,
                }]))
            }
            MessageType::ToolUse(_) => None,
            MessageType::ToolResult(_) => None,
        },
        tool_calls: match &chat_msg.message_type {
            MessageType::ToolUse(calls) => {
                let owned_calls: Vec<MistralFunctionCall> = calls
                    .iter()
                    .map(|c| {
                        MistralFunctionCall {
                            id: c.id.clone(),
                            content_type: MESSAGE_TYPE_FUNCTION.to_string(),
                            function: MistralFunctionPayload {
                                name: c.function.name.clone(),
                                arguments: c.function.arguments.clone(),
                            },
                        }
                    })
                    .collect();
                Some(owned_calls)
            }
            _ => None,
        },
    }
}

#[async_trait]
impl CompletionProvider for Mistral {
    /// Sends a completion request to Mistral's API.
    ///
    /// Currently not implemented.
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "Mistral completion not implemented.".into(),
        })
    }
}

#[cfg(feature = "mistral")]
#[async_trait]
impl EmbeddingProvider for Mistral {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Mistral API key".into()));
        }

        let emb_format = self
            .embedding_encoding_format
            .clone()
            .unwrap_or_else(|| "float".to_string());

        let body = MistralEmbeddingRequest {
            model: self.model.clone(),
            input,
            encoding_format: Some(emb_format),
        };

        let url = self
            .base_url
            .join("embeddings")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let resp = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await?
            .error_for_status()?;

        let json_resp: MistralEmbeddingResponse = resp.json().await?;

        let embeddings = json_resp.data.into_iter().map(|d| d.embedding).collect();
        Ok(embeddings)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct MistralModelEntry {
    pub id: String,
    pub created: Option<u64>,
    #[serde(flatten)]
    pub extra: Value,
}

impl ModelListRawEntry for MistralModelEntry {
    fn get_id(&self) -> String {
        self.id.clone()
    }

    fn get_created_at(&self) -> DateTime<Utc> {
        self.created
            .map(|t| chrono::DateTime::from_timestamp(t as i64, 0).unwrap_or_default())
            .unwrap_or_default()
    }

    fn get_raw(&self) -> Value {
        self.extra.clone()
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct MistralModelListResponse {
    pub data: Vec<MistralModelEntry>,
}

impl ModelListResponse for MistralModelListResponse {
    fn get_models(&self) -> Vec<String> {
        self.data.iter().map(|e| e.id.clone()).collect()
    }

    fn get_models_raw(&self) -> Vec<Box<dyn ModelListRawEntry>> {
        self.data
            .iter()
            .map(|e| Box::new(e.clone()) as Box<dyn ModelListRawEntry>)
            .collect()
    }

    fn get_backend(&self) -> LLMBackend {
        LLMBackend::Mistral
    }
}

#[async_trait]
impl ModelsProvider for Mistral {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        let url = self
            .base_url
            .join("models")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut req = self
            .client
            .get(url)
            .bearer_auth(&self.api_key);

        if let Some(timeout) = self.timeout_seconds {
            req = req.timeout(Duration::from_secs(timeout));
        }

        let resp = req
            .send()
            .await?
            .error_for_status()?;

        let result = resp.json::<MistralModelListResponse>().await?;

        Ok(Box::new(result))
    }
}

impl LLMProvider for Mistral {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}

#[async_trait]
impl SpeechToTextProvider for Mistral {
    /// Transcribes audio data to text using Mistral API
    ///
    /// Currently not implemented as Mistral doesn't provide STT capabilities
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::InvalidRequest(
            "Speech-to-text is not supported by Mistral".to_string(),
        ))
    }

    /// Transcribes audio file to text using Mistral API
    ///
    /// Currently not implemented as Mistral doesn't provide STT capabilities
    async fn transcribe_file(&self, _file_path: &str) -> Result<String, LLMError> {
        Err(LLMError::InvalidRequest(
            "Speech-to-text is not supported by Mistral".to_string(),
        ))
    }
}

#[async_trait]
impl TextToSpeechProvider for Mistral {
    /// Converts text to speech using Mistral's TTS API
    ///
    /// Currently not implemented as Mistral doesn't provide TTS capabilities
    async fn speech(&self, _text: &str) -> Result<Vec<u8>, LLMError> {
        Err(LLMError::InvalidRequest(
            "Text-to-speech is not supported by Mistral".to_string(),
        ))
    }
}