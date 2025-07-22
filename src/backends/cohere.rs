//! Cohere API client implementation for chat and completion functionality.
//!
//! This module provides integration with Cohere's LLM models through their Compatibility API.
use crate::constants::*;
use std::time::Duration;

#[cfg(feature = "cohere")]
use crate::{
    chat::Tool,
    chat::{ChatMessage, ChatProvider, ChatRole, MessageType, StructuredOutputFormat},
    completion::{CompletionProvider, CompletionRequest, CompletionResponse},
    embedding::EmbeddingProvider,
    error::LLMError,
    models::ModelsProvider,
    stt::SpeechToTextProvider,
    tts::TextToSpeechProvider,
    LLMProvider,
};
#[cfg(feature = "cohere")]
use crate::{
    chat::{ChatResponse, ToolChoice},
    ToolCall,
};
use async_trait::async_trait;
use either::*;
use futures::stream::Stream;
use reqwest::{Client, Url};
use serde::{Deserialize, Serialize};

/// Client for interacting with Cohere's API (OpenAI compatibility mode).
///
/// Provides methods for chat and embedding requests using Cohere's models.  
/// **Note:** Cohere expects system instructions to use the `developer` role instead of `system`:contentReference[oaicite:0]{index=0}.
pub struct Cohere {
    pub api_key: String,
    pub base_url: Url,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub temperature: Option<f32>,
    pub system: Option<String>,
    pub timeout_seconds: Option<u64>,
    pub stream: Option<bool>,
    pub top_p: Option<f32>,
    pub top_k: Option<u32>,
    pub tools: Option<Vec<Tool>>,
    pub tool_choice: Option<ToolChoice>,
    /// Embedding parameters
    pub embedding_encoding_format: Option<String>,
    pub embedding_dimensions: Option<u32>,
    pub reasoning_effort: Option<String>,
    /// JSON schema for structured output
    pub json_schema: Option<StructuredOutputFormat>,
    client: Client,
}

/// Individual message in a Cohere chat conversation.
#[derive(Serialize, Debug)]
struct CohereChatMessage {
    #[allow(dead_code)]
    role: String,
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "either::serde_untagged_optional"
    )]
    content: Option<Either<Vec<CohereMessageContent>, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<CohereFunctionCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Debug)]
struct CohereFunctionPayload {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug)]
struct CohereFunctionCall {
    id: String,
    #[serde(rename = "type")]
    content_type: String,
    function: CohereFunctionPayload,
}

#[derive(Serialize, Debug)]
struct CohereMessageContent {
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

/// Individual image message (URL) in a Cohere chat conversation.
#[derive(Serialize, Debug)]
struct ImageUrlContent {
    url: String,
}

#[derive(Serialize)]
struct CohereEmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

/// Request payload for Cohere's chat API endpoint.
#[derive(Serialize, Debug)]
struct CohereChatRequest {
    model: String,
    messages: Vec<CohereChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_k: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<Tool>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_choice: Option<ToolChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reasoning_effort: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    response_format: Option<CohereResponseFormat>,
}

/// Response from Cohere's chat API endpoint.
#[derive(Deserialize, Debug)]
struct CohereChatResponse {
    choices: Vec<CohereChatChoice>,
}

/// Individual choice within a Cohere chat API response.
#[derive(Deserialize, Debug)]
struct CohereChatChoice {
    message: CohereChatMsg,
}

/// Message content within a Cohere chat API response.
#[derive(Deserialize, Debug)]
struct CohereChatMsg {
    #[allow(dead_code)]
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

/// Response from Cohere's embedding API endpoint.
#[derive(Deserialize, Debug)]
struct CohereEmbeddingData {
    embedding: Vec<f32>,
}
#[derive(Deserialize, Debug)]
struct CohereEmbeddingResponse {
    data: Vec<CohereEmbeddingData>,
}

/// Output format type for structured responses in Cohere.
#[derive(Deserialize, Debug, Serialize)]
enum CohereResponseType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_schema")]
    JsonSchema,
    #[serde(rename = "json_object")]
    JsonObject,
}

/// Configuration for forcing the model output format (e.g., JSON schema).
#[derive(Deserialize, Debug, Serialize)]
struct CohereResponseFormat {
    #[serde(rename = "type")]
    response_type: CohereResponseType,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<StructuredOutputFormat>,
}

impl From<StructuredOutputFormat> for CohereResponseFormat {
    fn from(structured_response_format: StructuredOutputFormat) -> Self {
        match structured_response_format.schema {
            None => CohereResponseFormat {
                response_type: CohereResponseType::JsonSchema,
                json_schema: Some(structured_response_format),
            },
            Some(mut schema) => {
                // Ensure "additionalProperties": false in schema if missing
                if schema.get("additionalProperties").is_none() {
                    schema["additionalProperties"] = serde_json::json!(false);
                }
                CohereResponseFormat {
                    response_type: CohereResponseType::JsonSchema,
                    json_schema: Some(StructuredOutputFormat {
                        name: structured_response_format.name,
                        description: structured_response_format.description,
                        schema: Some(schema),
                        strict: structured_response_format.strict,
                    }),
                }
            }
        }
    }
}

impl ChatResponse for CohereChatResponse {
    fn text(&self) -> Option<String> {
        self.choices.first().and_then(|c| c.message.content.clone())
    }
    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.choices
            .first()
            .and_then(|c| c.message.tool_calls.clone())
    }
}

impl std::fmt::Display for CohereChatResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let first_choice = match self.choices.first() {
            Some(choice) => choice,
            None => return write!(f, "{}", ERR_NO_RESPONSE_CHOICES),
        };
        
        match (
            &first_choice.message.content,
            &first_choice.message.tool_calls,
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

impl Cohere {
    /// Creates a new Cohere client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - Cohere API key  
    /// * `base_url` - Base URL for Cohere API (defaults to Cohere compatibility API endpoint)  
    /// * `model` - Model to use (e.g., "command-xlarge")  
    /// * `max_tokens` - Maximum tokens to generate  
    /// * `temperature` - Sampling temperature  
    /// * `timeout_seconds` - Request timeout in seconds  
    /// * `system` - System prompt (sent as a developer role message)  
    /// * `stream` - Whether to stream responses  
    /// * `top_p` - Top-p sampling parameter  
    /// * `top_k` - Top-k sampling parameter  
    /// * `embedding_encoding_format` - Format for embedding outputs (`float` or `base64`)  
    /// * `embedding_dimensions` - (Unused by Cohere) Dimensions for embedding vectors  
    /// * `tools` - Function tools available to the model  
    /// * `tool_choice` - Determines how the model uses tools  
    /// * `reasoning_effort` - Reasoning effort level (unsupported by Cohere)  
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
        top_k: Option<u32>,
        embedding_encoding_format: Option<String>,
        embedding_dimensions: Option<u32>,
        tools: Option<Vec<Tool>>,
        tool_choice: Option<ToolChoice>,
        reasoning_effort: Option<String>,
        json_schema: Option<StructuredOutputFormat>,
    ) -> Result<Self, LLMError> {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(Duration::from_secs(sec));
        }
        
        let base_url_str = base_url.unwrap_or_else(|| DEFAULT_COHERE_BASE_URL.to_owned());
        let base_url = Url::parse(&base_url_str)
            .map_err(|e| LLMError::InvalidRequest(format!("Invalid base URL '{}': {}", base_url_str, e)))?;
        
        let client = builder.build()
            .map_err(|e| LLMError::InvalidRequest(format!("Failed to build HTTP client: {}", e)))?;
        
        Ok(Self {
            api_key: api_key.into(),
            base_url,
            model: model.unwrap_or_else(|| DEFAULT_COHERE_MODEL.to_string()),
            max_tokens,
            temperature,
            system,
            timeout_seconds,
            stream,
            top_p,
            top_k,
            tools,
            tool_choice,
            embedding_encoding_format,
            embedding_dimensions,
            reasoning_effort,
            json_schema,
            client,
        })
    }
}

#[async_trait]
impl ChatProvider for Cohere {
    /// Sends a chat request to Cohere's API (optionally with tool usage).
    async fn chat_with_tools(
        &self,
        messages: &[ChatMessage],
        tools: Option<&[Tool]>,
    ) -> Result<Box<dyn ChatResponse>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Cohere API key".to_string()));
        }
        // Clone messages to own them
        let messages = messages.to_vec();
        let mut cohere_msgs: Vec<CohereChatMessage> = vec![];

        for msg in messages {
            if let MessageType::ToolResult(results) = &msg.message_type {
                // Include tool result as a message with role "tool"
                for result in results {
                    cohere_msgs.push(CohereChatMessage {
                        role: ROLE_TOOL.to_string(),
                        tool_call_id: Some(result.id.clone()),
                        tool_calls: None,
                        content: Some(Right(result.function.arguments.clone())),
                    });
                }
            } else {
                cohere_msgs.push(chat_message_to_api_message(msg));
            }
        }

        // Prepend system prompt as a "developer" role message if provided
        if let Some(system) = &self.system {
            cohere_msgs.insert(
                0,
                CohereChatMessage {
                    role: ROLE_DEVELOPER.to_string(),
                    content: Some(Left(vec![CohereMessageContent {
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

        let response_format: Option<CohereResponseFormat> =
            self.json_schema.clone().map(|s| s.into());
        let request_tools = tools.map(|t| t.to_vec()).or_else(|| self.tools.clone());
        let request_tool_choice = if request_tools.is_some() {
            self.tool_choice.clone()
        } else {
            None
        };

        // Build the request payload
        let body = CohereChatRequest {
            model: self.model.clone(),
            messages: cohere_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: self.stream.unwrap_or(false),
            top_p: self.top_p,
            top_k: self.top_k,
            tools: request_tools,
            tool_choice: request_tool_choice,
            reasoning_effort: self.reasoning_effort.clone(),
            response_format,
        };

        let url = self
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;
        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("Cohere request payload: {json}");
            }
        }
        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(Duration::from_secs(timeout));
        }
        let response = request.send().await?;
        log::debug!("Cohere HTTP status: {}", response.status());

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Cohere API returned error status: {status}"),
                raw_response: error_text,
            });
        }
        // Parse the successful response
        let resp_text = response.text().await?;
        let json_resp: Result<CohereChatResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);
        match json_resp {
            Ok(res) => Ok(Box::new(res)),
            Err(e) => Err(LLMError::ResponseFormatError {
                message: format!("Failed to decode Cohere API response: {e}"),
                raw_response: resp_text,
            }),
        }
    }

    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None).await
    }

    /// Sends a streaming chat request to Cohere's API.
    ///
    /// # Returns
    /// A stream of response text chunks or an error if the request fails.
    async fn chat_stream(
        &self,
        messages: &[ChatMessage],
    ) -> Result<std::pin::Pin<Box<dyn Stream<Item = Result<String, LLMError>> + Send>>, LLMError>
    {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Cohere API key".to_string()));
        }
        let messages = messages.to_vec();
        let mut cohere_msgs: Vec<CohereChatMessage> = vec![];

        for msg in messages {
            if let MessageType::ToolResult(results) = &msg.message_type {
                for result in results {
                    cohere_msgs.push(CohereChatMessage {
                        role: ROLE_TOOL.to_string(),
                        tool_call_id: Some(result.id.clone()),
                        tool_calls: None,
                        content: Some(Right(result.function.arguments.clone())),
                    });
                }
            } else {
                cohere_msgs.push(chat_message_to_api_message(msg));
            }
        }
        if let Some(system) = &self.system {
            cohere_msgs.insert(
                0,
                CohereChatMessage {
                    role: ROLE_DEVELOPER.to_string(),
                    content: Some(Left(vec![CohereMessageContent {
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

        let body = CohereChatRequest {
            model: self.model.clone(),
            messages: cohere_msgs,
            max_tokens: self.max_tokens,
            temperature: self.temperature,
            stream: true,
            top_p: self.top_p,
            top_k: self.top_k,
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            reasoning_effort: self.reasoning_effort.clone(),
            response_format: None,
        };
        let url = self
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;
        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);
        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(Duration::from_secs(timeout));
        }
        let response = request.send().await?;
        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("Cohere API returned error status: {status}"),
                raw_response: error_text,
            });
        }
        // Return a Server-Sent Events stream of the response content
        Ok(crate::chat::create_sse_stream(response, |chunk| {
            crate::sse::parse_sse_chunk_json::<CohereChatStreamResponse>(chunk)
        }))
    }
}

// Convert a ChatMessage into a CohereChatMessage.
fn chat_message_to_api_message(chat_msg: ChatMessage) -> CohereChatMessage {
    CohereChatMessage {
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
                Some(Left(vec![CohereMessageContent {
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
                let owned_calls: Vec<CohereFunctionCall> = calls
                    .iter()
                    .map(|c| {
                        CohereFunctionCall {
                            id: c.id.clone(),
                            content_type: MESSAGE_TYPE_FUNCTION.to_string(),
                            function: CohereFunctionPayload {
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


#[derive(Deserialize, Debug)]
struct CohereChatStreamResponse {
    choices: Vec<CohereChatStreamChoice>,
}
#[derive(Deserialize, Debug)]
struct CohereChatStreamChoice {
    delta: CohereChatStreamDelta,
}
#[derive(Deserialize, Debug)]
struct CohereChatStreamDelta {
    content: Option<String>,
}

impl crate::sse::SSEContentExtractor for CohereChatStreamResponse {
    fn extract_content(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.delta.content.as_deref())
    }
}

#[async_trait]
impl CompletionProvider for Cohere {
    /// Sends a completion request to Cohere's API (not supported in compatibility mode).
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "Cohere completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl EmbeddingProvider for Cohere {
    /// Generates embeddings for the given input texts using Cohere's API.
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing Cohere API key".into()));
        }
        let emb_format = self
            .embedding_encoding_format
            .clone()
            .unwrap_or_else(|| "float".to_string());
        let body = CohereEmbeddingRequest {
            model: self.model.clone(),
            input,
            encoding_format: Some(emb_format),
            dimensions: self.embedding_dimensions,
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
        let json_resp: CohereEmbeddingResponse = resp.json().await?;
        let embeddings = json_resp.data.into_iter().map(|d| d.embedding).collect();
        Ok(embeddings)
    }
}

impl LLMProvider for Cohere {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}

#[async_trait]
impl SpeechToTextProvider for Cohere {
    /// Transcribing audio is not supported by Cohere.
    async fn transcribe(&self, _audio: Vec<u8>) -> Result<String, LLMError> {
        Err(LLMError::ProviderError(
            "Cohere does not implement speech-to-text.".into(),
        ))
    }
}

#[async_trait]
impl TextToSpeechProvider for Cohere {
    /// Text-to-speech conversion is not supported by Cohere.
    async fn speech(&self, _text: &str) -> Result<Vec<u8>, LLMError> {
        Err(LLMError::ProviderError(
            "Text-to-speech not supported by Cohere.".into(),
        ))
    }
}

#[async_trait]
impl ModelsProvider for Cohere {
    // Uses default implementation: listing models is not supported by Cohere
}
