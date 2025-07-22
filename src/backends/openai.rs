//! OpenAI API client implementation for chat and completion functionality.
//!
//! This module provides integration with OpenAI's GPT models through their API.

use crate::constants::*;
use std::time::Duration;

#[cfg(feature = "openai")]
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

/// Client for interacting with OpenAI's API.
///
/// Provides methods for chat and completion requests using OpenAI's models.
pub struct OpenAI {
    pub api_key: String,
    pub base_url: Url,
    pub model: String,
    pub max_tokens: Option<u32>,
    pub max_completion_tokens: Option<u32>,
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
    pub voice: Option<String>,
    pub enable_web_search: Option<bool>,
    pub web_search_context_size: Option<String>,
    pub web_search_user_location_type: Option<String>,
    pub web_search_user_location_approximate_country: Option<String>,
    pub web_search_user_location_approximate_city: Option<String>,
    pub web_search_user_location_approximate_region: Option<String>,
    client: Client,
}

/// Individual message in an OpenAI chat conversation.
#[derive(Serialize, Debug)]
struct OpenAIChatMessage {
    #[allow(dead_code)]
    role: String,
    #[serde(
        skip_serializing_if = "Option::is_none",
        with = "either::serde_untagged_optional"
    )]
    content: Option<Either<Vec<MessageContent>, String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIFunctionCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Serialize, Debug)]
struct OpenAIFunctionPayload {
    name: String,
    arguments: String,
}

#[derive(Serialize, Debug)]
struct OpenAIFunctionCall {
    id: String,
    #[serde(rename = "type")]
    content_type: String,
    function: OpenAIFunctionPayload,
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

/// Individual image message in an OpenAI chat conversation.
#[derive(Serialize, Debug)]
struct ImageUrlContent {
    url: String,
}

#[derive(Serialize)]
struct OpenAIEmbeddingRequest {
    model: String,
    input: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    encoding_format: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    dimensions: Option<u32>,
}

/// Request payload for OpenAI's chat API endpoint.
#[derive(Serialize, Debug)]
struct OpenAIChatRequest {
    model: String,
    messages: Vec<OpenAIChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
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
    response_format: Option<OpenAIResponseFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    web_search_options: Option<OpenAIWebSearchOptions>,
}

impl std::fmt::Display for ToolCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{\n  \"id\": \"{}\",\n  \"type\": \"{}\",\n  \"function\": {}\n}}",
            self.id, self.call_type, self.function
        )
    }
}

impl std::fmt::Display for FunctionCall {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{{\n  \"name\": \"{}\",\n  \"arguments\": {}\n}}",
            self.name, self.arguments
        )
    }
}

/// Response from OpenAI's chat API endpoint.
#[derive(Deserialize, Debug)]
struct OpenAIChatResponse {
    choices: Vec<OpenAIChatChoice>,
    usage: Option<OpenAIUsage>,
}

/// Usage information from OpenAI's API response
#[derive(Deserialize, Debug)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

/// Individual choice within an OpenAI chat API response.
#[derive(Deserialize, Debug)]
struct OpenAIChatChoice {
    message: OpenAIChatMsg,
}

/// Message content within an OpenAI chat API response.
#[derive(Deserialize, Debug)]
struct OpenAIChatMsg {
    #[allow(dead_code)]
    role: String,
    content: Option<String>,
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Deserialize, Debug)]
struct OpenAIEmbeddingData {
    embedding: Vec<f32>,
}
#[derive(Deserialize, Debug)]
struct OpenAIEmbeddingResponse {
    data: Vec<OpenAIEmbeddingData>,
}

/// Response from OpenAI's streaming chat API endpoint.
#[derive(Deserialize, Debug)]
struct OpenAIChatStreamResponse {
    choices: Vec<OpenAIChatStreamChoice>,
}

/// Individual choice within an OpenAI streaming chat API response.
#[derive(Deserialize, Debug)]
struct OpenAIChatStreamChoice {
    delta: OpenAIChatStreamDelta,
}

/// Delta content within an OpenAI streaming chat API response.
#[derive(Deserialize, Debug)]
struct OpenAIChatStreamDelta {
    content: Option<String>,
}

impl crate::sse::SSEContentExtractor for OpenAIChatStreamResponse {
    fn extract_content(&self) -> Option<&str> {
        self.choices
            .first()
            .and_then(|c| c.delta.content.as_deref())
    }
}

/// An object specifying the format that the model must output.
///Setting to `{ "type": "json_schema", "json_schema": {...} }` enables Structured Outputs which ensures the model will match your supplied JSON schema. Learn more in the [Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs).
/// Setting to `{ "type": "json_object" }` enables the older JSON mode, which ensures the message the model generates is valid JSON. Using `json_schema` is preferred for models that support it.
#[derive(Deserialize, Debug, Serialize)]
enum OpenAIResponseType {
    #[serde(rename = "text")]
    Text,
    #[serde(rename = "json_schema")]
    JsonSchema,
    #[serde(rename = "json_object")]
    JsonObject,
}

#[derive(Deserialize, Debug, Serialize)]
struct OpenAIResponseFormat {
    #[serde(rename = "type")]
    response_type: OpenAIResponseType,
    #[serde(skip_serializing_if = "Option::is_none")]
    json_schema: Option<StructuredOutputFormat>,
}

#[derive(Deserialize, Debug, Serialize)]
struct OpenAIWebSearchOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    user_location: Option<UserLocation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    search_context_size: Option<String>,
}

#[derive(Deserialize, Debug, Serialize)]
struct UserLocation {
    #[serde(rename = "type")]
    location_type: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    approximate: Option<ApproximateLocation>,
}

#[derive(Deserialize, Debug, Serialize)]
struct ApproximateLocation {
    country: String,
    city: String,
    region: String,
}

impl From<StructuredOutputFormat> for OpenAIResponseFormat {
    /// Modify the schema to ensure that it meets OpenAI's requirements.
    fn from(structured_response_format: StructuredOutputFormat) -> Self {
        // It's possible to pass a StructuredOutputJsonSchema without an actual schema.
        // In this case, just pass the StructuredOutputJsonSchema object without modifying it.
        match structured_response_format.schema {
            None => OpenAIResponseFormat {
                response_type: OpenAIResponseType::JsonSchema,
                json_schema: Some(structured_response_format),
            },
            Some(mut schema) => {
                // Although [OpenAI's specifications](https://platform.openai.com/docs/guides/structured-outputs?api-mode=chat#additionalproperties-false-must-always-be-set-in-objects) say that the "additionalProperties" field is required, my testing shows that it is not.
                // Just to be safe, add it to the schema if it is missing.
                schema = if schema.get("additionalProperties").is_none() {
                    schema["additionalProperties"] = serde_json::json!(false);
                    schema
                } else {
                    schema
                };

                OpenAIResponseFormat {
                    response_type: OpenAIResponseType::JsonSchema,
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

impl ChatResponse for OpenAIChatResponse {
    fn text(&self) -> Option<String> {
        self.choices.first().and_then(|c| c.message.content.clone())
    }

    fn tool_calls(&self) -> Option<Vec<ToolCall>> {
        self.choices
            .first()
            .and_then(|c| c.message.tool_calls.clone())
    }

    fn usage(&self) -> Option<Usage> {
        self.usage.as_ref().map(|u| Usage {
            prompt_tokens: u.prompt_tokens,
            completion_tokens: u.completion_tokens,
            total_tokens: u.total_tokens,
        })
    }
}

impl std::fmt::Display for OpenAIChatResponse {
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

impl OpenAI {
    /// Determines whether to use max_tokens or max_completion_tokens based on the model
    fn should_use_max_completion_tokens(model: &str) -> bool {
        REASONING_MODEL_PREFIXES.iter().any(|&prefix| model.starts_with(prefix))
    }
    
    /// Determines whether a model supports temperature parameter
    fn supports_temperature(model: &str) -> bool {
        // Reasoning models don't support temperature
        !Self::should_use_max_completion_tokens(model)
    }
    
    /// Determines whether a model supports top_p parameter
    fn supports_top_p(model: &str) -> bool {
        // Reasoning models don't support top_p
        !Self::should_use_max_completion_tokens(model)
    }

    /// Creates a new OpenAI client with the specified configuration.
    ///
    /// # Arguments
    ///
    /// * `api_key` - OpenAI API key
    /// * `model` - Model to use (defaults to "gpt-4")
    /// * `max_tokens` - Maximum tokens to generate (deprecated for newer models)
    /// * `max_completion_tokens` - Maximum completion tokens (for newer models)
    /// * `temperature` - Sampling temperature (ignored for reasoning models like o1, o3)
    /// * `timeout_seconds` - Request timeout in seconds
    /// * `system` - System prompt
    /// * `stream` - Whether to stream responses
    /// * `top_p` - Top-p sampling parameter (ignored for reasoning models like o1, o3)
    /// * `top_k` - Top-k sampling parameter
    /// * `embedding_encoding_format` - Format for embedding outputs
    /// * `embedding_dimensions` - Dimensions for embedding vectors
    /// * `tools` - Function tools that the model can use
    /// * `tool_choice` - Determines how the model uses tools
    /// * `reasoning_effort` - Reasoning effort level
    /// * `json_schema` - JSON schema for structured output
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        api_key: impl Into<String>,
        base_url: Option<String>,
        model: Option<String>,
        max_tokens: Option<u32>,
        max_completion_tokens: Option<u32>,
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
        voice: Option<String>,
        enable_web_search: Option<bool>,
        web_search_context_size: Option<String>,
        web_search_user_location_type: Option<String>,
        web_search_user_location_approximate_country: Option<String>,
        web_search_user_location_approximate_city: Option<String>,
        web_search_user_location_approximate_region: Option<String>,
    ) -> Result<Self, LLMError> {
        let mut builder = Client::builder();
        if let Some(sec) = timeout_seconds {
            builder = builder.timeout(std::time::Duration::from_secs(sec));
        }
        
        let base_url_str = base_url.unwrap_or_else(|| DEFAULT_OPENAI_BASE_URL.to_owned());
        let base_url = Url::parse(&base_url_str)
            .map_err(|e| LLMError::InvalidRequest(format!("Invalid base URL '{}': {}", base_url_str, e)))?;
        
        let client = builder.build()
            .map_err(|e| LLMError::InvalidRequest(format!("Failed to build HTTP client: {}", e)))?;
        
        Ok(Self {
            api_key: api_key.into(),
            base_url,
            model: model.unwrap_or_else(|| DEFAULT_OPENAI_MODEL.to_string()),
            max_tokens,
            max_completion_tokens,
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
            client,
            reasoning_effort,
            json_schema,
            voice,
            enable_web_search,
            web_search_context_size,
            web_search_user_location_type,
            web_search_user_location_approximate_country,
            web_search_user_location_approximate_city,
            web_search_user_location_approximate_region,
        })
    }
}

#[async_trait]
impl ChatProvider for OpenAI {
    /// Sends a chat request to OpenAI's API.
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
            return Err(LLMError::AuthError("Missing OpenAI API key".to_string()));
        }

        // Clone the messages to have an owned mutable vector.
        let messages = messages.to_vec();

        let mut openai_msgs: Vec<OpenAIChatMessage> = vec![];

        for msg in messages {
            if let MessageType::ToolResult(results) = &msg.message_type {
                for result in results {
                    openai_msgs.push(
                        OpenAIChatMessage {
                            role: ROLE_TOOL.to_string(),
                            tool_call_id: Some(result.id.clone()),
                            tool_calls: None,
                            content: Some(Right(result.function.arguments.clone())),
                        },
                    );
                }
            } else {
                openai_msgs.push(chat_message_to_api_message(msg))
            }
        }

        if let Some(system) = &self.system {
            openai_msgs.insert(
                0,
                OpenAIChatMessage {
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

        let response_format: Option<OpenAIResponseFormat> =
            self.json_schema.clone().map(|s| s.into());

        let request_tools = tools.map(|t| t.to_vec()).or_else(|| self.tools.clone());

        let request_tool_choice = if request_tools.is_some() {
            self.tool_choice.clone()
        } else {
            None
        };

        let web_search_options = if self.enable_web_search.unwrap_or(false) {
            let loc_type_opt = self
                .web_search_user_location_type
                .as_ref()
                .filter(|t| matches!(t.as_str(), "exact" | "approximate"));

            let country = self.web_search_user_location_approximate_country.as_ref();
            let city = self.web_search_user_location_approximate_city.as_ref();
            let region = self.web_search_user_location_approximate_region.as_ref();

            let approximate = if [country, city, region].iter().any(|v| v.is_some()) {
                Some(ApproximateLocation {
                    country: country.cloned().unwrap_or_default(),
                    city: city.cloned().unwrap_or_default(),
                    region: region.cloned().unwrap_or_default(),
                })
            } else {
                None
            };

            let user_location = loc_type_opt.map(|loc_type| UserLocation {
                location_type: loc_type.clone(),
                approximate,
            });

            Some(OpenAIWebSearchOptions {
                search_context_size: self.web_search_context_size.clone(),
                user_location,
            })
        } else {
            None
        };

        // Determine which token limit parameter to use based on the model
        let (max_tokens, max_completion_tokens) = if Self::should_use_max_completion_tokens(&self.model) {
            (None, self.max_completion_tokens.or(self.max_tokens))
        } else {
            (self.max_tokens, None)
        };
        
        // Only include temperature and top_p for models that support them
        let temperature = if Self::supports_temperature(&self.model) {
            self.temperature
        } else {
            None
        };
        
        let top_p = if Self::supports_top_p(&self.model) {
            self.top_p
        } else {
            None
        };

        let body = OpenAIChatRequest {
            model: self.model.clone(),
            messages: openai_msgs,
            max_tokens,
            max_completion_tokens,
            temperature,
            stream: self.stream.unwrap_or(false),
            top_p,
            top_k: self.top_k,
            tools: request_tools,
            tool_choice: request_tool_choice,
            reasoning_effort: self.reasoning_effort.clone(),
            response_format,
            web_search_options,
        };

        let url = self
            .base_url
            .join("chat/completions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut request = self.client.post(url).bearer_auth(&self.api_key).json(&body);

        if log::log_enabled!(log::Level::Trace) {
            if let Ok(json) = serde_json::to_string(&body) {
                log::trace!("OpenAI request payload: {json}");
            }
        }

        if let Some(timeout) = self.timeout_seconds {
            request = request.timeout(std::time::Duration::from_secs(timeout));
        }

        let response = request.send().await?;

        log::debug!("OpenAI HTTP status: {}", response.status());

        if !response.status().is_success() {
            let status = response.status();
            let error_text = response.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("OpenAI API returned error status: {status}"),
                raw_response: error_text,
            });
        }

        // Parse the successful response
        let resp_text = response.text().await?;
        let json_resp: Result<OpenAIChatResponse, serde_json::Error> =
            serde_json::from_str(&resp_text);

        match json_resp {
            Ok(response) => Ok(Box::new(response)),
            Err(e) => Err(LLMError::ResponseFormatError {
                message: format!("Failed to decode OpenAI API response: {e}"),
                raw_response: resp_text,
            }),
        }
    }

    async fn chat(&self, messages: &[ChatMessage]) -> Result<Box<dyn ChatResponse>, LLMError> {
        self.chat_with_tools(messages, None).await
    }

    /// Sends a streaming chat request to OpenAI's API.
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
            return Err(LLMError::AuthError("Missing OpenAI API key".to_string()));
        }

        let messages = messages.to_vec();
        let mut openai_msgs: Vec<OpenAIChatMessage> = vec![];

        for msg in messages {
            if let MessageType::ToolResult(results) = &msg.message_type {
                for result in results {
                    openai_msgs.push(OpenAIChatMessage {
                        role: "tool".to_string(),
                        tool_call_id: Some(result.id.clone()),
                        tool_calls: None,
                        content: Some(Right(result.function.arguments.clone())),
                    });
                }
            } else {
                openai_msgs.push(chat_message_to_api_message(msg))
            }
        }

        if let Some(system) = &self.system {
            openai_msgs.insert(
                0,
                OpenAIChatMessage {
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

        // Determine which token limit parameter to use based on the model
        let (max_tokens, max_completion_tokens) = if Self::should_use_max_completion_tokens(&self.model) {
            (None, self.max_completion_tokens.or(self.max_tokens))
        } else {
            (self.max_tokens, None)
        };
        
        // Only include temperature for models that support it
        let temperature = if Self::supports_temperature(&self.model) {
            self.temperature
        } else {
            None
        };

        let body = OpenAIChatRequest {
            model: self.model.clone(),
            messages: openai_msgs,
            max_tokens,
            max_completion_tokens,
            temperature,
            stream: true,
            top_p: self.top_p,
            top_k: self.top_k,
            tools: self.tools.clone(),
            tool_choice: self.tool_choice.clone(),
            reasoning_effort: self.reasoning_effort.clone(),
            response_format: None,
            web_search_options: None,
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
                message: format!("OpenAI API returned error status: {status}"),
                raw_response: error_text,
            });
        }

        Ok(crate::chat::create_sse_stream(response, |chunk| {
            crate::sse::parse_sse_chunk_json::<OpenAIChatStreamResponse>(chunk)
        }))
    }
}

// Create an owned OpenAIChatMessage
fn chat_message_to_api_message(chat_msg: ChatMessage) -> OpenAIChatMessage {
    OpenAIChatMessage {
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
                let owned_calls: Vec<OpenAIFunctionCall> = calls
                    .iter()
                    .map(|c| {
                        OpenAIFunctionCall {
                            id: c.id.clone(),
                            content_type: MESSAGE_TYPE_FUNCTION.to_string(),
                            function: OpenAIFunctionPayload {
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
impl CompletionProvider for OpenAI {
    /// Sends a completion request to OpenAI's API.
    ///
    /// Currently not implemented.
    async fn complete(&self, _req: &CompletionRequest) -> Result<CompletionResponse, LLMError> {
        Ok(CompletionResponse {
            text: "OpenAI completion not implemented.".into(),
        })
    }
}

#[async_trait]
impl SpeechToTextProvider for OpenAI {
    /// Transcribes audio data to text using OpenAI API
    ///
    /// # Arguments
    ///
    /// * `audio` - Raw audio data as bytes
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - Transcribed text
    /// * `Err(LLMError)` - Error if transcription fails
    async fn transcribe(&self, audio: Vec<u8>) -> Result<String, LLMError> {
        let url = self
            .base_url
            .join("audio/transcriptions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let part = reqwest::multipart::Part::bytes(audio).file_name("audio.m4a");
        let form = reqwest::multipart::Form::new()
            .text("model", self.model.clone())
            .text("response_format", "text")
            .part("file", part);

        let mut req = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .multipart(form);

        if let Some(t) = self.timeout_seconds {
            req = req.timeout(Duration::from_secs(t));
        }

        let resp = req.send().await?;
        let text = resp.text().await?;
        let raw = text.clone();
        Ok(raw)
    }

    /// Transcribes audio file to text using OpenAI API
    ///
    /// # Arguments
    ///
    /// * `file_path` - Path to the audio file
    ///
    /// # Returns
    ///
    /// * `Ok(String)` - Transcribed text
    /// * `Err(LLMError)` - Error if transcription fails
    async fn transcribe_file(&self, file_path: &str) -> Result<String, LLMError> {
        let url = self
            .base_url
            .join("audio/transcriptions")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let form = reqwest::multipart::Form::new()
            .text("model", self.model.clone())
            .text("response_format", "text")
            .file("file", file_path)
            .await
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let mut req = self
            .client
            .post(url)
            .bearer_auth(&self.api_key)
            .multipart(form);

        if let Some(t) = self.timeout_seconds {
            req = req.timeout(Duration::from_secs(t));
        }

        let resp = req.send().await?;
        let text = resp.text().await?;
        let raw = text.clone();
        Ok(raw)
    }
}

#[cfg(feature = "openai")]
#[async_trait]
impl EmbeddingProvider for OpenAI {
    async fn embed(&self, input: Vec<String>) -> Result<Vec<Vec<f32>>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing OpenAI API key".into()));
        }

        let emb_format = self
            .embedding_encoding_format
            .clone()
            .unwrap_or_else(|| "float".to_string());

        let body = OpenAIEmbeddingRequest {
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

        let json_resp: OpenAIEmbeddingResponse = resp.json().await?;

        let embeddings = json_resp.data.into_iter().map(|d| d.embedding).collect();
        Ok(embeddings)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct OpenAIModelEntry {
    pub id: String,
    pub created: Option<u64>,
    #[serde(flatten)]
    pub extra: Value,
}

impl ModelListRawEntry for OpenAIModelEntry {
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
pub struct OpenAIModelListResponse {
    pub data: Vec<OpenAIModelEntry>,
}

impl ModelListResponse for OpenAIModelListResponse {
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
        LLMBackend::OpenAI
    }
}

#[async_trait]
impl ModelsProvider for OpenAI {
    async fn list_models(
        &self,
        _request: Option<&ModelListRequest>,
    ) -> Result<Box<dyn ModelListResponse>, LLMError> {
        let url = self
            .base_url
            .join("models")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        let resp = self
            .client
            .get(url)
            .bearer_auth(&self.api_key)
            .send()
            .await?
            .error_for_status()?;

        let result = resp.json::<OpenAIModelListResponse>().await?;

        Ok(Box::new(result))
    }
}

impl LLMProvider for OpenAI {
    fn tools(&self) -> Option<&[Tool]> {
        self.tools.as_deref()
    }
}

#[async_trait]
impl TextToSpeechProvider for OpenAI {
    /// Converts text to speech using OpenAI's TTS API
    ///
    /// # Arguments
    /// * `text` - The text to convert to speech
    ///
    /// # Returns
    /// * `Result<Vec<u8>, LLMError>` - Audio data as bytes or error
    async fn speech(&self, text: &str) -> Result<Vec<u8>, LLMError> {
        if self.api_key.is_empty() {
            return Err(LLMError::AuthError("Missing OpenAI API key".into()));
        }

        let url = self
            .base_url
            .join("audio/speech")
            .map_err(|e| LLMError::HttpError(e.to_string()))?;

        #[derive(Serialize)]
        struct SpeechRequest {
            model: String,
            input: String,
            voice: String,
        }

        let body = SpeechRequest {
            model: self.model.clone(),
            input: text.to_string(),
            voice: self.voice.clone().unwrap_or("alloy".to_string()),
        };

        let mut req = self.client.post(url).bearer_auth(&self.api_key).json(&body);

        if let Some(t) = self.timeout_seconds {
            req = req.timeout(Duration::from_secs(t));
        }

        let resp = req.send().await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let error_text = resp.text().await?;
            return Err(LLMError::ResponseFormatError {
                message: format!("OpenAI API returned error status: {status}"),
                raw_response: error_text,
            });
        }

        Ok(resp.bytes().await?.to_vec())
    }
}

