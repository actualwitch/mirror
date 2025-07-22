// Import required modules from the LLM library for OpenAI integration
use mirror::{
    builder::{FunctionBuilder, LLMBackend, LLMBuilder, ParamBuilder},
    chat::ChatMessage, // Chat-related structures
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get OpenAI API key from environment variable or use test key as fallback
    let api_key = std::env::var("AZURE_OPENAI_API_KEY").unwrap_or("".into());
    let api_version =
        std::env::var("AZURE_OPENAI_API_VERSION").unwrap_or("2025-01-01-preview".into());
    let endpoint = std::env::var("AZURE_OPENAI_API_ENDPOINT").unwrap_or("".into());

    // Initialize and configure the LLM client
    let llm = LLMBuilder::new()
        .backend(LLMBackend::AzureOpenAI) // Use OpenAI as the LLM provider
        .base_url(endpoint)
        .api_key(api_key) // Set the API key
        .api_version(api_version) // Set the API key
        .deployment_id("gpt-4o-mini")
        .model("gpt-4o-mini") // Use GPT-4o-mini model
        .max_tokens(512) // Limit response length
        .temperature(0.7) // Control response randomness (0.0-1.0)
        .stream(false) // Disable streaming responses
        .function(
            FunctionBuilder::new("weather_function")
                .description("Use this tool to get the weather in a specific city")
                .param(
                    ParamBuilder::new("url")
                        .type_of("string")
                        .description("The url to get the weather from for the city"),
                )
                .required(vec!["url".to_string()]),
        )
        .build()
        .expect("Failed to build LLM");

    // Prepare conversation history with example messages
    let messages = vec![ChatMessage::user().content("You are a weather assistant. What is the weather in Tokyo? Use the tools that you have available").build()];

    // Send chat request and handle the response
    // this returns the response as a string. The tool call is also returned as a serialized string. We can deserialize if needed.
    match llm.chat_with_tools(&messages, llm.tools()).await {
        Ok(text) => println!("Chat response:\n{}", text),
        Err(e) => eprintln!("Chat error: {}", e),
    }

    Ok(())
}
