//! Example of using the Mistral backend with the mirror crate.
//!
//! This example demonstrates how to use Mistral's models for chat and embeddings.

use mirror::{
    builder::{LLMBackend, LLMBuilder, FunctionBuilder, ParamBuilder}, 
    chat::{ChatMessage, ChatRole, MessageType, StructuredOutputFormat}
};
use serde_json::Value;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the Mistral provider
    let provider = LLMBuilder::default()
        .backend(LLMBackend::Mistral)
        .api_key(std::env::var("MISTRAL_API_KEY")?)
        .model("mistral-small-latest") // or "mistral-medium-latest", "mistral-large-latest"
        .temperature(0.7)
        .max_tokens(1000)
        .build()?;

    // Example 1: Simple chat
    println!("=== Simple Chat Example ===");
    let messages = vec![
        ChatMessage {
            role: ChatRole::Assistant,
            message_type: MessageType::Text,
            content: "You are a helpful assistant.".to_string(),
        },
        ChatMessage {
            role: ChatRole::User,
            message_type: MessageType::Text,
            content: "What is the capital of France?".to_string(),
        },
    ];

    let response = provider.chat(&messages).await?;
    println!("Response: {}", response.text().unwrap_or_default());

    // Example 2: Streaming chat
    println!("\n=== Streaming Chat Example ===");
    let streaming_provider = LLMBuilder::default()
        .backend(LLMBackend::Mistral)
        .api_key(std::env::var("MISTRAL_API_KEY")?)
        .model("mistral-small-latest")
        .stream(true)
        .build()?;

    let messages = vec![ChatMessage {
        role: ChatRole::User,
        message_type: MessageType::Text,
        content: "Tell me a short story about a robot.".to_string(),
    }];

    println!("Streaming response:");
    let mut stream = streaming_provider.chat_stream(&messages).await?;
    use futures::StreamExt;
    while let Some(chunk) = stream.next().await {
        match chunk {
            Ok(response) => print!("{}", response),
            Err(e) => eprintln!("Stream error: {}", e),
        }
    }
    println!();

    // Example 3: Embeddings
    println!("\n=== Embeddings Example ===");
    let embedding_provider = LLMBuilder::default()
        .backend(LLMBackend::Mistral)
        .api_key(std::env::var("MISTRAL_API_KEY")?)
        .model("mistral-embed") // Mistral's embedding model
        .build()?;

    let texts = vec![
        "The quick brown fox jumps over the lazy dog.".to_string(),
        "Machine learning is transforming technology.".to_string(),
    ];

    let embeddings = embedding_provider.embed(texts).await?;
    println!("Generated {} embeddings", embeddings.len());
    println!("First embedding dimension: {}", embeddings[0].len());

    // Example 4: List available models
    println!("\n=== Available Models ===");
    let models = provider.list_models(None).await?;
    println!("Available Mistral models:");
    for model_id in models.get_models() {
        println!("  - {}", model_id);
    }

    // Example 5: Using Tools (Function Calling)
    println!("\n=== Function Calling Example ===");
    
    // Define a weather function
    let weather_function = FunctionBuilder::new("get_weather")
        .description("Get the current weather for a location")
        .param(
            ParamBuilder::new("location")
                .type_of("string")
                .description("The city and state, e.g. San Francisco, CA")
        )
        .param(
            ParamBuilder::new("unit")
                .type_of("string")
                .description("The temperature unit (celsius or fahrenheit)")
        )
        .required(vec!["location".to_string()]);

    // Create a provider with tools
    let tool_provider = LLMBuilder::default()
        .backend(LLMBackend::Mistral)
        .api_key(std::env::var("MISTRAL_API_KEY")?)
        .model("mistral-large-latest") // Use a model that supports function calling
        .function(weather_function)
        .build()?;

    let messages = vec![ChatMessage {
        role: ChatRole::User,
        message_type: MessageType::Text,
        content: "What's the weather like in Paris, France?".to_string(),
    }];

    let response = tool_provider.chat(&messages).await?;
    println!("Response: {}", response.text().unwrap_or_default());
    
    // Check if the model wants to use a tool
    if let Some(tool_calls) = response.tool_calls() {
        println!("Tool calls requested:");
        for tool_call in tool_calls {
            println!("  - Function: {}", tool_call.function.name);
            println!("    Arguments: {}", tool_call.function.arguments);
        }
    }

    // Example 6: Structured Output (JSON Mode)
    println!("\n=== Structured Output Example ===");
    
    // Create a schema for the response
    let schema_json = r#"{
        "name": "Recipe",
        "description": "A cooking recipe",
        "schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the recipe"
                },
                "ingredients": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "name": { "type": "string" },
                            "amount": { "type": "string" }
                        },
                        "required": ["name", "amount"]
                    }
                },
                "instructions": {
                    "type": "array",
                    "items": { "type": "string" }
                }
            },
            "required": ["name", "ingredients", "instructions"]
        }
    }"#;
    
    let schema: StructuredOutputFormat = serde_json::from_str(schema_json)?;
    
    // Create a provider with structured output
    let structured_provider = LLMBuilder::default()
        .backend(LLMBackend::Mistral)
        .api_key(std::env::var("MISTRAL_API_KEY")?)
        .model("mistral-large-latest")
        .schema(schema)
        .build()?;

    let messages = vec![ChatMessage {
        role: ChatRole::User,
        message_type: MessageType::Text,
        content: "Give me a simple recipe for chocolate chip cookies.".to_string(),
    }];

    println!("Generating structured recipe response...");
    let response = structured_provider.chat(&messages).await?;
    
    // Parse and pretty-print the structured response
    let response_text = response.text().unwrap_or_default();
    let recipe: Value = serde_json::from_str(&response_text)?;
    println!("Recipe JSON:");
    println!("{}", serde_json::to_string_pretty(&recipe)?);

    Ok(())
}