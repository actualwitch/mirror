// Import required builder types from llm
use mirror::builder::{LLMBackend, LLMBuilder};

/// Example demonstrating how to generate embeddings using Google's API
///
/// This example shows how to:
/// - Configure a Google LLM provider
/// - Generate embeddings for text input
/// - Access and display the resulting embedding vector
#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize the LLM builder with Google configuration
    let llm = LLMBuilder::new()
        .backend(LLMBackend::AzureOpenAI)
        .base_url("your base url here")
        .deployment_id("text-embedding-3-large")
        // Get API key from environment variable or use test key
        .api_key(std::env::var("AZURE_OPENAI_API_KEY").unwrap_or("your api key here".to_owned()))
        .api_version("2024-12-01-preview")
        // Use Azure OpenAI text embedding model
        .model("text-embedding-3-large")
        .embedding_dimensions(256)
        .build()?;

    // Generate embedding vector for sample text
    let vector = llm.embed(vec!["Hello world!".to_string()]).await?;

    // Print embedding statistics and data
    println!("Data: {:?}", &vector);

    Ok(())
}
