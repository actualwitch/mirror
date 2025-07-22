//! Example demonstrating text-to-speech synthesis using OpenAI
//!
//! This example shows how to:
//! 1. Initialize the OpenAI text-to-speech provider
//! 2. Generate speech from text
//! 3. Save the audio output to a file

use mirror::builder::{LLMBackend, LLMBuilder};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Get API key from environment variable or use test key
    let api_key = std::env::var("OPENAI_API_KEY").unwrap_or("test_key".into());

    // Initialize OpenAI text-to-speech provider
    let tts = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(api_key)
        .model("tts-1")
        .voice("ash")
        .build()?;

    // Text to convert to speech
    let text = "Hello! This is an example of text-to-speech synthesis using OpenAI.";

    // Generate speech
    let audio_data = tts.speech(text).await?;

    // Save the audio to a file
    std::fs::write("output-speech.mp3", audio_data)?;

    println!("Audio file generated successfully: output-speech.mp3");
    Ok(())
}
