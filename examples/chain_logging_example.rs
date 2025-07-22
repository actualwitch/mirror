//! Multi-backend chain example with logging enabled.
//!
//! This shows how to:
//! 1. Initialise a concrete logger (`env_logger`) when the `logging` feature is
//!    active so that the crate’s `trace! / debug!` statements are printed.
//! 2. Register two different back-ends (OpenAI and Anthropic).
//! 3. Execute a two-step `MultiPromptChain` where the second step consumes the
//!    output of the first.
//!
//! To run:
//! ```bash
//! RUST_LOG=llm=trace \
//! cargo run --example chain_logging_example --features "logging cli"
//! ```
//! The `cli` feature transitively enables all back-ends (`full` feature).

use mirror::{
    builder::{LLMBackend, LLMBuilder},
    chain::{LLMRegistryBuilder, MultiChainStepBuilder, MultiChainStepMode, MultiPromptChain},
};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    #[cfg(feature = "logging")]
    env_logger::init();

    let openai_llm = LLMBuilder::new()
        .backend(LLMBackend::OpenAI)
        .api_key(std::env::var("OPENAI_API_KEY")?)
        .model("gpt-4o")
        .build()?;

    let anthropic_llm = LLMBuilder::new()
        .backend(LLMBackend::Anthropic)
        .api_key(std::env::var("ANTHROPIC_API_KEY")?)
        .model("claude-3-opus-20240229")
        .build()?;

    let registry = LLMRegistryBuilder::new()
        .register("openai", openai_llm)
        .register("anthropic", anthropic_llm)
        .build();

    let results = MultiPromptChain::new(&registry)
        .step(
            MultiChainStepBuilder::new(MultiChainStepMode::Chat)
                .provider_id("openai")
                .id("summary")
                .template("Please summarise the following text in one concise sentence:\n\nRust is a multi-paradigm, general-purpose programming language that emphasises performance, type safety and concurrency.")
                .temperature(0.3)
                .build()?,
        )
        .step(
            MultiChainStepBuilder::new(MultiChainStepMode::Chat)
                .provider_id("anthropic")
                .id("evaluation")
                .template("Here is a summary: {{summary}}\n\nPlease critique its accuracy and completeness in less than 100 words.")
                .temperature(0.7)
                .build()?,
        )
        .run()
        .await?;

    println!("\nChain results:\n{results:#?}");

    Ok(())
}
