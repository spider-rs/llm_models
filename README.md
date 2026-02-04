# llm_models_spider

[![Crates.io](https://img.shields.io/crates/v/llm_models_spider.svg)](https://crates.io/crates/llm_models_spider)
[![Documentation](https://docs.rs/llm_models_spider/badge.svg)](https://docs.rs/llm_models_spider)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Auto-updated registry of LLM model capabilities (vision, audio, etc.).

## Features

- **Zero runtime dependencies** - Pure compile-time lookups
- **Auto-updated** - Model lists are fetched daily from [OpenRouter](https://openrouter.ai) via GitHub Actions
- **Fallback patterns** - Unknown models are matched against common naming patterns
- **Multiple capabilities** - Vision, audio, video, and file input detection

## Installation

```toml
[dependencies]
llm_models_spider = "0.1"
```

## Usage

```rust
use llm_models_spider::{supports_vision, supports_audio, is_text_only, ModelCapabilities};

// Quick capability checks
assert!(supports_vision("gpt-4o"));
assert!(supports_vision("claude-3-sonnet-20240229"));
assert!(supports_vision("google/gemini-2.0-flash"));
assert!(!supports_vision("gpt-3.5-turbo"));

// Audio support
assert!(supports_audio("gemini-2.0-flash"));

// Text-only check
assert!(is_text_only("gpt-3.5-turbo"));
assert!(is_text_only("llama-3-70b"));

// Full capabilities lookup
if let Some(caps) = ModelCapabilities::lookup("gemini-2.0-flash") {
    println!("Vision: {}", caps.vision);  // true
    println!("Audio: {}", caps.audio);    // true
}
```

## How It Works

1. **GitHub Actions** runs daily to fetch the latest model data from OpenRouter's API
2. Models are categorized by their `input_modalities` field:
   - `["text", "image"]` → Vision model
   - `["text", "image", "audio"]` → Vision + Audio model
   - `["text"]` → Text-only model
3. The `src/generated.rs` file is updated and a new version is published to crates.io

## Data Source

Model capabilities are sourced from [OpenRouter's API](https://openrouter.ai/api/v1/models):
- Vision models: `input_modalities` contains `"image"`
- Audio models: `input_modalities` contains `"audio"`
- Text-only: `input_modalities` is `["text"]` only

## Fallback Patterns

For models not yet in OpenRouter, the library falls back to pattern matching:

| Pattern | Examples |
|---------|----------|
| `gpt-4o`, `gpt-4-turbo`, `gpt-4-vision` | OpenAI vision models |
| `claude-3`, `claude-4` | Anthropic multimodal |
| `gemini-1.5`, `gemini-2` | Google Gemini |
| `qwen2-vl`, `qwen-vl` | Alibaba Qwen VL |
| `-vision`, `-vl` suffix | Generic vision indicators |

## Manual Update

To manually update the model lists:

```bash
cargo run --bin update-models --features updater
```

## License

MIT License - see [LICENSE](LICENSE) for details.
