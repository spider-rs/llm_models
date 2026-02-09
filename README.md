# llm_models_spider

[![Crates.io](https://img.shields.io/crates/v/llm_models_spider.svg)](https://crates.io/crates/llm_models_spider)
[![Documentation](https://docs.rs/llm_models_spider/badge.svg)](https://docs.rs/llm_models_spider)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Auto-updated registry of LLM model capabilities, arena rankings, and pricing.

## Features

- **Zero runtime dependencies** - Pure compile-time lookups
- **Multi-source data** - Aggregated from [OpenRouter](https://openrouter.ai), [LiteLLM](https://github.com/BerriAI/litellm), and [Chatbot Arena](https://huggingface.co/datasets/mathewhe/chatbot-arena-elo)
- **Rich model profiles** - Capabilities, pricing, context windows, and arena rankings
- **Auto-updated** - Model lists fetched daily via GitHub Actions
- **Fallback patterns** - Unknown models matched against common naming patterns

## Installation

```toml
[dependencies]
llm_models_spider = "0.1"
```

## Usage

### Capability Checks

```rust
use llm_models_spider::{supports_vision, supports_audio, supports_video, supports_pdf, is_text_only};

// Vision support
assert!(supports_vision("gpt-4o"));
assert!(supports_vision("claude-3-sonnet-20240229"));
assert!(supports_vision("google/gemini-2.0-flash"));
assert!(!supports_vision("gpt-3.5-turbo"));

// Audio, video, PDF support
assert!(supports_audio("gemini-2.0-flash"));
assert!(supports_video("gemini-2.0-flash"));
assert!(supports_pdf("claude-3-sonnet"));

// Text-only check
assert!(is_text_only("gpt-3.5-turbo"));
assert!(is_text_only("llama-3-70b"));
```

### Model Profiles

Get full model intelligence — capabilities, pricing, context window, and arena ranking:

```rust
use llm_models_spider::{model_profile, arena_rank};

if let Some(profile) = model_profile("gpt-4o") {
    println!("Vision: {}", profile.capabilities.vision);
    println!("Max input: {} tokens", profile.max_input_tokens);
    println!("Max output: {} tokens", profile.max_output_tokens);

    if let Some(cost) = profile.pricing.input_cost_per_m_tokens {
        println!("Input cost: ${:.2}/M tokens", cost);
    }
    if let Some(cost) = profile.pricing.output_cost_per_m_tokens {
        println!("Output cost: ${:.2}/M tokens", cost);
    }
}

// Arena ranking (0.0-100.0, higher is better)
if let Some(rank) = arena_rank("gpt-4o") {
    println!("Arena rank: {:.1}/100", rank);
}
```

### Full Capabilities Lookup

```rust
use llm_models_spider::ModelCapabilities;

if let Some(caps) = ModelCapabilities::lookup("gemini-2.0-flash") {
    println!("Vision: {}", caps.vision);
    println!("Audio: {}", caps.audio);
    println!("Video: {}", caps.video);
    println!("File/PDF: {}", caps.file);
}
```

### Direct Model Info Access

For advanced use cases, access the raw `MODEL_INFO` array directly:

```rust
use llm_models_spider::MODEL_INFO;

// MODEL_INFO is sorted by name — use binary search for fast lookups
let idx = MODEL_INFO.binary_search_by(|e| e.name.cmp("gpt-4o"));
if let Ok(idx) = idx {
    let info = &MODEL_INFO[idx];
    println!("{}: vision={}, audio={}, arena={}",
        info.name, info.supports_vision, info.supports_audio, info.arena_overall);
}
```

## Data Sources

Model data is aggregated from three sources:

| Source | Data Provided |
|--------|--------------|
| [OpenRouter API](https://openrouter.ai/api/v1/models) | Modalities (vision, audio), pricing, context length |
| [LiteLLM](https://github.com/BerriAI/litellm) | Vision, audio, video, PDF support, pricing, context window |
| [Chatbot Arena](https://huggingface.co/datasets/mathewhe/chatbot-arena-elo) | Arena Elo scores (normalized to 0-100) |

**Merge strategy:**
- Capabilities: OR across sources (if any source says vision=true, it's vision)
- Pricing: Prefer LiteLLM (more granular), fall back to OpenRouter
- Context window: Take max across sources
- Arena scores: From Chatbot Arena only

## Fallback Patterns

For models not yet in any data source, the library falls back to pattern matching:

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
