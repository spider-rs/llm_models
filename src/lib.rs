//! # llm_models_spider
//!
//! Auto-updated registry of LLM model capabilities.
//!
//! This crate provides zero-cost compile-time lookups for model capabilities
//! like vision support, audio input, etc. The model lists are automatically
//! updated via GitHub Actions by scraping OpenRouter's API.
//!
//! ## Usage
//!
//! ```rust
//! use llm_models_spider::{supports_vision, supports_audio, ModelCapabilities};
//!
//! // Quick checks
//! assert!(supports_vision("gpt-4o"));
//! assert!(supports_vision("claude-3-sonnet"));
//! assert!(!supports_vision("gpt-3.5-turbo"));
//!
//! // Get full capabilities
//! if let Some(caps) = ModelCapabilities::lookup("google/gemini-2.0-flash") {
//!     println!("Vision: {}, Audio: {}", caps.vision, caps.audio);
//! }
//! ```
//!
//! ## Auto-Updates
//!
//! Model capabilities are fetched from OpenRouter's API and committed to this
//! repo via scheduled GitHub Actions. New releases are published automatically
//! when the model list changes.

mod generated;

pub use generated::{VISION_MODELS, TEXT_ONLY_MODELS, AUDIO_MODELS, ALL_MODELS};

/// Model capabilities struct.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelCapabilities {
    /// Model supports image/vision input.
    pub vision: bool,
    /// Model supports audio input.
    pub audio: bool,
    /// Model supports video input.
    pub video: bool,
    /// Model supports file input.
    pub file: bool,
}

impl ModelCapabilities {
    /// Create capabilities for a vision model.
    pub const fn vision_only() -> Self {
        Self {
            vision: true,
            audio: false,
            video: false,
            file: false,
        }
    }

    /// Create capabilities for a text-only model.
    pub const fn text_only() -> Self {
        Self {
            vision: false,
            audio: false,
            video: false,
            file: false,
        }
    }

    /// Create capabilities for a full multimodal model.
    pub const fn full_multimodal() -> Self {
        Self {
            vision: true,
            audio: true,
            video: true,
            file: true,
        }
    }

    /// Lookup capabilities by model ID or name.
    ///
    /// Matches against the full model ID first, then tries partial matching.
    pub fn lookup(model: &str) -> Option<Self> {
        let lower = model.to_lowercase();

        // Check vision models first (more specific)
        if is_in_list(&lower, VISION_MODELS) {
            // Check if it's also an audio model
            let audio = is_in_list(&lower, AUDIO_MODELS);
            return Some(Self {
                vision: true,
                audio,
                video: audio, // Usually audio models also support video
                file: audio,
            });
        }

        // Check text-only models
        if is_in_list(&lower, TEXT_ONLY_MODELS) {
            return Some(Self::text_only());
        }

        // Fallback to pattern matching for unknown models
        None
    }
}

/// Check if a model supports vision/image input.
///
/// This function checks against the auto-generated list of vision models
/// from OpenRouter, then falls back to pattern matching for unknown models.
///
/// # Examples
///
/// ```rust
/// use llm_models_spider::supports_vision;
///
/// assert!(supports_vision("gpt-4o"));
/// assert!(supports_vision("claude-3-sonnet-20240229"));
/// assert!(supports_vision("google/gemini-2.0-flash"));
/// assert!(!supports_vision("gpt-3.5-turbo"));
/// ```
pub fn supports_vision(model: &str) -> bool {
    let lower = model.to_lowercase();

    // Check the generated list first
    if is_in_list(&lower, VISION_MODELS) {
        return true;
    }

    // Fallback to pattern matching for models not in the list
    supports_vision_by_pattern(&lower)
}

/// Check if a model supports audio input.
pub fn supports_audio(model: &str) -> bool {
    let lower = model.to_lowercase();
    is_in_list(&lower, AUDIO_MODELS)
}

/// Check if a model is text-only (no vision/audio).
pub fn is_text_only(model: &str) -> bool {
    let lower = model.to_lowercase();

    // If it's in vision or audio lists, it's not text-only
    if is_in_list(&lower, VISION_MODELS) || is_in_list(&lower, AUDIO_MODELS) {
        return false;
    }

    // If it's in the text-only list, it's definitely text-only
    if is_in_list(&lower, TEXT_ONLY_MODELS) {
        return true;
    }

    // Unknown model - assume text-only unless pattern suggests otherwise
    !supports_vision_by_pattern(&lower)
}

/// Check if a model string matches any entry in a list.
///
/// Uses substring matching for flexibility with model ID variations.
fn is_in_list(model: &str, list: &[&str]) -> bool {
    for entry in list {
        // Exact match
        if model == *entry {
            return true;
        }
        // Model contains the entry (e.g., "openai/gpt-4o" contains "gpt-4o")
        if model.contains(entry) {
            return true;
        }
        // Entry contains the model (for short names)
        if entry.contains(model) && model.len() >= 4 {
            return true;
        }
    }
    false
}

/// Fallback pattern matching for vision models not in the generated list.
///
/// This catches new models that haven't been added to OpenRouter yet.
fn supports_vision_by_pattern(model: &str) -> bool {
    const VISION_PATTERNS: &[&str] = &[
        // OpenAI
        "gpt-4o",
        "gpt-4-turbo",
        "gpt-4-vision",
        "o1",
        "o3",
        "o4",
        // Anthropic
        "claude-3",
        "claude-4",
        // Google
        "gemini-1.5",
        "gemini-2",
        "gemini-flash",
        "gemini-pro-vision",
        // Qwen
        "qwen2-vl",
        "qwen2.5-vl",
        "qwen-vl",
        "qwq",
        // Llama
        "llama-3.2-vision",
        // Generic indicators
        "-vision",
        "-vl-",
        "-vl:",
        "/vl-",
    ];

    for pattern in VISION_PATTERNS {
        if model.contains(pattern) {
            return true;
        }
    }

    // Check VL suffix
    model.ends_with("-vl") || model.ends_with(":vl") || model.ends_with("/vl")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_supports_vision_openai() {
        assert!(supports_vision("gpt-4o"));
        assert!(supports_vision("gpt-4o-mini"));
        assert!(supports_vision("openai/gpt-4o"));
        assert!(!supports_vision("gpt-3.5-turbo"));
    }

    #[test]
    fn test_supports_vision_anthropic() {
        assert!(supports_vision("claude-3-sonnet"));
        assert!(supports_vision("anthropic/claude-3-opus"));
        assert!(!supports_vision("claude-2"));
    }

    #[test]
    fn test_supports_vision_google() {
        assert!(supports_vision("gemini-2.0-flash"));
        assert!(supports_vision("google/gemini-1.5-pro"));
    }

    #[test]
    fn test_is_text_only() {
        assert!(is_text_only("gpt-3.5-turbo"));
        assert!(is_text_only("claude-2"));
        assert!(!is_text_only("gpt-4o"));
    }

    #[test]
    fn test_model_capabilities_lookup() {
        let caps = ModelCapabilities::lookup("gpt-4o");
        assert!(caps.is_some());
        assert!(caps.unwrap().vision);
    }
}
