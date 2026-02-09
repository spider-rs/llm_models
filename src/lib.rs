//! # llm_models_spider
//!
//! Auto-updated registry of LLM model capabilities, rankings, and pricing.
//!
//! This crate provides zero-cost compile-time lookups for model capabilities
//! like vision support, audio input, etc. The model lists are automatically
//! updated via GitHub Actions by scraping OpenRouter, LiteLLM, and Chatbot Arena.
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
//! ## Rich Model Profiles
//!
//! ```rust
//! use llm_models_spider::{model_profile, arena_rank};
//!
//! if let Some(profile) = model_profile("gpt-4o") {
//!     println!("Max input: {} tokens", profile.max_input_tokens);
//!     if let Some(cost) = profile.pricing.input_cost_per_m_tokens {
//!         println!("Input cost: ${}/M tokens", cost);
//!     }
//! }
//!
//! if let Some(rank) = arena_rank("gpt-4o") {
//!     println!("Arena rank: {:.1}/100", rank);
//! }
//! ```
//!
//! ## Auto-Updates
//!
//! Model data is fetched from OpenRouter, LiteLLM, and Chatbot Arena,
//! then committed to this repo via scheduled GitHub Actions. New releases
//! are published automatically when the data changes.

mod generated;

pub use generated::{VISION_MODELS, TEXT_ONLY_MODELS, AUDIO_MODELS, ALL_MODELS};
pub use generated::{ModelInfoEntry, MODEL_INFO};

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

/// Arena and task-specific rankings, normalized to 0.0-100.0.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelRanks {
    /// Overall arena score (0.0-100.0).
    pub overall: Option<f32>,
    /// Coding task score. Future: when category data available.
    pub coding: Option<f32>,
    /// Math task score. Future.
    pub math: Option<f32>,
    /// Hard prompts score. Future.
    pub hard_prompts: Option<f32>,
    /// Instruction following score. Future.
    pub instruction_following: Option<f32>,
    /// Vision task rank. Future.
    pub vision_rank: Option<f32>,
    /// Style control score. Future.
    pub style_control: Option<f32>,
}

/// Pricing in USD.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModelPricing {
    /// Cost per 1 million input tokens in USD.
    pub input_cost_per_m_tokens: Option<f32>,
    /// Cost per 1 million output tokens in USD.
    pub output_cost_per_m_tokens: Option<f32>,
}

/// Full model profile: capabilities, ranks, pricing, and context window.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelProfile {
    pub capabilities: ModelCapabilities,
    pub ranks: ModelRanks,
    pub pricing: ModelPricing,
    pub max_input_tokens: u32,
    pub max_output_tokens: u32,
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
    /// Merges data from MODEL_INFO, generated lists, and pattern matching
    /// to provide the most accurate capabilities.
    pub fn lookup(model: &str) -> Option<Self> {
        let lower = model.to_lowercase();

        // Start with MODEL_INFO data if available
        let info = lookup_model_info(&lower);

        // Check vision: MODEL_INFO OR VISION_MODELS list OR pattern
        let vision = info.map_or(false, |i| i.supports_vision)
            || is_in_list(&lower, VISION_MODELS)
            || supports_vision_by_pattern(&lower);

        // Check audio: MODEL_INFO OR AUDIO_MODELS list
        let audio = info.map_or(false, |i| i.supports_audio)
            || is_in_list(&lower, AUDIO_MODELS);

        // Video and PDF come only from MODEL_INFO
        let video = info.map_or(false, |i| i.supports_video);
        let file = info.map_or(false, |i| i.supports_pdf);

        if info.is_some() || vision || audio || is_in_list(&lower, TEXT_ONLY_MODELS) {
            Some(Self { vision, audio, video, file })
        } else {
            None
        }
    }
}

/// Get the arena rank (0.0-100.0) for a model, if available.
///
/// Higher scores are better. Returns `None` if the model has no arena ranking.
pub fn arena_rank(model: &str) -> Option<f32> {
    let lower = model.to_lowercase();
    let info = lookup_model_info(&lower)?;
    if info.arena_overall == 0 {
        None
    } else {
        Some(info.arena_overall as f32 / 100.0)
    }
}

/// Get a full model profile including capabilities, pricing, ranks, and context window.
pub fn model_profile(model: &str) -> Option<ModelProfile> {
    let lower = model.to_lowercase();
    let info = lookup_model_info(&lower)?;

    let capabilities = ModelCapabilities {
        vision: info.supports_vision,
        audio: info.supports_audio,
        video: info.supports_video,
        file: info.supports_pdf,
    };

    let ranks = ModelRanks {
        overall: if info.arena_overall > 0 {
            Some(info.arena_overall as f32 / 100.0)
        } else {
            None
        },
        coding: None,
        math: None,
        hard_prompts: None,
        instruction_following: None,
        vision_rank: None,
        style_control: None,
    };

    let pricing = ModelPricing {
        input_cost_per_m_tokens: if info.cost_input_x1000 > 0 {
            Some(info.cost_input_x1000 as f32 / 1000.0)
        } else {
            None
        },
        output_cost_per_m_tokens: if info.cost_output_x1000 > 0 {
            Some(info.cost_output_x1000 as f32 / 1000.0)
        } else {
            None
        },
    };

    Some(ModelProfile {
        capabilities,
        ranks,
        pricing,
        max_input_tokens: info.max_input_tokens,
        max_output_tokens: info.max_output_tokens,
    })
}

/// Check if a model supports video input.
pub fn supports_video(model: &str) -> bool {
    let lower = model.to_lowercase();
    if let Some(info) = lookup_model_info(&lower) {
        return info.supports_video;
    }
    false
}

/// Check if a model supports PDF/file input.
pub fn supports_pdf(model: &str) -> bool {
    let lower = model.to_lowercase();
    if let Some(info) = lookup_model_info(&lower) {
        return info.supports_pdf;
    }
    false
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

    // Check MODEL_INFO first
    if let Some(info) = lookup_model_info(&lower) {
        if info.supports_vision {
            return true;
        }
        // MODEL_INFO may have incomplete vision data from merge; check other sources
    }

    // Check the generated VISION_MODELS list
    if is_in_list(&lower, VISION_MODELS) {
        return true;
    }

    // Fallback to pattern matching for models not in any list
    supports_vision_by_pattern(&lower)
}

/// Check if a model supports audio input.
pub fn supports_audio(model: &str) -> bool {
    let lower = model.to_lowercase();

    if let Some(info) = lookup_model_info(&lower) {
        if info.supports_audio {
            return true;
        }
        // MODEL_INFO may have incomplete audio data from merge; check other sources
    }

    is_in_list(&lower, AUDIO_MODELS)
}

/// Check if a model is text-only (no vision/audio).
pub fn is_text_only(model: &str) -> bool {
    !supports_vision(model) && !supports_audio(model)
}

/// Lookup a model in MODEL_INFO using binary search with substring fallback.
fn lookup_model_info(model: &str) -> Option<&'static ModelInfoEntry> {
    // Try binary search first (exact match)
    if let Ok(idx) = MODEL_INFO.binary_search_by(|entry| entry.name.cmp(model)) {
        return Some(&MODEL_INFO[idx]);
    }

    // Try stripping provider prefix and doing binary search (e.g. "openai/gpt-4o" -> "gpt-4o")
    if let Some(pos) = model.rfind('/') {
        let short = &model[pos + 1..];
        if let Ok(idx) = MODEL_INFO.binary_search_by(|entry| entry.name.cmp(short)) {
            return Some(&MODEL_INFO[idx]);
        }
    }

    // Substring matching fallback — prefer longest match to avoid "gpt-4" matching before "gpt-4o"
    let mut best: Option<&'static ModelInfoEntry> = None;
    let mut best_len = 0;

    for entry in MODEL_INFO {
        // Model contains the entry name (e.g., "openai/gpt-4o" contains "gpt-4o")
        if model.contains(entry.name) && entry.name.len() > best_len {
            best = Some(entry);
            best_len = entry.name.len();
        }
        // Entry name contains the model (for short names)
        if entry.name.contains(model) && model.len() >= 4 {
            return Some(entry);
        }
    }

    best
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

    #[test]
    fn test_arena_rank_known_models() {
        // Top models should have high ranks if they have arena data
        if let Some(rank) = arena_rank("gpt-4o") {
            assert!(rank > 50.0 && rank <= 100.0);
        }
    }

    #[test]
    fn test_model_profile_has_pricing() {
        if let Some(profile) = model_profile("gpt-4o") {
            assert!(profile.pricing.input_cost_per_m_tokens.is_some());
            assert!(profile.pricing.output_cost_per_m_tokens.is_some());
        }
    }

    #[test]
    fn test_model_profile_has_context_window() {
        if let Some(profile) = model_profile("gpt-4o") {
            assert!(profile.max_input_tokens > 0);
        }
    }

    #[test]
    fn test_supports_video_and_pdf() {
        // Gemini models typically support video
        assert!(supports_video("gemini-2.5-pro") || supports_video("gemini-2.0-flash"));
    }

    #[test]
    fn test_lookup_model_info_binary_search() {
        // Exact match should work
        let info = lookup_model_info("gpt-4o");
        assert!(info.is_some());
    }

    #[test]
    fn test_lookup_model_info_substring() {
        // Provider-prefixed model should match via substring
        let info = lookup_model_info("openai/gpt-4o");
        assert!(info.is_some());
    }

    #[test]
    fn test_model_info_sorted() {
        // Verify MODEL_INFO is sorted for binary search
        for window in MODEL_INFO.windows(2) {
            assert!(
                window[0].name <= window[1].name,
                "MODEL_INFO not sorted: {:?} > {:?}",
                window[0].name,
                window[1].name
            );
        }
    }

    #[test]
    fn test_supports_vision_qwen_vl() {
        // Qwen VL models should be detected as vision even if MODEL_INFO data is incomplete
        assert!(supports_vision("qwen2-vl-72b"));
        assert!(supports_vision("qwen2.5-vl-7b"));
        assert!(supports_vision("qwen-vl-max"));
        assert!(supports_vision("QWEN2-VL"));
    }

    #[test]
    fn test_supports_vision_case_insensitive() {
        assert!(supports_vision("GPT-4O"));
        assert!(supports_vision("Claude-3-Sonnet"));
        assert!(supports_vision("Gemini-2.0-Flash"));
    }

    #[test]
    fn test_capabilities_merge_sources() {
        // Even if MODEL_INFO has vision=false for a VL model, the VISION_MODELS list should win
        let caps = ModelCapabilities::lookup("qwen2-vl-72b");
        assert!(caps.is_some());
        assert!(caps.unwrap().vision);
    }
}
