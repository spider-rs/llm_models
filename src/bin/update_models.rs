//! Binary to fetch model capabilities from multiple sources and update generated.rs.
//!
//! Sources:
//! - OpenRouter API: modalities, pricing, context length
//! - LiteLLM: vision, audio, video, pdf support, pricing, context window
//! - Chatbot Arena: Elo rankings
//!
//! Run with: cargo run --bin update-models --features updater

use serde::Deserialize;
use std::collections::{HashMap, HashSet};
use std::fs;
use std::io::Write;

const OPENROUTER_API: &str = "https://openrouter.ai/api/v1/models";
const LITELLM_URL: &str = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";
const ARENA_ELO_URL: &str = "https://huggingface.co/datasets/mathewhe/chatbot-arena-elo/resolve/main/elo.csv";
const OUTPUT_FILE: &str = "src/generated.rs";

// ── OpenRouter types ──────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ApiResponse {
    data: Vec<ORModel>,
}

#[derive(Debug, Deserialize)]
struct ORModel {
    id: String,
    #[serde(default)]
    architecture: Option<Architecture>,
    #[serde(default)]
    pricing: Option<ORPricing>,
    #[serde(default)]
    context_length: Option<u64>,
    #[serde(default)]
    top_provider: Option<TopProvider>,
}

#[derive(Debug, Deserialize)]
struct Architecture {
    #[serde(default)]
    input_modalities: Vec<String>,
    #[serde(default)]
    output_modalities: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct ORPricing {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    completion: Option<String>,
}

#[derive(Debug, Deserialize)]
struct TopProvider {
    #[serde(default)]
    max_completion_tokens: Option<u64>,
}

// ── LiteLLM types ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct LiteLLMEntry {
    #[serde(default)]
    max_input_tokens: Option<u64>,
    #[serde(default)]
    max_output_tokens: Option<u64>,
    #[serde(default)]
    input_cost_per_token: Option<f64>,
    #[serde(default)]
    output_cost_per_token: Option<f64>,
    #[serde(default)]
    supports_vision: Option<bool>,
    #[serde(default)]
    supports_audio_input: Option<bool>,
    #[serde(default)]
    supports_video_input: Option<bool>,
    #[serde(default)]
    supports_pdf_input: Option<bool>,
    #[serde(default)]
    mode: Option<String>,
}

// ── Arena types ───────────────────────────────────────────────────────────────

#[derive(Debug)]
struct ArenaEntry {
    score: f64,
}

// ── Merged model ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
struct MergedModel {
    /// The canonical short name (lowercased, provider-stripped).
    name: String,
    supports_vision: bool,
    supports_audio: bool,
    supports_video: bool,
    supports_pdf: bool,
    max_input_tokens: u64,
    max_output_tokens: u64,
    /// Input cost per token in USD (raw).
    input_cost_per_token: Option<f64>,
    /// Output cost per token in USD (raw).
    output_cost_per_token: Option<f64>,
    /// Arena Elo normalized to 0-100.
    arena_score_normalized: Option<f64>,
}

// ── Fetch functions ───────────────────────────────────────────────────────────

fn fetch_openrouter() -> Result<Vec<MergedModel>, Box<dyn std::error::Error>> {
    println!("Fetching models from OpenRouter API...");
    let response = reqwest::blocking::get(OPENROUTER_API)?;
    let api_response: ApiResponse = response.json()?;
    println!("  Found {} models from OpenRouter", api_response.data.len());

    let mut models = Vec::new();

    for m in &api_response.data {
        let short_name = extract_short_name(&m.id);

        let (has_image, has_audio, has_video) = if let Some(arch) = &m.architecture {
            (
                arch.input_modalities.iter().any(|x| x == "image"),
                arch.input_modalities.iter().any(|x| x == "audio"),
                arch.input_modalities.iter().any(|x| x == "video"),
            )
        } else {
            (false, false, false)
        };

        let (input_cost, output_cost) = if let Some(pricing) = &m.pricing {
            (
                pricing.prompt.as_deref().and_then(|s| s.parse::<f64>().ok()),
                pricing.completion.as_deref().and_then(|s| s.parse::<f64>().ok()),
            )
        } else {
            (None, None)
        };

        let context_length = m.context_length.unwrap_or(0);
        let max_output = m
            .top_provider
            .as_ref()
            .and_then(|tp| tp.max_completion_tokens)
            .unwrap_or(0);

        models.push(MergedModel {
            name: short_name,
            supports_vision: has_image,
            supports_audio: has_audio,
            supports_video: has_video,
            supports_pdf: false, // OpenRouter doesn't track PDF
            max_input_tokens: context_length,
            max_output_tokens: max_output,
            input_cost_per_token: input_cost,
            output_cost_per_token: output_cost,
            arena_score_normalized: None,
        });
    }

    Ok(models)
}

fn fetch_litellm() -> Result<HashMap<String, MergedModel>, Box<dyn std::error::Error>> {
    println!("Fetching model data from LiteLLM...");
    let response = reqwest::blocking::get(LITELLM_URL)?;
    let raw: HashMap<String, serde_json::Value> = response.json()?;
    println!("  Found {} entries from LiteLLM", raw.len());

    let mut models = HashMap::new();

    for (key, value) in &raw {
        // Skip the sample_spec key and non-object entries
        if key == "sample_spec" || !value.is_object() {
            continue;
        }

        let entry: LiteLLMEntry = match serde_json::from_value(value.clone()) {
            Ok(e) => e,
            Err(_) => continue,
        };

        // Only include chat models
        if let Some(ref mode) = entry.mode {
            if mode != "chat" && mode != "completion" {
                continue;
            }
        }

        let short_name = extract_short_name(key);

        let merged = MergedModel {
            name: short_name.clone(),
            supports_vision: entry.supports_vision.unwrap_or(false),
            supports_audio: entry.supports_audio_input.unwrap_or(false),
            supports_video: entry.supports_video_input.unwrap_or(false),
            supports_pdf: entry.supports_pdf_input.unwrap_or(false),
            max_input_tokens: entry.max_input_tokens.unwrap_or(0),
            max_output_tokens: entry.max_output_tokens.unwrap_or(0),
            input_cost_per_token: entry.input_cost_per_token,
            output_cost_per_token: entry.output_cost_per_token,
            arena_score_normalized: None,
        };

        // Keep the entry with more data if duplicate
        models
            .entry(short_name)
            .and_modify(|existing: &mut MergedModel| {
                existing.supports_vision |= merged.supports_vision;
                existing.supports_audio |= merged.supports_audio;
                existing.supports_video |= merged.supports_video;
                existing.supports_pdf |= merged.supports_pdf;
                if merged.max_input_tokens > existing.max_input_tokens {
                    existing.max_input_tokens = merged.max_input_tokens;
                }
                if merged.max_output_tokens > existing.max_output_tokens {
                    existing.max_output_tokens = merged.max_output_tokens;
                }
                if existing.input_cost_per_token.is_none() {
                    existing.input_cost_per_token = merged.input_cost_per_token;
                }
                if existing.output_cost_per_token.is_none() {
                    existing.output_cost_per_token = merged.output_cost_per_token;
                }
            })
            .or_insert(merged);
    }

    Ok(models)
}

fn fetch_arena_elo() -> Result<HashMap<String, ArenaEntry>, Box<dyn std::error::Error>> {
    println!("Fetching Arena Elo rankings...");
    let response = reqwest::blocking::get(ARENA_ELO_URL)?;
    let text = response.text()?;

    let mut rdr = csv::Reader::from_reader(text.as_bytes());
    let headers = rdr.headers()?.clone();

    // Find column indices
    let model_col = headers
        .iter()
        .position(|h| h.eq_ignore_ascii_case("Model") || h.eq_ignore_ascii_case("key"))
        .unwrap_or(0);
    let score_col = headers
        .iter()
        .position(|h| {
            h.eq_ignore_ascii_case("Arena Score")
                || h.eq_ignore_ascii_case("rating")
                || h.eq_ignore_ascii_case("elo")
        })
        .unwrap_or(1);

    let mut raw_entries: Vec<(String, f64)> = Vec::new();

    for result in rdr.records() {
        let record = match result {
            Ok(r) => r,
            Err(_) => continue,
        };

        let model_name = match record.get(model_col) {
            Some(s) => s.trim().to_string(),
            None => continue,
        };

        let score: f64 = match record.get(score_col).and_then(|s| s.trim().parse().ok()) {
            Some(s) => s,
            None => continue,
        };

        raw_entries.push((model_name, score));
    }

    println!("  Found {} Arena entries", raw_entries.len());

    if raw_entries.is_empty() {
        return Ok(HashMap::new());
    }

    // Normalize to 0-100
    let min_score = raw_entries
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::INFINITY, f64::min);
    let max_score = raw_entries
        .iter()
        .map(|(_, s)| *s)
        .fold(f64::NEG_INFINITY, f64::max);
    let range = max_score - min_score;

    let mut arena_map = HashMap::new();
    for (name, score) in raw_entries {
        let normalized = if range > 0.0 {
            ((score - min_score) / range) * 100.0
        } else {
            50.0
        };
        let canonical = normalize_model_name(&name);
        arena_map
            .entry(canonical)
            .or_insert(ArenaEntry { score: normalized });
    }

    Ok(arena_map)
}

// ── Name normalization ────────────────────────────────────────────────────────

fn normalize_model_name(name: &str) -> String {
    let mut s = name.to_lowercase();

    // Strip provider prefixes (e.g. "openai/gpt-4o" -> "gpt-4o")
    if let Some(pos) = s.find('/') {
        s = s[pos + 1..].to_string();
    }

    // Strip parenthesized date suffixes like "(20241022)" or "(2025-03-26)"
    if let Some(paren_start) = s.rfind('(') {
        let before = s[..paren_start].trim_end().to_string();
        s = before;
    }

    // chatgpt- -> gpt-
    if s.starts_with("chatgpt-") {
        s = format!("gpt-{}", &s[8..]);
    }

    // Spaces -> hyphens
    s = s.replace(' ', "-");

    // Strip version suffixes like "-v1:0", "-v2:0"
    let patterns = ["-v1:0", "-v2:0", "-v3:0"];
    for pat in &patterns {
        if s.ends_with(pat) {
            s = s[..s.len() - pat.len()].to_string();
        }
    }

    s
}

fn extract_short_name(id: &str) -> String {
    if let Some(pos) = id.rfind('/') {
        id[pos + 1..].to_lowercase()
    } else {
        id.to_lowercase()
    }
}

// ── Merge ─────────────────────────────────────────────────────────────────────

fn merge_sources(
    openrouter: Vec<MergedModel>,
    litellm: HashMap<String, MergedModel>,
    arena: HashMap<String, ArenaEntry>,
) -> Vec<MergedModel> {
    let mut merged: HashMap<String, MergedModel> = HashMap::new();

    // Insert OpenRouter models
    for m in openrouter {
        merged
            .entry(m.name.clone())
            .and_modify(|existing| merge_into(existing, &m))
            .or_insert(m);
    }

    // Merge LiteLLM data
    for (_, m) in litellm {
        merged
            .entry(m.name.clone())
            .and_modify(|existing| {
                // Capabilities: OR across sources
                existing.supports_vision |= m.supports_vision;
                existing.supports_audio |= m.supports_audio;
                existing.supports_video |= m.supports_video;
                existing.supports_pdf |= m.supports_pdf;
                // Context window: take max
                if m.max_input_tokens > existing.max_input_tokens {
                    existing.max_input_tokens = m.max_input_tokens;
                }
                if m.max_output_tokens > existing.max_output_tokens {
                    existing.max_output_tokens = m.max_output_tokens;
                }
                // Pricing: prefer LiteLLM (more granular)
                if m.input_cost_per_token.is_some() {
                    existing.input_cost_per_token = m.input_cost_per_token;
                }
                if m.output_cost_per_token.is_some() {
                    existing.output_cost_per_token = m.output_cost_per_token;
                }
            })
            .or_insert(m);
    }

    // Apply Arena scores
    for (name, model) in merged.iter_mut() {
        let canonical = normalize_model_name(name);
        if let Some(entry) = arena.get(&canonical) {
            model.arena_score_normalized = Some(entry.score);
        }
    }

    let mut result: Vec<MergedModel> = merged.into_values().collect();
    result.sort_by(|a, b| a.name.cmp(&b.name));
    result
}

fn merge_into(existing: &mut MergedModel, other: &MergedModel) {
    existing.supports_vision |= other.supports_vision;
    existing.supports_audio |= other.supports_audio;
    existing.supports_video |= other.supports_video;
    existing.supports_pdf |= other.supports_pdf;
    if other.max_input_tokens > existing.max_input_tokens {
        existing.max_input_tokens = other.max_input_tokens;
    }
    if other.max_output_tokens > existing.max_output_tokens {
        existing.max_output_tokens = other.max_output_tokens;
    }
    if existing.input_cost_per_token.is_none() {
        existing.input_cost_per_token = other.input_cost_per_token;
    }
    if existing.output_cost_per_token.is_none() {
        existing.output_cost_per_token = other.output_cost_per_token;
    }
}

// ── Code generation ───────────────────────────────────────────────────────────

fn cost_to_x1000(cost_per_token: Option<f64>) -> u32 {
    match cost_per_token {
        Some(c) => {
            // cost_per_token * 1_000_000 = cost per M tokens
            // then * 1000 to get integer encoding
            let per_m = c * 1_000_000.0;
            let x1000 = per_m * 1000.0;
            x1000.round() as u32
        }
        None => 0,
    }
}

fn arena_to_u16(score: Option<f64>) -> u16 {
    match score {
        Some(s) => {
            // 0-100 scale, store as 0-10000 (two decimal places * 100)
            let val = (s * 100.0).round() as u16;
            val.min(10000)
        }
        None => 0,
    }
}

fn generate_rust_file(
    vision: &[String],
    text_only: &[String],
    audio: &[String],
    all_models: &[MergedModel],
    timestamp: &str,
) -> String {
    // Build ALL_MODELS as union of legacy lists
    let mut all_legacy: Vec<&str> = Vec::new();
    all_legacy.extend(vision.iter().map(|s| s.as_str()));
    all_legacy.extend(text_only.iter().map(|s| s.as_str()));
    all_legacy.extend(audio.iter().map(|s| s.as_str()));
    all_legacy.sort();
    all_legacy.dedup();

    let mut out = String::with_capacity(128_000);

    out.push_str(&format!(
        r#"//! Auto-generated model lists from OpenRouter, LiteLLM, and Chatbot Arena.
//!
//! DO NOT EDIT MANUALLY - This file is regenerated by the update-models binary.
//! Last updated: {}

/// Models that support vision/image input.
///
/// Source: OpenRouter API + LiteLLM (input_modalities contains "image")
pub const VISION_MODELS: &[&str] = &[
"#,
        timestamp
    ));
    for model in vision {
        out.push_str(&format!("    \"{}\",\n", model));
    }
    out.push_str("];\n\n");

    out.push_str(
        r#"/// Models that are text-only (no vision/audio support).
///
/// Source: OpenRouter API + LiteLLM (input_modalities is ["text"] only)
pub const TEXT_ONLY_MODELS: &[&str] = &[
"#,
    );
    for model in text_only {
        out.push_str(&format!("    \"{}\",\n", model));
    }
    out.push_str("];\n\n");

    out.push_str(
        r#"/// Models that support audio input.
///
/// Source: OpenRouter API + LiteLLM (input_modalities contains "audio")
pub const AUDIO_MODELS: &[&str] = &[
"#,
    );
    for model in audio {
        out.push_str(&format!("    \"{}\",\n", model));
    }
    out.push_str("];\n\n");

    out.push_str("/// All known models (union of all lists).\npub const ALL_MODELS: &[&str] = &[\n");
    for model in &all_legacy {
        out.push_str(&format!("    \"{}\",\n", model));
    }
    out.push_str("];\n\n");

    // New MODEL_INFO struct + array
    out.push_str(
        r#"/// Detailed model information entry.
///
/// Integer encoding avoids floating-point in const context.
/// Conversion to f32 happens at lookup time in lib.rs.
pub struct ModelInfoEntry {
    pub name: &'static str,
    pub supports_vision: bool,
    pub supports_audio: bool,
    pub supports_video: bool,
    pub supports_pdf: bool,
    pub max_input_tokens: u32,
    pub max_output_tokens: u32,
    /// Input cost USD per 1M tokens * 1000 (0 = unknown). e.g. 3000 = $3.00/M
    pub cost_input_x1000: u32,
    /// Output cost USD per 1M tokens * 1000 (0 = unknown).
    pub cost_output_x1000: u32,
    /// Arena overall score 0-10000 (two decimal places * 100). 0 = unranked.
    pub arena_overall: u16,
}

/// Sorted by name for binary search lookup.
pub const MODEL_INFO: &[ModelInfoEntry] = &[
"#,
    );

    for m in all_models {
        let input_x1000 = cost_to_x1000(m.input_cost_per_token);
        let output_x1000 = cost_to_x1000(m.output_cost_per_token);
        let arena = arena_to_u16(m.arena_score_normalized);
        let max_in = m.max_input_tokens.min(u32::MAX as u64) as u32;
        let max_out = m.max_output_tokens.min(u32::MAX as u64) as u32;

        out.push_str(&format!(
            "    ModelInfoEntry {{ name: \"{}\", supports_vision: {}, supports_audio: {}, supports_video: {}, supports_pdf: {}, max_input_tokens: {}, max_output_tokens: {}, cost_input_x1000: {}, cost_output_x1000: {}, arena_overall: {} }},\n",
            m.name,
            m.supports_vision,
            m.supports_audio,
            m.supports_video,
            m.supports_pdf,
            max_in,
            max_out,
            input_x1000,
            output_x1000,
            arena,
        ));
    }

    out.push_str("];\n");

    out
}

/// Simple timestamp without chrono dependency.
fn chrono_lite_now() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let secs = duration.as_secs();

    let days_since_epoch = secs / 86400;
    let years = 1970 + days_since_epoch / 365;
    let remaining_days = days_since_epoch % 365;
    let months = remaining_days / 30 + 1;
    let days = remaining_days % 30 + 1;

    format!("{}-{:02}-{:02}T00:00:00Z", years, months, days)
}

// ── Main ──────────────────────────────────────────────────────────────────────

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Fetch from all sources (resilient to individual failures)
    let openrouter = fetch_openrouter().unwrap_or_else(|e| {
        eprintln!("Warning: OpenRouter fetch failed: {}", e);
        vec![]
    });
    let litellm = fetch_litellm().unwrap_or_else(|e| {
        eprintln!("Warning: LiteLLM fetch failed: {}", e);
        HashMap::new()
    });
    let arena = fetch_arena_elo().unwrap_or_else(|e| {
        eprintln!("Warning: Arena Elo fetch failed: {}", e);
        HashMap::new()
    });

    // Fail only if ALL capability sources fail
    assert!(
        !openrouter.is_empty() || !litellm.is_empty(),
        "All capability sources failed — cannot generate model list"
    );

    // Merge all sources
    let all_models = merge_sources(openrouter, litellm, arena);

    // Build legacy lists from merged data
    let mut vision_set = HashSet::new();
    let mut text_only_set = HashSet::new();
    let mut audio_set = HashSet::new();

    for m in &all_models {
        if m.supports_vision {
            vision_set.insert(m.name.clone());
        }
        if m.supports_audio {
            audio_set.insert(m.name.clone());
        }
        if !m.supports_vision && !m.supports_audio && !m.supports_video {
            text_only_set.insert(m.name.clone());
        }
    }

    let mut vision: Vec<String> = vision_set.into_iter().collect();
    let mut text_only: Vec<String> = text_only_set.into_iter().collect();
    let mut audio: Vec<String> = audio_set.into_iter().collect();
    vision.sort();
    text_only.sort();
    audio.sort();

    println!("\nMerged results:");
    println!("  Vision models: {}", vision.len());
    println!("  Text-only models: {}", text_only.len());
    println!("  Audio models: {}", audio.len());
    println!("  Total MODEL_INFO entries: {}", all_models.len());

    let ranked_count = all_models
        .iter()
        .filter(|m| m.arena_score_normalized.is_some())
        .count();
    let priced_count = all_models
        .iter()
        .filter(|m| m.input_cost_per_token.is_some())
        .count();
    println!("  Models with arena scores: {}", ranked_count);
    println!("  Models with pricing: {}", priced_count);

    // Validation
    assert!(
        vision.len() >= 20,
        "Too few vision models: {}",
        vision.len()
    );
    assert!(
        text_only.len() >= 50,
        "Too few text-only models: {}",
        text_only.len()
    );
    assert!(audio.len() >= 5, "Too few audio models: {}", audio.len());
    assert!(
        all_models.len() >= 50,
        "Too few total models: {}",
        all_models.len()
    );

    // Validate arena scores
    for m in &all_models {
        let arena_val = arena_to_u16(m.arena_score_normalized);
        assert!(
            arena_val <= 10000,
            "Bad arena score for {}: {}",
            m.name,
            arena_val
        );
    }

    // Generate and write
    let timestamp = chrono_lite_now();
    let content = generate_rust_file(&vision, &text_only, &audio, &all_models, &timestamp);

    let mut file = fs::File::create(OUTPUT_FILE)?;
    file.write_all(content.as_bytes())?;

    println!("\nUpdated {}", OUTPUT_FILE);

    Ok(())
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_normalize_model_name() {
        assert_eq!(
            normalize_model_name("ChatGPT-4o-latest (2025-03-26)"),
            "gpt-4o-latest"
        );
        assert_eq!(
            normalize_model_name("anthropic/claude-3.5-sonnet"),
            "claude-3.5-sonnet"
        );
        assert_eq!(
            normalize_model_name("Claude 3.5 Sonnet (20241022)"),
            "claude-3.5-sonnet"
        );
        assert_eq!(
            normalize_model_name("Gemini-2.5-Pro"),
            "gemini-2.5-pro"
        );
        assert_eq!(
            normalize_model_name("DeepSeek-R1-0528"),
            "deepseek-r1-0528"
        );
        assert_eq!(normalize_model_name("openai/gpt-4o"), "gpt-4o");
        assert_eq!(normalize_model_name("google/gemini-2.0-flash"), "gemini-2.0-flash");
        assert_eq!(normalize_model_name("GPT-4o"), "gpt-4o");
        assert_eq!(normalize_model_name("o1-preview"), "o1-preview");
    }

    #[test]
    fn test_extract_short_name() {
        assert_eq!(extract_short_name("openai/gpt-4o"), "gpt-4o");
        assert_eq!(extract_short_name("gpt-4o"), "gpt-4o");
        assert_eq!(
            extract_short_name("anthropic/claude-3-opus"),
            "claude-3-opus"
        );
    }

    #[test]
    fn test_cost_to_x1000() {
        // $3.00 per M tokens = 0.000003 per token
        assert_eq!(cost_to_x1000(Some(0.000003)), 3000);
        // $15.00 per M tokens
        assert_eq!(cost_to_x1000(Some(0.000015)), 15000);
        // Unknown
        assert_eq!(cost_to_x1000(None), 0);
    }

    #[test]
    fn test_arena_to_u16() {
        assert_eq!(arena_to_u16(Some(95.20)), 9520);
        assert_eq!(arena_to_u16(Some(0.0)), 0);
        assert_eq!(arena_to_u16(Some(100.0)), 10000);
        assert_eq!(arena_to_u16(None), 0);
    }

    #[test]
    fn test_merge_or_capabilities() {
        let or_models = vec![MergedModel {
            name: "gpt-4o".into(),
            supports_vision: true,
            supports_audio: false,
            supports_video: false,
            supports_pdf: false,
            max_input_tokens: 128000,
            max_output_tokens: 4096,
            input_cost_per_token: Some(0.000005),
            output_cost_per_token: Some(0.000015),
            arena_score_normalized: None,
        }];

        let mut litellm_models = HashMap::new();
        litellm_models.insert(
            "gpt-4o".into(),
            MergedModel {
                name: "gpt-4o".into(),
                supports_vision: true,
                supports_audio: false,
                supports_video: false,
                supports_pdf: true,
                max_input_tokens: 128000,
                max_output_tokens: 16384,
                input_cost_per_token: Some(0.0000025),
                output_cost_per_token: Some(0.00001),
                arena_score_normalized: None,
            },
        );

        let merged = merge_sources(or_models, litellm_models, HashMap::new());
        let gpt4o = merged.iter().find(|m| m.name == "gpt-4o").unwrap();

        assert!(gpt4o.supports_vision);
        assert!(gpt4o.supports_pdf); // From LiteLLM
        assert_eq!(gpt4o.max_output_tokens, 16384); // Max of both
        // Pricing should prefer LiteLLM
        assert_eq!(gpt4o.input_cost_per_token, Some(0.0000025));
    }
}
