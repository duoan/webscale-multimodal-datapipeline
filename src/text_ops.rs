//! Text processing operators: HTML extraction
//!
//! Provides Rust-accelerated text operations:
//! - `html_extract_text`: Extract readable text from a single HTML string
//! - `html_extract_text_batch`: Extract readable text from multiple HTML strings (parallel)

use dom_smoothie::Readability;
use pyo3::prelude::*;
use rayon::prelude::*;
use std::panic::{self, AssertUnwindSafe};

// ============================================================================
// HTML Text Extraction
// ============================================================================

/// Extract readable text from HTML using dom_smoothie (Rust port of readability.js)
/// Uses catch_unwind to handle panics from the dom_smoothie library gracefully,
/// especially for malformed HTML or UTF-8 boundary issues with non-ASCII characters.
fn html_extract_text_core(html: &str) -> Option<(String, String)> {
    // Wrap in catch_unwind to handle panics from dom_smoothie
    // This is necessary because dom_smoothie can panic on certain malformed HTML
    // or when it tries to slice strings at invalid UTF-8 boundaries
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        let mut readability = Readability::new(html, None, None).ok()?;
        let article = readability.parse().ok()?;

        let title = article.title;
        let content = article.text_content.to_string();

        // Skip if content is empty or too short
        if content.trim().is_empty() || content.len() < 50 {
            return None;
        }

        Some((title, content))
    }));

    // If panic occurred, return None instead of propagating the panic
    match result {
        Ok(opt) => opt,
        Err(_) => None, // Panic occurred, return None gracefully
    }
}

/// Extract readable text from a single HTML string
/// Returns (title, text, text_length) or None if extraction fails
#[pyfunction]
pub fn html_extract_text(html: String) -> PyResult<Option<(String, String, usize)>> {
    match html_extract_text_core(&html) {
        Some((title, text)) => {
            let text_len = text.len();
            Ok(Some((title, text, text_len)))
        }
        None => Ok(None),
    }
}

/// Batch extract readable text from multiple HTML strings (parallel)
/// Returns Vec of (title, text, text_length) for successful extractions
#[pyfunction]
pub fn html_extract_text_batch(htmls: Vec<String>) -> PyResult<Vec<Option<(String, String, usize)>>> {
    let results: Vec<_> = htmls
        .into_par_iter()
        .map(|html| {
            html_extract_text_core(&html).map(|(title, text)| {
                let text_len = text.len();
                (title, text, text_len)
            })
        })
        .collect();
    Ok(results)
}
