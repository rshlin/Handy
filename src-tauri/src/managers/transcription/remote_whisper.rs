//! Fork-only (rshlin/Handy): batch transcription on a shared whisper-server.
//!
//! Upstream Handy always transcribes with a model loaded in-process. This
//! module sends the recording to a whisper.cpp `whisper-server` instead when
//! `<app data dir>/remote-whisper.json` (or `HANDY_REMOTE_WHISPER_URL`) names
//! one, and falls back to the local model on any error. It is a child module
//! of `transcription` so it can reuse Handy's private output post-processing;
//! upstream files only carry the `mod` line and two hook calls (see FORK.md).

use super::{
    post_process_transcription_text, resolve_output_language_evidence, ModelStateEvent,
    TranscriptionManager,
};
use crate::settings::get_settings;
use anyhow::{anyhow, Result};
use log::{info, warn};
use serde::Deserialize;
use std::cell::Cell;
use std::io::Cursor;
use std::path::Path;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};
use tauri::Emitter;

const CONFIG_FILE: &str = "remote-whisper.json";
const URL_ENV: &str = "HANDY_REMOTE_WHISPER_URL";
const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
const HEALTH_TIMEOUT: Duration = Duration::from_secs(1);

#[derive(Debug, Clone, PartialEq)]
pub(super) struct Config {
    url: String,
    timeout: Duration,
}

#[derive(Deserialize)]
struct ConfigFile {
    url: Option<String>,
    timeout_secs: Option<u64>,
}

/// What the server is asked to do for one recording.
#[derive(Debug, Clone, PartialEq)]
struct Request {
    language: String,
    prompt: Option<String>,
    translate: bool,
}

// The hooks are brace-delimited macros so each stays one line in upstream
// files: rustfmt leaves `name! { .. }` alone but would split an `if` block.

/// `load_model_with_device` hook: return before loading the local model.
macro_rules! load_hook {
    ($tm:expr, $model_id:expr) => {
        if remote_whisper::skips_local_load($tm, $model_id) {
            return Ok(());
        }
    };
}
pub(super) use load_hook;

/// `transcribe` hook: return the remote text, or carry on with the local path.
macro_rules! transcribe_hook {
    ($tm:expr, $audio:expr) => {
        if let Some(text) = remote_whisper::transcribe($tm, &$audio) {
            return Ok(text);
        }
    };
}
pub(super) use transcribe_hook;

/// The local engine, as far as remote transcription needs it. A trait so the
/// hook logic can be tested without a running app.
trait LocalModel {
    /// Unload the local engine if it is loaded and idle: no load in flight, not
    /// leased to a live-preview stream, not taken out by a local transcription.
    /// Returns true if it unloaded.
    fn release_if_idle(&self) -> bool;
    /// Load the selected model if nothing is loaded (the fallback path).
    fn ensure_loaded(&self);
    /// A load of `model_id` was skipped: drop the engine the load would have
    /// replaced and tell the UI the selection changed.
    fn stand_down(&self, model_id: &str);
}

impl LocalModel for TranscriptionManager {
    fn release_if_idle(&self) -> bool {
        // Holding the loading flag's mutex keeps a load from starting while
        // we unload; same lock order (is_loading, then engine) as transcribe().
        let is_loading = self.is_loading.lock().unwrap();
        if *is_loading
            || self.active_stream_worker.load(Ordering::Acquire) != 0
            || self.active_engine_lease.load(Ordering::Acquire) != 0
            || self.lock_engine().is_none()
        {
            return false;
        }
        info!("Remote whisper is serving; unloading the local model");
        let unloaded = match self.unload_model() {
            Ok(()) => true,
            Err(e) => {
                warn!("Failed to unload the local model: {}", e);
                false
            }
        };
        drop(is_loading);
        unloaded
    }

    /// Wait for any load in flight, then load the selected model if nothing
    /// is loaded (preload was skipped or the engine was released).
    fn ensure_loaded(&self) {
        {
            let mut is_loading = self.is_loading.lock().unwrap();
            while *is_loading {
                is_loading = self.loading_condvar.wait(is_loading).unwrap();
            }
        }
        if self.is_model_loaded() {
            return;
        }
        // A load that starts between the wait and here holds the flag; the local
        // path waits for it, so only load when we own the flag.
        let model_id = get_settings(&self.app_handle).selected_model;
        if let Some(_guard) = self.try_start_loading() {
            if let Err(e) = fallback_load(|| self.load_model(&model_id)) {
                warn!("Local fallback could not load model '{}': {}", model_id, e);
            }
        }
    }

    fn stand_down(&self, model_id: &str) {
        // load_model_with_device drops the current engine before building the
        // new one; do the same so a switched-away model does not stay resident.
        if self.active_engine_lease.load(Ordering::Acquire) == 0 && self.lock_engine().is_some() {
            if let Err(e) = self.unload_model() {
                warn!("Failed to unload the local model: {}", e);
            }
        }
        // Same event upstream emits when it skips eager loading (unload set to
        // "Immediately"): the UI shows the model as selected, the tray refreshes.
        let _ = self.app_handle.emit(
            "model-state-changed",
            ModelStateEvent {
                event_type: "selection_changed".to_string(),
                model_id: Some(model_id.to_string()),
                model_name: self.model_manager.get_model_info(model_id).map(|m| m.name),
                error: None,
            },
        );
    }
}

thread_local! {
    /// Set while the fork's own fallback loads the local model, so the load
    /// hook lets that load through even if the server answers health checks.
    static FALLBACK_LOAD: Cell<bool> = const { Cell::new(false) };
}

/// Runs `load` with the load hook bypassed on this thread.
fn fallback_load<T>(load: impl FnOnce() -> T) -> T {
    struct Reset(bool);
    impl Drop for Reset {
        fn drop(&mut self) {
            FALLBACK_LOAD.with(|f| f.set(self.0));
        }
    }
    let _reset = Reset(FALLBACK_LOAD.with(|f| f.replace(true)));
    load()
}

/// Hook for `TranscriptionManager::load_model_with_device`: true when the
/// remote server is configured and healthy, so the local model is not loaded.
/// Every local load goes through there (startup and model selection, the
/// recording-start preload, the CLI), so nothing loads while the server is up.
pub(super) fn skips_local_load(tm: &TranscriptionManager, model_id: &str) -> bool {
    load_decision(tm, config(tm).as_ref(), model_id)
}

fn load_decision(local: &impl LocalModel, cfg: Option<&Config>, model_id: &str) -> bool {
    if FALLBACK_LOAD.with(Cell::get) {
        return false;
    }
    let Some(cfg) = cfg else {
        return false;
    };
    if !health(cfg) {
        warn!(
            "Remote whisper at {} is not healthy; loading the local model '{}'",
            cfg.url, model_id
        );
        return false;
    }
    info!(
        "Remote whisper at {} is healthy; not loading the local model '{}'",
        cfg.url, model_id
    );
    local.stand_down(model_id);
    true
}

/// Hook for `TranscriptionManager::transcribe`: the post-processed remote text,
/// or `None` to let Handy's local path run (feature off, or the server failed —
/// in which case the local model is loaded here first if it is not loaded).
pub(super) fn transcribe(tm: &TranscriptionManager, audio: &[f32]) -> Option<String> {
    let cfg = config(tm)?;
    let settings = get_settings(&tm.app_handle);
    let request = Request {
        language: request_language(&settings.selected_language),
        prompt: (!settings.custom_words.is_empty()).then(|| settings.custom_words.join(", ")),
        translate: settings.translate_to_english,
    };

    let started = Instant::now();
    let raw = remote_or_fallback(tm, &cfg, audio, &request)?;

    let hint = (request.language != "auto").then_some(request.language.as_str());
    let output_language = resolve_output_language_evidence(&settings, hint, &[], request.translate);
    let text = post_process_transcription_text(
        raw,
        &settings,
        request.prompt.is_some(),
        &output_language,
        &[],
    );
    info!(
        "Remote whisper transcribed {:.2}s of audio in {:.2}s via {}: {}",
        audio.len() as f64 / 16_000.0,
        started.elapsed().as_secs_f64(),
        cfg.url,
        crate::utils::redact_text(&text)
    );
    Some(text)
}

/// The raw remote text, releasing an idle local engine on success (each
/// transcribe() resets Handy's idle-unload timer, so the timer alone would
/// never free it); on failure, make sure the local model is loaded and return
/// `None`.
fn remote_or_fallback(
    local: &impl LocalModel,
    cfg: &Config,
    audio: &[f32],
    request: &Request,
) -> Option<String> {
    match transcribe_remote(cfg, audio, request) {
        Ok(text) => {
            local.release_if_idle();
            Some(text)
        }
        Err(e) => {
            warn!(
                "Remote whisper failed ({}); falling back to the local model",
                e
            );
            local.ensure_loaded();
            None
        }
    }
}

fn config(tm: &TranscriptionManager) -> Option<Config> {
    let file = crate::portable::app_data_dir(&tm.app_handle)
        .ok()
        .map(|dir| dir.join(CONFIG_FILE));
    read_config(file.as_deref(), std::env::var(URL_ENV).ok().as_deref())
}

fn read_config(path: Option<&Path>, env_url: Option<&str>) -> Option<Config> {
    let json = path
        .and_then(|p| std::fs::read_to_string(p).ok())
        .unwrap_or_else(|| "{}".to_string());
    parse_config(&json, env_url)
}

fn parse_config(json: &str, env_url: Option<&str>) -> Option<Config> {
    let file: ConfigFile = match serde_json::from_str(json) {
        Ok(file) => file,
        Err(e) => {
            warn!("Ignoring {}: {}", CONFIG_FILE, e);
            ConfigFile {
                url: None,
                timeout_secs: None,
            }
        }
    };
    let url = env_url
        .filter(|u| !u.trim().is_empty())
        .map(str::to_string)
        .or(file.url)?;
    let url = url.trim().trim_end_matches('/').to_string();
    if url.is_empty() {
        return None;
    }
    Some(Config {
        url,
        timeout: file
            .timeout_secs
            .map(Duration::from_secs)
            .unwrap_or(DEFAULT_TIMEOUT),
    })
}

/// whisper-server takes ISO codes or `auto`; Handy may store regional or
/// alias codes (`zh-Hans`, `nb`).
fn request_language(selected: &str) -> String {
    match selected.trim() {
        "" | "auto" => "auto".to_string(),
        other => crate::managers::model::canonical_language_code(other).to_string(),
    }
}

/// 16 kHz mono 16-bit PCM, the same conversion as `save_wav_file`.
fn encode_wav(samples: &[f32]) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut cursor = Cursor::new(Vec::with_capacity(44 + samples.len() * 2));
    {
        let mut writer = hound::WavWriter::new(&mut cursor, spec)?;
        for sample in samples {
            writer.write_sample((sample * i16::MAX as f32) as i16)?;
        }
        writer.finalize()?;
    }
    Ok(cursor.into_inner())
}

/// Handy's reqwest has no `multipart` feature (and Cargo.toml is upstream's),
/// so the form is built by hand.
fn multipart_body(boundary: &str, wav: &[u8], request: &Request) -> Vec<u8> {
    let mut fields = vec![
        ("language", request.language.as_str()),
        ("response_format", "json"),
        ("temperature", "0.0"),
        ("no_timestamps", "true"),
    ];
    if let Some(prompt) = &request.prompt {
        fields.push(("prompt", prompt));
    }
    if request.translate {
        fields.push(("translate", "true"));
    }

    let mut body = Vec::with_capacity(wav.len() + 1024);
    for (name, value) in fields {
        body.extend_from_slice(
            format!(
                "--{}\r\nContent-Disposition: form-data; name=\"{}\"\r\n\r\n{}\r\n",
                boundary, name, value
            )
            .as_bytes(),
        );
    }
    body.extend_from_slice(
        format!(
            "--{}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n",
            boundary
        )
        .as_bytes(),
    );
    body.extend_from_slice(wav);
    body.extend_from_slice(format!("\r\n--{}--\r\n", boundary).as_bytes());
    body
}

#[derive(Deserialize)]
struct InferenceReply {
    text: String,
}

/// Runs `fut` to completion on a fresh thread. `transcribe()` is called from
/// inside a tauri async task, where blocking on the runtime would panic.
fn block_on_thread<T, F>(fut: F) -> Result<T>
where
    T: Send + 'static,
    F: std::future::Future<Output = Result<T>> + Send + 'static,
{
    std::thread::spawn(move || tauri::async_runtime::block_on(fut))
        .join()
        .map_err(|_| anyhow!("remote whisper thread panicked"))?
}

fn health(cfg: &Config) -> bool {
    let url = format!("{}/", cfg.url);
    block_on_thread(async move {
        let client = reqwest::Client::builder().timeout(HEALTH_TIMEOUT).build()?;
        Ok(client.get(url).send().await?.status().is_success())
    })
    .unwrap_or(false)
}

fn transcribe_remote(cfg: &Config, audio: &[f32], request: &Request) -> Result<String> {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or_default();
    let boundary = format!("handy-remote-whisper-{:x}", nanos);
    let body = multipart_body(&boundary, &encode_wav(audio)?, request);
    let url = format!("{}/inference", cfg.url);
    let timeout = cfg.timeout;
    block_on_thread(async move {
        let client = reqwest::Client::builder().timeout(timeout).build()?;
        let response = client
            .post(url)
            .header(
                reqwest::header::CONTENT_TYPE,
                format!("multipart/form-data; boundary={}", boundary),
            )
            .body(body)
            .send()
            .await?;
        let status = response.status();
        if !status.is_success() {
            return Err(anyhow!("server answered {}", status));
        }
        let reply: InferenceReply = serde_json::from_slice(&response.bytes().await?)?;
        Ok(reply.text.trim().to_string())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{Read, Write};
    use std::net::{TcpListener, TcpStream};
    use std::sync::mpsc;
    use std::thread;

    fn cfg(url: String, timeout: Duration) -> Config {
        Config { url, timeout }
    }

    fn request() -> Request {
        Request {
            language: "ru".to_string(),
            prompt: Some("Handy, whisper".to_string()),
            translate: false,
        }
    }

    /// Reads one HTTP request (headers + Content-Length body) off the socket.
    fn read_request(stream: &mut TcpStream) -> Vec<u8> {
        let mut buf = Vec::new();
        let mut chunk = [0u8; 8192];
        loop {
            let n = stream.read(&mut chunk).unwrap();
            if n == 0 {
                return buf;
            }
            buf.extend_from_slice(&chunk[..n]);
            if let Some(end) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
                let head = String::from_utf8_lossy(&buf[..end]).to_lowercase();
                let len = head
                    .lines()
                    .find_map(|l| l.strip_prefix("content-length:"))
                    .map(|v| v.trim().parse::<usize>().unwrap())
                    .unwrap_or(0);
                while buf.len() < end + 4 + len {
                    let n = stream.read(&mut chunk).unwrap();
                    if n == 0 {
                        break;
                    }
                    buf.extend_from_slice(&chunk[..n]);
                }
                return buf;
            }
        }
    }

    /// A one-shot fake whisper-server: answers the first request with
    /// `status`/`body` after `delay` and hands back the raw request it got.
    fn fake_server(status: &str, body: &str, delay: Duration) -> (String, mpsc::Receiver<Vec<u8>>) {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let (tx, rx) = mpsc::channel();
        let status = status.to_string();
        let body = body.to_string();
        thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            let raw = read_request(&mut stream);
            let _ = tx.send(raw);
            thread::sleep(delay);
            let reply = format!(
                "HTTP/1.1 {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
                status,
                body.len(),
                body
            );
            let _ = stream.write_all(reply.as_bytes());
        });
        (url, rx)
    }

    fn dead_url() -> String {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        url
    }

    fn form_field(raw: &str, name: &str) -> Option<String> {
        let marker = format!("name=\"{}\"\r\n\r\n", name);
        let start = raw.find(&marker)? + marker.len();
        let end = raw[start..].find("\r\n--")? + start;
        Some(raw[start..end].to_string())
    }

    #[test]
    fn config_missing_file_is_off() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(read_config(Some(&dir.path().join(CONFIG_FILE)), None), None);
        assert_eq!(read_config(None, None), None);
    }

    #[test]
    fn config_file_sets_url_and_timeout() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(CONFIG_FILE);
        std::fs::write(
            &path,
            r#"{"url": "http://127.0.0.1:18457/", "timeout_secs": 5}"#,
        )
        .unwrap();
        assert_eq!(
            read_config(Some(&path), None),
            Some(cfg(
                "http://127.0.0.1:18457".to_string(),
                Duration::from_secs(5)
            ))
        );
    }

    #[test]
    fn config_without_url_or_invalid_is_off() {
        assert_eq!(parse_config(r#"{"timeout_secs": 5}"#, None), None);
        assert_eq!(parse_config(r#"{"url": "  "}"#, None), None);
        assert_eq!(parse_config("not json", None), None);
    }

    #[test]
    fn config_defaults_timeout() {
        assert_eq!(
            parse_config(r#"{"url": "http://h:1"}"#, None),
            Some(cfg("http://h:1".to_string(), DEFAULT_TIMEOUT))
        );
    }

    #[test]
    fn env_url_overrides_file_and_works_without_it() {
        assert_eq!(
            parse_config(
                r#"{"url": "http://h:1", "timeout_secs": 7}"#,
                Some("http://e:2")
            ),
            Some(cfg("http://e:2".to_string(), Duration::from_secs(7)))
        );
        assert_eq!(
            read_config(None, Some("http://e:2")),
            Some(cfg("http://e:2".to_string(), DEFAULT_TIMEOUT))
        );
        assert_eq!(read_config(None, Some("")), None);
    }

    #[test]
    fn request_language_passes_auto_and_canonicalises() {
        assert_eq!(request_language("auto"), "auto");
        assert_eq!(request_language(""), "auto");
        assert_eq!(request_language("ru"), "ru");
        assert_eq!(request_language("zh-Hans"), "zh");
        assert_eq!(request_language("nb"), "no");
    }

    #[test]
    fn wav_has_16k_mono_16bit_header_and_length() {
        let samples = vec![0.0f32, 0.5, -0.5, 1.0, -1.0];
        let wav = encode_wav(&samples).unwrap();
        assert_eq!(&wav[0..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(wav.len(), 44 + samples.len() * 2);
        let reader = hound::WavReader::new(Cursor::new(wav)).unwrap();
        let spec = reader.spec();
        assert_eq!(spec.channels, 1);
        assert_eq!(spec.sample_rate, 16_000);
        assert_eq!(spec.bits_per_sample, 16);
        assert_eq!(spec.sample_format, hound::SampleFormat::Int);
        let decoded: Vec<i16> = reader.into_samples::<i16>().map(|s| s.unwrap()).collect();
        assert_eq!(decoded, vec![0, 16383, -16383, i16::MAX, -i16::MAX]);
    }

    #[test]
    fn posts_inference_fields_and_trims_text() {
        let (url, rx) = fake_server("200 OK", r#"{"text": " Привет, мир\n"}"#, Duration::ZERO);
        let audio = vec![0.1f32; 1600];
        let text =
            transcribe_remote(&cfg(url, Duration::from_secs(5)), &audio, &request()).unwrap();
        assert_eq!(text, "Привет, мир");

        let raw = rx.recv().unwrap();
        let raw = String::from_utf8_lossy(&raw);
        assert!(
            raw.starts_with("POST /inference HTTP/1.1\r\n"),
            "{}",
            &raw[..40]
        );
        assert!(raw
            .to_lowercase()
            .contains("content-type: multipart/form-data; boundary="));
        assert_eq!(form_field(&raw, "language").as_deref(), Some("ru"));
        assert_eq!(form_field(&raw, "response_format").as_deref(), Some("json"));
        assert_eq!(form_field(&raw, "temperature").as_deref(), Some("0.0"));
        assert_eq!(form_field(&raw, "no_timestamps").as_deref(), Some("true"));
        assert_eq!(
            form_field(&raw, "prompt").as_deref(),
            Some("Handy, whisper")
        );
        assert_eq!(form_field(&raw, "translate"), None);
        assert!(raw.contains("name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav"));
        assert!(raw.contains("RIFF"));
    }

    #[test]
    fn omits_prompt_and_sends_translate_when_asked() {
        let (url, rx) = fake_server("200 OK", r#"{"text": "hello"}"#, Duration::ZERO);
        let request = Request {
            language: "auto".to_string(),
            prompt: None,
            translate: true,
        };
        transcribe_remote(&cfg(url, Duration::from_secs(5)), &[0.0; 160], &request).unwrap();
        let raw = rx.recv().unwrap();
        let raw = String::from_utf8_lossy(&raw);
        assert_eq!(form_field(&raw, "language").as_deref(), Some("auto"));
        assert_eq!(form_field(&raw, "prompt"), None);
        assert_eq!(form_field(&raw, "translate").as_deref(), Some("true"));
    }

    #[test]
    fn server_error_is_an_error() {
        let (url, _rx) = fake_server(
            "500 Internal Server Error",
            r#"{"error":"x"}"#,
            Duration::ZERO,
        );
        assert!(
            transcribe_remote(&cfg(url, Duration::from_secs(5)), &[0.0; 160], &request()).is_err()
        );
    }

    #[test]
    fn bad_json_is_an_error() {
        let (url, _rx) = fake_server("200 OK", "not json", Duration::ZERO);
        assert!(
            transcribe_remote(&cfg(url, Duration::from_secs(5)), &[0.0; 160], &request()).is_err()
        );
        let (url, _rx) = fake_server("200 OK", r#"{"error": "no text"}"#, Duration::ZERO);
        assert!(
            transcribe_remote(&cfg(url, Duration::from_secs(5)), &[0.0; 160], &request()).is_err()
        );
    }

    #[test]
    fn connection_refused_is_an_error() {
        assert!(transcribe_remote(
            &cfg(dead_url(), Duration::from_secs(5)),
            &[0.0; 160],
            &request()
        )
        .is_err());
    }

    #[test]
    fn slow_server_times_out() {
        let (url, _rx) = fake_server("200 OK", r#"{"text": "late"}"#, Duration::from_secs(3));
        let started = Instant::now();
        let result = transcribe_remote(
            &cfg(url, Duration::from_millis(300)),
            &[0.0; 160],
            &request(),
        );
        assert!(result.is_err());
        assert!(started.elapsed() < Duration::from_secs(2));
    }

    #[test]
    fn health_follows_server_status() {
        let (url, rx) = fake_server("200 OK", "ok", Duration::ZERO);
        assert!(health(&cfg(url, DEFAULT_TIMEOUT)));
        let raw = rx.recv().unwrap();
        assert!(String::from_utf8_lossy(&raw).starts_with("GET / HTTP/1.1\r\n"));

        let (url, _rx) = fake_server("503 Service Unavailable", "loading", Duration::ZERO);
        assert!(!health(&cfg(url, DEFAULT_TIMEOUT)));
        assert!(!health(&cfg(dead_url(), DEFAULT_TIMEOUT)));
    }

    /// A fake local engine. Like the real one, its `ensure_loaded` loads
    /// through the load hook (`load_decision`) with the fallback bypass set.
    #[derive(Default)]
    struct FakeLocal {
        loaded: std::sync::atomic::AtomicBool,
        busy: bool,
        hook_cfg: Option<Config>,
        releases: std::sync::atomic::AtomicUsize,
        loads: std::sync::atomic::AtomicUsize,
        stand_downs: std::sync::atomic::AtomicUsize,
    }

    impl FakeLocal {
        fn loaded(busy: bool) -> Self {
            let fake = FakeLocal {
                busy,
                ..Default::default()
            };
            fake.loaded.store(true, Ordering::SeqCst);
            fake
        }
        fn is_loaded(&self) -> bool {
            self.loaded.load(Ordering::SeqCst)
        }
        fn count(counter: &std::sync::atomic::AtomicUsize) -> usize {
            counter.load(Ordering::SeqCst)
        }
        /// What `load_model_with_device` does: the hook, then the load.
        fn load(&self, model_id: &str) {
            if !load_decision(self, self.hook_cfg.as_ref(), model_id) {
                self.loaded.store(true, Ordering::SeqCst);
                self.loads.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    impl LocalModel for FakeLocal {
        fn release_if_idle(&self) -> bool {
            if self.busy || !self.is_loaded() {
                return false;
            }
            self.loaded.store(false, Ordering::SeqCst);
            self.releases.fetch_add(1, Ordering::SeqCst);
            true
        }
        fn ensure_loaded(&self) {
            if !self.is_loaded() {
                fallback_load(|| self.load("m"));
            }
        }
        fn stand_down(&self, _model_id: &str) {
            self.loaded.store(false, Ordering::SeqCst);
            self.stand_downs.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn healthy_server_skips_the_local_load() {
        let (url, _rx) = fake_server("200 OK", "ok", Duration::ZERO);
        let local = FakeLocal {
            hook_cfg: Some(cfg(url, DEFAULT_TIMEOUT)),
            ..Default::default()
        };
        local.load("turbo");
        assert!(!local.is_loaded());
        assert_eq!(FakeLocal::count(&local.loads), 0);
        assert_eq!(FakeLocal::count(&local.stand_downs), 1);
    }

    #[test]
    fn server_down_or_feature_off_loads_locally() {
        let local = FakeLocal {
            hook_cfg: Some(cfg(dead_url(), DEFAULT_TIMEOUT)),
            ..Default::default()
        };
        local.load("turbo");
        assert!(local.is_loaded());
        assert_eq!(FakeLocal::count(&local.stand_downs), 0);

        let local = FakeLocal::default();
        local.load("turbo");
        assert!(local.is_loaded());
    }

    #[test]
    fn fallback_load_bypasses_the_hook_and_resets_after() {
        let (url, _rx) = fake_server("200 OK", "ok", Duration::ZERO);
        let healthy = cfg(url, DEFAULT_TIMEOUT);
        let local = FakeLocal::default();
        assert!(!fallback_load(|| load_decision(
            &local,
            Some(&healthy),
            "m"
        )));
        assert_eq!(FakeLocal::count(&local.stand_downs), 0);
        assert!(!FALLBACK_LOAD.with(Cell::get));
        // The bypass is per thread: another thread is not affected.
        fallback_load(|| {
            let other = thread::spawn(|| FALLBACK_LOAD.with(Cell::get))
                .join()
                .unwrap();
            assert!(!other);
        });
    }

    #[test]
    fn remote_success_unloads_an_idle_local_model() {
        let (url, _rx) = fake_server("200 OK", r#"{"text": " hi "}"#, Duration::ZERO);
        let local = FakeLocal::loaded(false);
        let text = remote_or_fallback(
            &local,
            &cfg(url, Duration::from_secs(5)),
            &[0.0; 160],
            &request(),
        );
        assert_eq!(text.as_deref(), Some("hi"));
        assert!(!local.is_loaded());
        assert_eq!(FakeLocal::count(&local.loads), 0);
    }

    #[test]
    fn remote_success_keeps_a_busy_local_model() {
        let (url, _rx) = fake_server("200 OK", r#"{"text": "hi"}"#, Duration::ZERO);
        let local = FakeLocal::loaded(true);
        remote_or_fallback(
            &local,
            &cfg(url, Duration::from_secs(5)),
            &[0.0; 160],
            &request(),
        );
        assert!(local.is_loaded());
    }

    #[test]
    fn remote_failure_loads_locally_even_while_health_checks_pass() {
        // The inference request fails, but the hook's health check would
        // still pass (e.g. the server errors on this one job): the fallback
        // must load anyway.
        let (health_url, _rx) = fake_server("200 OK", "ok", Duration::ZERO);
        let local = FakeLocal {
            hook_cfg: Some(cfg(health_url, DEFAULT_TIMEOUT)),
            ..Default::default()
        };
        let (url, _rx) = fake_server("500 Internal Server Error", "{}", Duration::ZERO);
        let text = remote_or_fallback(
            &local,
            &cfg(url, Duration::from_secs(5)),
            &[0.0; 160],
            &request(),
        );
        assert_eq!(text, None);
        assert!(local.is_loaded());
        assert_eq!(FakeLocal::count(&local.loads), 1);
        assert_eq!(FakeLocal::count(&local.stand_downs), 0);
    }

    #[test]
    fn works_from_inside_an_async_runtime() {
        // transcribe() is called from a tauri async task; blocking there must not panic.
        let (url, _rx) = fake_server("200 OK", r#"{"text": "ok"}"#, Duration::ZERO);
        let config = cfg(url, Duration::from_secs(5));
        let text = tauri::async_runtime::block_on(async move {
            transcribe_remote(&config, &[0.0; 160], &request())
        })
        .unwrap();
        assert_eq!(text, "ok");
    }
}
