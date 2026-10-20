# rshlin/Handy fork

This fork of [cjpais/Handy](https://github.com/cjpais/Handy) adds one thing: batch transcription
on a shared [whisper.cpp](https://github.com/ggml-org/whisper.cpp) `whisper-server` (on this Mac,
`http://127.0.0.1:18457`, which keeps large-v3-turbo loaded), so Handy does not load its own copy
of the model. Upstream keeps Handy local-only (closed PR #886), so this stays a fork.

- Branch: `rshlin/remote-whisper`, rebased onto each upstream release tag.
- Remotes: `origin` = `rshlin/Handy`, `upstream` = `cjpais/Handy`.
- Bundle identifier unchanged (`com.pais.handy`), so settings and models are shared with a stock
  Handy install.

## What changes

All logic is in one new file, `src-tauri/src/managers/transcription/remote_whisper.rs` (with its
unit tests). It is a child module of `managers/transcription.rs`, so it can call Handy's private
output post-processing. Upstream files get three lines, all in
`src-tauri/src/managers/transcription.rs`, each tagged `// fork: rshlin/remote-whisper`:

```rust
mod remote_whisper; // fork: rshlin/remote-whisper
```

before `const STREAM_PERF_LOG_INTERVAL`;

```rust
        remote_whisper::preload_hook! { self } // fork: rshlin/remote-whisper
```

as the first line of `TranscriptionManager::initiate_model_load()`;

```rust
        remote_whisper::transcribe_hook! { self, audio } // fork: rshlin/remote-whisper
```

in `TranscriptionManager::transcribe()`, right after the `if audio.is_empty() { ... }` block.

The hooks are brace-delimited macros because rustfmt leaves `name! { .. }` untouched; a plain
`if .. { return; }` would be split over three lines by `cargo fmt`. They expand to:

- `preload_hook!`: `if skips_local_preload(tm) { return; }` — when the server is configured and
  `GET /` answers 200 within 1 s, recording starts without loading the local model.
- `transcribe_hook!`: `if let Some(text) = transcribe(tm, &audio) { return Ok(text); }`.

`remote_whisper::transcribe`:

1. Encodes the 16 kHz mono f32 samples to an in-memory 16-bit wav.
2. `POST <url>/inference` (multipart: `file`, `language`, `response_format=json`,
   `temperature=0.0`, `no_timestamps=true`, `prompt` = custom words joined by `, ` when there are
   any, `translate=true` when "Translate to English" is on). The language is Handy's selected
   language, `auto` when Handy says auto.
3. Runs Handy's own `post_process_transcription_text` on the reply (Chinese script conversion,
   filler-word removal, normalisation; custom-word fuzzy correction is skipped when the words went
   in as the prompt, as Handy does for its local whisper models), logs, and returns it.
4. On any error (server down, timeout, non-200, bad JSON) it logs a warning, loads the selected
   local model if preload was skipped, and returns `None`, so Handy's normal local path runs.
   Dictation never fails because the server is down.

Not covered: live preview (streaming models) has no local engine to stream with while remote is
on, so it falls back to batch, which goes to the server. Picking a model in the UI still loads it
locally (`commands/models.rs`). Recording still requires a downloaded local model, both because
upstream checks for one before opening the mic and because the fallback needs it.

## Config

`~/Library/Application Support/com.pais.handy/remote-whisper.json` (Handy's app data dir; the
portable data dir in a portable install):

```json
{"url": "http://127.0.0.1:18457", "timeout_secs": 60}
```

- No file, or no `url` → the feature is off and Handy behaves exactly like upstream.
- `timeout_secs` (default 60) bounds one request. The server is serial, so a long job from
  another client can make Handy wait up to this long before it falls back to local.
- `HANDY_REMOTE_WHISPER_URL` overrides `url` (and turns the feature on without a file).
- The file is read on each recording, so changes apply without a restart.

## Updates

Turn off update checks in Handy, or the updater offers upstream builds, which would replace the
fork: press Cmd+Shift+D to show the Debug settings, then switch off "Check for Updates" (setting
`update_checks_enabled`). The fork does not patch the updater. (Upstream also honours
`HANDY_DISABLE_UPDATER=1` in the environment, but a Finder-launched app does not see shell
variables, so the setting is the simpler switch.)

## Rebase onto a new upstream tag

```bash
scripts/fork/rebase.sh v0.9.9
```

It fetches `upstream` tags, rebases `rshlin/remote-whisper` onto the tag, prints the
upstream-file diff (should still be the three lines above) and runs the module tests. On a
conflict it lists the files and stops; re-apply the hook lines by hand (they only need
`initiate_model_load()` and `transcribe()` to exist with those names and the early-return
semantics), then `git add` and `git rebase --continue`. Push with
`git push --force-with-lease origin rshlin/remote-whisper`.

Tests alone: `cd src-tauri && cargo test --lib remote_whisper`.

## Build and install

```bash
scripts/fork/build.sh
```

`bun install`, then `bun run tauri build --target aarch64-apple-darwin --bundles app` with
updater artifacts disabled (they need upstream's signing key), re-signs the `.app` with the local
identity, quits a running Handy, replaces `/Applications/Handy.app` and prints the version.
`HANDY_SIGN_IDENTITY` overrides the identity name.

## Signing

macOS ties the microphone and Accessibility grants to the code signature. Ad-hoc signatures
change on every build, so the grants would be lost each time. Builds are signed with a
self-signed identity, "Handy Fork Local", in the login keychain; its certificate is stable, so
the designated requirement (and the grants) survive rebuilds.

One-time setup (already done on this Mac):

```bash
cat > cert.cnf <<'CNF'
[req]
distinguished_name = dn
prompt = no
x509_extensions = ext
[dn]
CN = Handy Fork Local
[ext]
basicConstraints = critical,CA:false
keyUsage = critical,digitalSignature
extendedKeyUsage = critical,codeSigning
subjectKeyIdentifier = hash
CNF
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 3650 -config cert.cnf
openssl pkcs12 -export -legacy -inkey key.pem -in cert.pem -name "Handy Fork Local" -out id.p12 -passout pass:TEMP
security import id.p12 -k ~/Library/Keychains/login.keychain-db -P TEMP -T /usr/bin/codesign
# Let codesign use the key without a dialog (asks for the login password in the terminal):
security set-key-partition-list -S apple-tool:,apple:,codesign: -s -l "Handy Fork Local" ~/Library/Keychains/login.keychain-db
rm key.pem id.p12
```

`security find-identity -p codesigning` lists it as `CSSMERR_TP_NOT_TRUSTED`; that is expected
for a self-signed certificate and does not stop `codesign` or TCC.
