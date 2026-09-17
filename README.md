# Nybble importer

A portable, local importer for the **past behavior** half of the Nybble plan:
one-time extraction of everything a person has already written, said, saved, and
liked, normalized into a single record format for model personalization.

It runs on the user's own machine, reads only what they approve, and uploads
nothing unless they separately say so.

```
pip install -e .

nybble scan        # what personal data exists on this machine
nybble consent     # approve sources  (--yes-all approves everything at once)
nybble collect     # read them into ~/.nybble/records/*.jsonl
nybble taste       # infer what this person actually likes
nybble bundle      # flatten into one JSONL for fine-tuning
nybble revoke      # delete the data and the approvals
```

Zero required dependencies. Optional extras widen coverage:
`pip install 'nybble[formats,audio,llm,ocr]'`.

---

## What it collects

59 sources, mapped to the six past-behavior categories in the plan doc. Run
`nybble sources` for the catalog with descriptions, or `nybble sources --exports`
for click-by-click instructions on requesting each platform export.

`local` = already on disk. `export` = the user must request a data download from
the platform first. `mount` = an attached device.

### 1. Essays / content written by the user

| id | source | how | platform |
|---|---|---|---|
| `local_documents` | Documents, essays, drafts (.md .txt .docx .odt .rtf .pdf .tex) | local | all |
| `obsidian` | Obsidian vaults | local | all |
| `logseq` | Logseq graphs | local | all |
| `apple_notes` | Apple Notes | local | macOS |
| `bear` | Bear notes | local | macOS |
| `dayone` | Day One journal | local | macOS |
| `joplin` | Joplin notes | local | all |
| `notion_export` | Notion workspace export | export | all |
| `evernote_export` | Evernote (.enex) | export | all |
| `roam_export` | Roam Research | export | all |
| `jupyter` | Jupyter markdown cells | local | all |
| `latex` | LaTeX / academic writing | local | all |
| `blog_exports` | Substack, Medium, WordPress, Ghost | export | all |
| `git_authored` | Commit messages the user wrote | local | all |

### 2. Text messages

| id | source | how | platform |
|---|---|---|---|
| `imessage` | iMessage / SMS from `chat.db` | local | macOS |
| `android_sms_backup` | SMS Backup & Restore XML | export | all |
| `whatsapp_export` | WhatsApp per-chat exports | export | all |
| `telegram_export` | Telegram Desktop JSON | export | all |
| `signal_export` | Signal (plaintext backup only — see below) | export | all |
| `discord_export` | Discord data package | export | all |
| `slack_export` | Slack workspace/user export | export | all |
| `messenger_export` | Facebook Messenger | export | all |
| `instagram_dms` | Instagram DMs | export | all |
| `google_chat` | Google Chat / Hangouts | export | all |
| `email_local` | Apple Mail, Thunderbird, mbox, Maildir | local | all |
| `email_export` | Gmail Takeout .mbox | export | all |
| `github_comments` | Issue and PR comments | export | all |

### 3. Transcriptions of audio interactions

| id | source | how | platform |
|---|---|---|---|
| `existing_transcripts` | Zoom, Teams, Granola, Otter, Fathom, Fireflies, Rev (.vtt .srt .json .txt) | local | all |
| `voice_memos` | Voice Memos, transcribed locally | local | macOS |
| `local_audio` | Any local recording, transcribed locally | local | all |

Audio is transcribed on-device with faster-whisper and is never uploaded, even
when off-device taste inference is approved.

### 4. Writing of others that the user likes

| id | source | how | platform |
|---|---|---|---|
| `kindle_clippings` | Kindle highlights (`My Clippings.txt`) | mount | all |
| `apple_books` | Apple Books highlights and margin notes | local | macOS |
| `zotero` | Papers, notes, PDF annotations | local | all |
| `readwise_export` | Readwise, Pocket, Instapaper, Matter, Omnivore | export | all |
| `saved_pdfs` | PDFs and papers downloaded and kept | local | all |

### 5. Content of others that the user likes

| id | source | how | platform |
|---|---|---|---|
| `browser_bookmarks` | Chrome, Edge, Brave, Arc, Vivaldi, Firefox, Safari | local | all |
| `browser_history` | With visit counts (revisits ≈ attention) | local | all |
| `rss_subscriptions` | OPML from any reader | export | all |
| `goodreads` | Library, ratings, reviews | export | all |
| `youtube_takeout` | Watch history, likes, subscriptions | export | all |
| `spotify_export` | Streaming history, saved tracks | export | all |
| `podcasts` | Overcast, Pocket Casts, Apple Podcasts | export | all |
| `github_stars` | Starred repositories | export | all |
| `letterboxd` | Letterboxd / IMDb ratings | export | all |
| `screenshots` | OCR of kept screenshots | local | all |

### 6. Preferences saved by social media

| id | source | how | platform |
|---|---|---|---|
| `twitter_archive` | Tweets, likes, bookmarks, **X's inferred interest tags** | export | all |
| `reddit_export` | Posts, comments, saved, upvoted, subscriptions | export | all |
| `facebook_prefs` | Posts, likes, follows, **ad interests** | export | all |
| `instagram_prefs` | Captions, saved, follows, **topic interests** | export | all |
| `tiktok_export` | Watch history with timestamps, likes, favorites, searches | export | all |
| `linkedin_export` | Posts, comments, company follows | export | all |
| `mastodon_bluesky` | Fediverse posts and favorites | export | all |
| `google_activity` | Search history, My Activity, **ad topics** | export | all |
| `netflix_amazon` | Viewing history, orders, wishlist | export | all |
| `hackernews` | Comments and favorites | export | all |
| `stackexchange` | Answers, with vote scores | export | all |

### Beyond the plan doc

Three sources the doc does not list but that carry real signal:

| id | source | why |
|---|---|---|
| `calendar` | Event titles, attendees, notes | Establishes rhythm and who the user actually works with |
| `contacts` | Names and handles only | Turns phone numbers in message archives into people, so the model learns the user writes differently to their manager than their sibling |
| `shell_history` | Commands run | Dense record of tooling and working style |

`git_authored` is filed under category 1 for the same reason: commit messages
are the user explaining decisions to an audience, which is unusually clean
writing signal.

---

## Consent

Nothing is read until it is in the ledger at `~/.nybble/consent.json` — plain
JSON the user can open, audit, and edit.

`nybble scan` only checks whether files exist; it never opens them, so it is
safe to run before any approval.

`nybble consent` shows every source found, what exactly would be read, and where
it lives. Then either:

- **per group** — `y` all / `n` none / `s` choose one by one / `A` yes to
  everything and stop asking
- **`--yes-all`** — approve everything found in one confirmation

Two grants exist, and the first never implies the second:

| grant | means |
|---|---|
| collection | read these sources from this machine |
| `off_device_llm` | send excerpts to a hosted model for taste inference |

`--yes-all` covers collection only. The upload grant is always asked separately,
because it is the only step where data leaves the machine.

Other guarantees:

- **Redaction on by default.** Emails, phone numbers, card numbers (Luhn-checked
  so order numbers survive), and SSNs are masked. API keys, tokens, private
  keys, and password assignments are stripped *even with redaction disabled* —
  opting out of PII masking is the user's call, leaking credentials into a
  training corpus is not.
- **Consent expires** after 90 days, so a one-time yes does not become permanent.
- **`nybble revoke`** deletes the records, the manifest, the profile, and the
  ledger.
- **Signal is not attacked.** Its database is encrypted by design; Nybble parses
  a plaintext backup only if the user makes one themselves.

---

## Inferring what someone likes

Categories 4 and 5 ask for "writing/content of others that you like." Nothing on
a computer stores that, and no API returns it. What exists is a large pile of
other-authored material the user kept, marked, replayed, followed, or returned
to.

So `nybble taste` pools all of it and reasons over the pool:

1. **Pool** every non-self-authored record, plus anything carrying an engagement
   signal. Self-authored writing is excluded — that is voice signal, not taste.
2. **Weight** by how deliberate the keeping was. Highlighting a passage (1.00)
   outranks bookmarking (0.95), which outranks a page visited forty times
   (~0.75), which outranks one visited once (below threshold, dropped).
3. **Stratify** by source so a 40,000-row browser history cannot drown out 200
   Kindle highlights, and cap per book/domain so one source is not overweighted.
4. **Judge.** The sample goes to a model that labels each item liked/not and
   writes the profile. The weights are handed over as priors, explicitly *not*
   as answers — people save things out of obligation, for work, or to argue
   with, and the model is told to separate genuine affinity from incidental
   retention.

Output is `~/.nybble/taste_profile.json`: topics, authors, sources, style
affinities, dislikes, a prose summary, and a per-item verdict with reasoning.

Two backends:

- **hosted** — a Claude model. Needs the `off_device_llm` grant *and*
  `ANTHROPIC_API_KEY`. Only pooled other-authored excerpts are sent; the user's
  own documents, messages, and audio never are.
- **local** — weighted frequency statistics over the same pool. Cruder, and it
  says so in the output. Runs with no key and no network, which keeps the hosted
  path optional rather than required.

If the grant is given but no key is set, it falls back to local and says why.

---

## Output

```
~/.nybble/
  consent.json          what was approved, when, by whom
  records/<source>.jsonl one file per source
  manifest.json         per-source counts, timings, errors, redaction tallies
  taste_profile.json    inferred affinities
  bundle.jsonl          flattened corpus for fine-tuning
```

Every record is the same shape regardless of origin:

```json
{
  "id": "a3f2…", "source": "kindle_clippings", "kind": "highlight",
  "authorship": "other", "doc_category": "liked_writing", "sensitivity": "low",
  "text": "The legibility of a society provides the capacity for…",
  "title": "Seeing Like a State", "author": "James C. Scott",
  "created_at": "2024-06-03T09:12:33+00:00",
  "app": "Kindle", "signals": {"highlighted": true}
}
```

`authorship` is the field downstream tuning cares about most: `self` teaches
voice, `other` teaches taste, `mixed` is a two-sided artifact like a chat thread.

One file per source means a failed collector cannot corrupt the rest, and
dropping a source after the fact is `rm records/imessage.jsonl`. Records are
content-hashed after redaction, so re-running never duplicates rows.

---

## Design notes

**Portability.** Zero required dependencies, because the target is a stranger's
laptop with a system Python and no willingness to install a toolchain. `.docx`
and `.odt` are parsed with stdlib `zipfile` + `ElementTree`, HTML with
`html.parser`. Optional extras only widen coverage; nothing breaks without them.

**Failure isolation.** Every collector is a generator drained inside a try/except
that records the error in the manifest and moves on. One unparseable mailbox
must not cost the user the other fifty sources.

**Locked databases.** Live app databases (Chrome, Messages) are copied with
their `-wal`/`-shm` sidecars to a temp dir before reading, so a running browser
neither blocks the read nor yields stale rows.

**Schema drift.** Apple ships a different Notes schema most releases, so the
note body is recovered by decompressing the blob and pulling printable runs out
of the protobuf rather than pinning a generated schema. Similarly, iMessage text
moved into an `NSAttributedString` archive; both the old `text` column and the
new `attributedBody` are handled.

**Walk discipline.** `node_modules`, `.git`, caches, and system directories are
pruned in place during traversal, with a depth limit. This is the difference
between a two-minute and a two-hour run.

---

## Present behavior

The plan doc's second half — continuous learning from that day's data, scroll
attention, screen recordings and mouse clicks (LongNAP, Next Action Predictor) —
is deliberately out of scope here. This importer is the one-time past-behavior
load.

The schema already accommodates it: `signals` carries dwell and play counts, and
`kind: "activity"` exists for timestamped interaction events. The nightly
pipeline can append to the same `records/` directory, and `nybble collect
--since` supports incremental runs.

The closest available proxies today are `tiktok_export` (watch timestamps),
`browser_history` (visit counts), and `spotify_export` (play counts) — all
already collected.

---

## Tests

```
pip install pytest && pytest
```

40 tests. The fixtures write real SQLite databases, real zip archives, and real
export-format files rather than mocking the parsers, so the tests fail when a
format handler drifts. They cover each collector's format handling, the consent
gate (unapproved sources are never read; a key in the environment cannot
override a declined upload grant), redaction, dedupe, and a full
consent → collect → taste run.
