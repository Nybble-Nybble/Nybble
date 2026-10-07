"""Catalog of MCP connectors shown in the gallery.

auth types:
  oauth   - "Sign in with ..." button (simulated until real OAuth is wired up)
  api_key - one or more secret fields
  local   - runs on the user's machine; asks for a path or command
"""

MCPS = [
    {
        "id": "gmail", "name": "Gmail", "color": "#EA4335",
        "category": "Communication", "auth": "oauth",
        "description": "Read, search and draft email.",
        "scopes": ["Read your email", "Create drafts"],
    },
    {
        "id": "messages", "name": "Messages", "color": "#34C759",
        "category": "Communication", "auth": "local",
        "description": "Read iMessage and SMS history on your Mac.",
        "fields": [
            {"name": "db_path", "label": "Messages database path (optional)",
             "placeholder": "~/Library/Messages/chat.db", "required": False},
        ],
        "note": "Leave blank to use the default location. The app running the Nybble server needs Full Disk Access.",
    },
    {
        "id": "whatsapp", "name": "WhatsApp", "color": "#25D366",
        "category": "Communication", "auth": "local",
        "description": "Read WhatsApp chats from the desktop app on your Mac.",
        "fields": [
            {"name": "db_path", "label": "WhatsApp database path (optional)",
             "placeholder": "~/Library/Group Containers/group.net.whatsapp.WhatsApp.shared/ChatStorage.sqlite",
             "required": False},
        ],
        "note": "Leave blank to use the default location. Needs the WhatsApp desktop app, signed in.",
    },
    {
        "id": "slack", "name": "Slack", "color": "#4A154B",
        "category": "Communication", "auth": "oauth",
        "description": "Channels, threads and direct messages.",
        "scopes": ["Read channels and DMs", "Post messages"],
    },
    {
        "id": "discord", "name": "Discord", "color": "#5865F2",
        "category": "Communication", "auth": "api_key",
        "description": "Servers, channels and messages via a bot.",
        "fields": [{"name": "bot_token", "label": "Bot token", "secret": True}],
    },
    {
        "id": "google-calendar", "name": "Google Calendar", "color": "#1A73E8",
        "category": "Productivity", "auth": "oauth",
        "description": "Events, availability and scheduling.",
        "scopes": ["View your calendars", "Create events"],
    },
    {
        "id": "google-drive", "name": "Google Drive", "color": "#0F9D58",
        "category": "Productivity", "auth": "oauth",
        "description": "Search and read Docs, Sheets and files.",
        "scopes": ["View your files"],
    },
    {
        "id": "notion", "name": "Notion", "color": "#191919",
        "category": "Productivity", "auth": "oauth",
        "description": "Pages, databases and notes.",
        "scopes": ["Read pages you share", "Edit pages you share"],
    },
    {
        "id": "linear", "name": "Linear", "color": "#5E6AD2",
        "category": "Productivity", "auth": "oauth",
        "description": "Issues, projects and cycles.",
        "scopes": ["Read issues", "Create and update issues"],
    },
    {
        "id": "github", "name": "GitHub", "color": "#24292F",
        "category": "Developer", "auth": "oauth",
        "description": "Repositories, issues and pull requests.",
        "scopes": ["Read repositories", "Read and write issues and PRs"],
    },
    {
        "id": "filesystem", "name": "Filesystem", "color": "#8E8E93",
        "category": "Developer", "auth": "local",
        "description": "Read and write files in folders you choose.",
        "fields": [
            {"name": "root", "label": "Allowed folder",
             "placeholder": "~/Documents"},
        ],
    },
    {
        "id": "postgres", "name": "PostgreSQL", "color": "#336791",
        "category": "Developer", "auth": "api_key",
        "description": "Query a Postgres database read-only.",
        "fields": [
            {"name": "url", "label": "Connection string", "secret": True,
             "placeholder": "postgresql://user:pass@host:5432/db"},
        ],
    },
    {
        "id": "brave-search", "name": "Brave Search", "color": "#FB542B",
        "category": "Web", "auth": "api_key",
        "description": "Web and local search results.",
        "fields": [{"name": "api_key", "label": "API key", "secret": True}],
    },
    {
        "id": "fetch", "name": "Fetch", "color": "#7CCF8A",
        "category": "Web", "auth": "local",
        "description": "Fetch web pages and convert them to text.",
        "fields": [
            {"name": "user_agent", "label": "User agent (optional)",
             "placeholder": "Nybble/1.0", "required": False},
        ],
    },
    {
        "id": "spotify", "name": "Spotify", "color": "#1DB954",
        "category": "Media", "auth": "oauth",
        "description": "Listening history and playlists.",
        "scopes": ["Read listening history", "Read playlists"],
    },
]

MCPS_BY_ID = {m["id"]: m for m in MCPS}
