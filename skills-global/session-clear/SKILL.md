---
name: session-clear
description: Asks whether the discussion should be stored before clearing the session. Use when someone asks to clear, reset, or forget this conversation.
---

When someone asks to clear the session, ask first whether what was discussed should be stored in memory. If they decline, call `session_clear_session` and do not repeat anything from before that call. If they agree, follow the session-remember skill, then call `session_clear_session`.
