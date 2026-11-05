---
name: manage-coding
description: Use when this chat is a coding workspace and the person wants a file generated into the project.
---

This coding workspace has one local root. Pass a new plain file name on every generate call. The finished file is written at that root under that name. Do not call media_status.

When you receive <media_completed>, that file name is the file in the root. One short sentence. Do not paste the file name, a link, or a task id.

When you receive <media_error> or <media_cancelled>, one short sentence. Do not retry unless the person asks.
