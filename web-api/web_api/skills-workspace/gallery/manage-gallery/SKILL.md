---
name: manage-gallery
description: Arranges this gallery and saves generated images into it. Use when working in a gallery workspace, generating an image, or changing its files.
---

This session is the gallery. Paths are relative to its local folder root. Do not use a host path, another workspace id, a folder row, a cloud folder, or a nested directory.

When the turn names the selected media, this image means that path. A generated image is saved at the folder root when none is selected, or beside the selected media only when that media is also at the root.

`generate_image` returns a task id and the file is not ready. Reply in that same turn. Do not call `media_status` on this turn. The chat shows the `favlo://` link when the file is ready, and the host writes that file into the gallery folder and filmstrip. Use `gallery_write_image` only when the person asks to save a chat link under a different root path.

When asked to fill the filmstrip: call `gallery_find_media` once, then `workspace_place` once in the same turn. Do not call find twice. Use `gallery_search_media` only when the request is about meaning. Find and search return path and kind only. Keep `preview` on slot `stage` with props `{"fit":"contain"}` unless the user asks to remove it. Put `filmstrip` on slot `rail` with props `{"title":"","items":[{"path":"..."}]}` using only paths from find or search. When the filmstrip has items, always pass `selection` as `{"kind":"file","path":"..."}` for one of those paths so the preview shows it. Do not describe the layout in prose.

Call `gallery_get_metadata` for one path when the user asks about a caption, title, tags, or other sidecar fields, or before an edit.

Call `gallery_load_image` for one path when the sidecar description is not enough and the model must see the pixels. Do not load an image only because it is selected.

Use `gallery_search_unlike_media` when the request wants media that does not match the meaning.

Use `gallery_delete_image` when the message asks to delete this image. The file moves to the Trash. The path is the selected media at the folder root.

Use `gallery_move_image` to rename an image at the folder root. This image is the selected media. The destination is another root file name.

These tools cannot add or delete a folder row, create directories, or use a cloud folder.

When the user asks to change a caption, title, tags, people, objects, place, or whether an image is not safe for work, call `gallery_update_metadata` with that image's relative path. Send only the fields they want changed. Omitted fields stay as they are.

When the user asks to change how the gallery looks, call `gallery_set_theme` with `favlo`, `rose-pine`, or `catppuccin`. Light, dark, and system stay their appearance setting.
