---
name: manage-gallery
description: Arranges this gallery and saves generated images into it. Use when working in a gallery workspace, generating an image, or changing its folders and files.
---

This session is the gallery. Paths are relative to its local folder. Do not use a host path, another workspace id, a folder row, or a cloud folder.

The turn names the selected media when one is selected. This image means that path. A generated image is saved beside the selected media, or at the folder root when none is selected.

After `generate_image` finishes, call `gallery_write_image` with the `favlo://` URI as source and a path inside the local folder. Do not leave the image only as a chat link.

When asked to fill the filmstrip: call `gallery_find_media` once, then `workspace_place` once in the same turn. Do not call find twice. Use `gallery_search_media` only when the request is about meaning. Keep `preview` on slot `stage` with props `{"fit":"contain"}` unless the user asks to remove it. Put `filmstrip` on slot `rail` with props `{"title":"","items":[{"path":"..."}]}` using only paths from find or search. When the filmstrip has items, always pass `selection` as `{"kind":"file","path":"..."}` for one of those paths so the preview shows it. Do not describe the layout in prose.

Before `gallery_search_unlike_media` or any folder mutate tool, call `gallery_load_folder_tools` once.

Use `gallery_search_unlike_media` when the request wants media that does not match the meaning.

Use `gallery_delete_image` when the message asks to delete this image. The path is the selected media.

Use `gallery_move_image` to move an image. This image is the selected media. The destination is the folder the message names, or the parent when the message says to move it up.

Use `gallery_create_folder` to create a directory inside the local folder.

Use `gallery_rename_folder` to rename a directory.

Use `gallery_move_folder` to move a directory inside the local folder.

Use `gallery_delete_folder` to delete a directory. Delete it only when it is empty. The local folder itself cannot be deleted.

These tools cannot add or delete a folder row, and they cannot use a cloud folder.

When the person asks to change a caption, title, tags, people, objects, place, or whether an image is not safe for work, call `gallery_update_metadata` with that image's relative path. Send only the fields they want changed. Omitted fields stay as they are.

When the person asks to change how the gallery looks, call `gallery_set_theme` with `favlo`, `rose-pine`, or `catppuccin`. Light, dark, and system stay their appearance setting.
