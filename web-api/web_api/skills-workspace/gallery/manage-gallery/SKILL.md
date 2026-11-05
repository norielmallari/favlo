---
name: manage-gallery
description: Use when this chat is a gallery workspace and the person wants an image, a video, music, or speech, or wants to find, select, or describe a file in the folder.
---

This gallery workspace has one local folder. The filmstrip lists every image in it.

The default model is already selected. Call generate with a new plain file name. Do not call a set model tool unless the person names a model. In that same turn, call gallery_show_generating_media with that file name. One short sentence. Do not say you are generating. Do not paste the JSON. Never mention a task id. Do not call media_status.

When you receive <media_completed>, call gallery_select_media with that file name. One short sentence. Do not paste the file name, a link, or a task id.

When you receive <media_error> or <media_cancelled>, the tile is already gone. One short sentence. Do not retry unless the person asks.

gallery_find_media and gallery_search_media return file names. gallery_select_media changes the selected file. gallery_get_metadata reads one file. gallery_load_image reads one image. Do not paste JSON or a link.
