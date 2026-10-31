---
name: manage-gallery
description: Arranges this gallery and saves generated images into it. Use when working in a gallery workspace, generating an image, or changing its folders and files.
---

This session is the gallery. Paths are relative to its local folder. Do not use a host path, another workspace id, a folder row, or a cloud folder.

The turn names the open folder and, when an image is selected, that image. This folder means the open folder. This image means the selected image. A generated image is saved into the open folder unless the message names a different folder.

After `generate_image` finishes, call `gallery_write_image` with the `favlo://` URI as source and a path inside the open folder. Do not leave the image only as a chat link.

Use `gallery_list_folder` to list the open folder. Pass another relative path only when the message names one.

Use `gallery_delete_image` when the message asks to delete this image. The path is the selected image.

Use `gallery_move_image` to move an image. This image is the selected image. The destination is the folder the message names, or the parent when the message says to move it up.

Use `gallery_create_folder` to create a directory inside the open folder.

Use `gallery_rename_folder` to rename a directory. This folder is the open folder.

Use `gallery_move_folder` to move a directory inside the local folder.

Use `gallery_delete_folder` to delete a directory. This folder is the open folder. Delete it only when it is empty. The local folder itself cannot be deleted.

These tools cannot add or delete a folder row, and they cannot use a cloud folder.
