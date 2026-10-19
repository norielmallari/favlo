# Prefix cache

As-built SSD prefix store. The map is [../ARCHITECTURE.md](../ARCHITECTURE.md). What a pipeline puts in the regions, and whether an exact-length file can be sampled, is that pipeline's note.

`src/favlo/prefix.py` stores one sequence as token ids plus named regions under `~/.favlo/prefix/<model>/<checkpoint>/<hash>.bin`. The checkpoint id is the hex SHA-256 of `manifest.json`, or of `config.json` when the record has no manifest. The file name is the hex SHA-256 of the format, the model id, that digest, the cache tag, and the token ids. The store does not know a key/value layout. Each pipeline packs its own buffers into regions and copies a matching file back.

A prompt that extends the live tokens keeps the device. Any other prompt writes the live sequence, clears the device, and loads the longest stored prefix that pipeline accepts. A match reads the header. Region bytes are read when that prefix is installed. Regions start on a 16 KB page. The write replaces the destination after the header hash is in place. A region whose stored bytes are a prefix of the new blob is copied forward. One model and one checkpoint stay within 32 GiB. Past that, the oldest files are deleted and the file just written stays.
