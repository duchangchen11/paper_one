# Frozen protocol audit erratum

This addendum supplements the immutable `protocol_frozen.json` (SHA256
`f287026ee2927d6a4b14d7f54baeb95d0b80c5420c8cbc51e52a4d41b434cd9e`). Its
original file and checksum are retained so the hash recorded with the official
test metrics remains verifiable.

Before protocol freeze, a schema preflight called `np.load(...,
allow_pickle=False)` on the processed `test.npz` and indexed every archive key to
print its array shape. With NumPy's compressed archive reader, indexing an array
materializes/decompresses that array in memory. Therefore the test archive was
technically read before freeze; the shorthand statement “test data not read” is
not accurate.

The preflight output was limited to archive field names and shapes. No array
contents were printed, summarized, or used to make a modeling decision; no
model inference, prediction, metric calculation, training, checkpoint
selection, or hyperparameter change used the test split before freeze. The
three formal training runs used `--skip-test` and did not load `test.npz`.
Official test inference began only after the frozen protocol and checksum had
been written.

This is disclosed as a procedural deviation from a strict “do not access test
until freeze” rule, while distinguishing schema/shape access from outcome-based
test use. The DGB settings and selected checkpoints were fixed using the
validation runs before official test inference.
