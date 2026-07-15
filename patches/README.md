# Dependency patches

## `@handlewithcare/prosemirror-suggest-changes@0.1.8`

The upstream command implementation probes one character beyond a ProseMirror
document when accepting or rejecting a block suggestion at the document end.
The patch changes the boundary check from `<=` to `<`, matching the surrounding
comment and `Node.textBetween` contract.

Keep the patch until an upstream release contains the equivalent fix. The
regression coverage is in `packages/editor-schema/tests/schema.test.ts` for both
block insertion and block deletion at the document boundary.
