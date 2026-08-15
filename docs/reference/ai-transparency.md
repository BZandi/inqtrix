# AI transparency

## Scope

Two separate things are described here: how this project's own source and documentation were written, and how Inqtrix marks the content its AI features generate. The second part is the one that matters if you operate Inqtrix for other people, because the markers are what you can rely on when you have to show where a text came from.

This page is not legal advice and not a declaration of conformity with any regulation. It states what the software does.

## How this project is written

Parts of the code and documentation were drafted with generative AI assistance: [Claude Code](https://www.anthropic.com/) (Anthropic), [GitHub Copilot](https://github.com/features/copilot) (GitHub / Microsoft), and [ChatGPT](https://openai.com/chatgpt) (OpenAI).

The notice exists so readers know how the material came about. It creates no guarantee, warranty, or additional liability, and the `AGPL-3.0-only` terms apply unchanged; see [`LICENSE`](../../LICENSE) for the warranty disclaimer and the limitation of liability.

## How Inqtrix marks generated output

**In the interface.** The sign-in screen and **Settings -> Legal** both state that Inqtrix is an AI system and that answers, reports, and suggestions come from language models. Because a signed-in user never sees the sign-in screen again, every AI workspace also carries a standing line under its prompt composer.

The composer is the anchor on purpose. An empty-state notice would only reach someone starting from nothing: open a project that already holds runs, threads, or sessions — imported, shared, or simply revisited — and the empty state never renders, so the statement would never be shown. The composer is present in every workspace in every state, and the line stays put while you type rather than disappearing on the first keystroke.

Generated answer bodies additionally carry two DOM attributes, so a script, an extension, or an archiving tool can identify them without parsing prose:

| Attribute | Value |
|---|---|
| `data-ai-generated` | `true` |
| `data-ai-producer` | `Inqtrix` |

These sit on the chat answer, the knowledge answer, the research report body, the agent answer, and the agent canvas artifact.

**When content leaves the app.** A marker that only exists in the browser is useless once a file is on someone's disk, so exports and whole-answer copies carry the disclosure with them:

| Path | Marker |
|---|---|
| Copied answer (chat, knowledge, report, agent, artifact) | A disclosure line appended below the text |
| Markdown export and project export | Front-matter keys `ai_generated`, `ai_producer`, `ai_marker`, `ai_disclosure`, plus a visible line |
| Word export of a research report | Document core properties (`creator`, `description`, `keywords`), custom properties in `docProps/custom.xml`, and a line in the page footer |

**For machine discovery.** The `/health` endpoint carries an `ai_disclosure` block next to the existing `legal` block, holding the producer name, the marker token, and the disclosure sentences. See [Web server mode](../deployment/webserver-mode.md).

## What is deliberately not marked

Marking the wrong thing is its own kind of inaccuracy, so these surfaces stay clean:

- **Your own text.** Prompts, questions, and chat messages you wrote round-trip byte-exact.
- **Retrieved source excerpts and document diffs.** These are quotations from your own documents, not model output. Labelling them would misattribute your sources to a language model.
- **Editor documents, including their Word export.** The editor performs assistive editing on text you wrote: AI revisions arrive as tracked changes and you decide what to accept. The finished document is yours, and stamping it as machine-generated would be wrong. If you paste a generated answer into an editor document yourself, that is your editorial decision and the export follows your document, not the answer's origin.
- **Fragment copies.** Copying a table's source, a code block, or a text selection yields exactly those bytes. Appending a legal sentence to a five-word selection would corrupt the copy for its actual purpose.

## Operator duties

If you run Inqtrix for anyone beyond yourself in a professional setting, transparency obligations for the content your users produce attach to you as the operator, not to this repository. The markers above are the surfaces you can build on. Three practical consequences:

1. **Do not remove or suppress the notices.** They are plain interface strings and easy to delete in a fork; deleting them removes your own evidence.
2. **Keep the exported metadata intact.** If a conversion step in your pipeline drops document properties or front matter, the disclosure is gone by the time the file reaches a reader.
3. **Check whether your use needs more than the defaults.** Publishing generated text to the public, or to an audience that cannot tell it apart from human writing, is a different situation from an internal research desk. The obligation follows the content, not the tool.

Inqtrix is experimental software; see the status warning in the root [`README.md`](../../README.md) before deploying it anywhere that matters.

## Related docs

- [Security hardening](../deployment/security-hardening.md) — trust boundaries, hardening, and what stays the operator's responsibility.
- [Web server mode](../deployment/webserver-mode.md) — the `/health` payload and the rest of the HTTP surface.
- [React UI](../deployment/react-ui.md) — how the browser interface is built and served.
- [FAQ](faq.md) — shorter answers to narrower questions.
