# llama-speed

Shows how fast the current model request is going, on pi's "Working" line above the chat input.

```
⠋ ⚡ 0.4s                            waiting for the first token
⠋ 🤖 ▓▓░░░░░░ 25% 8.4s              server is loading the model (llama.cpp only)
⠋ ⚡ ▓▓▓▓░░░░ 54% 3078 t/s 1.6s    server is processing the prompt (llama.cpp only)
⠋ 🔥 38.1 t/s 1.6s                  model is answering
⠋ Working                           request over, pi's own text back
```

Only one state is shown at a time. The line only carries rates and timings: token
totals, cache hit rate and cost are already in pi's footer, so they are not repeated here. Nothing
is written to the footer status row either, so other extensions keep their space there.

## What is measured

- **Prompt processing** comes from the server itself: llama.cpp sends `prompt_progress`
  (`total`, `cache`, `processed`, `time_ms`) when the request asks for it with
  `return_progress: true`, which this extension adds for local endpoints. The percentage, the rate
  and the elapsed time all come from that same sample, so they agree with each other; the time is
  what the server spent on the prompt, not the wall clock since the request was sent.
- **Generation** is measured in this client over a 3 second sliding window. Everything the model
  generates counts: prose, thinking and tool call arguments, which is where written code streams.
  For llama.cpp this reads the provider's own chunks; for other APIs it falls back to pi's stream
  events, and never both. Tool *results* are not generation; they land in the next request's
  prompt.
- Counts are taken from the stream while the request runs; the token and cache numbers `/speed`
  prints come from the server's own `usage` and are exact.
- **Model loading** is the third state. If a request has produced nothing for 1.2 s and no prompt
  progress either, the server's model list is read (`GET /models`). If it says the model is not in
  memory (`loading`, `unloaded`, `sleeping`, `downloading`), the extension then follows the server's
  model state stream (`GET /models/sse`), which is where llama.cpp publishes load progress from its
  `load_progress_callback`; `/models` alone only gives the state, not the percentage. That is where
  the bar and the percent come from, and `/speed` reports how long the model took to become ready.

## Which endpoints

llama.cpp is the focus and gets the most detail: only `openai-completions` models on a private or
LAN host are asked for `prompt_progress`, and the progress bar appears once the server actually
sends it, which is what identifies it as llama.cpp.

Every other endpoint (OpenAI, Copilot, Anthropic, OpenRouter, a LAN vLLM that ignores
`prompt_progress`) gets the same waiting and generation lines, counted from pi's own stream events.
Their request payloads are left untouched and no extra requests are made; there is simply no
prefill telemetry in those APIs to show. Rates for those endpoints are based on counted characters
rather than server token counts, so treat them as approximate; `/speed` reports the server's own
numbers once the request finishes.

## Commands

| Command | Effect |
|---|---|
| `/speed` | Details for the current and last request, plus which endpoints were recognised |
| `/speed off` / `/speed on` | Stop or resume writing to the working line |
| `/speed progress` | Toggle sending `return_progress` to local servers |
| `/speed load` | Toggle reading the server's model state (loading status and progress) |

## Install

From GitHub:

```bash
pi install git:github.com/pinkfluid/pi-llama-speed
```

Or over ssh, if you want the clone to use your key:

```bash
pi install git:git@github.com:pinkfluid/pi-llama-speed
```

Pin a release, so an update cannot move it under you:

```bash
pi install git:github.com/pinkfluid/pi-llama-speed@v0.1.0
```

Try it for one run without adding it to settings:

```bash
pi -e git:github.com/pinkfluid/pi-llama-speed
```

Restart pi after installing. Other package commands:

```bash
pi list                          # what is installed, and from where
pi update --extensions           # reconcile git and npm installs
pi remove git:github.com/pinkfluid/pi-llama-speed
```

A git install is cloned into pi's package directory. For development, install the checkout by path
instead: it is loaded in place, so an edit plus a pi restart is enough.

```bash
pi install ~/src/llama-speed
```

Or skip the package machinery and drop the single file into the extensions folder:
`~/.pi/agent/extensions/llama-speed.ts`.

## How it stays light and safe

It reads pi's own events (`before_provider_request`, `provider_stream_event`, `message_update`,
`message_end`, `agent_end`, `session_shutdown`) and makes only two requests of its own, both to the
local llama.cpp server: at most one `GET /models` per request, and one `GET /models/sse` while a
model is loading. Both open only while a request is silent, both close through an `AbortController`
on the first token, at message end or at shutdown, and every rejection is caught. The harness aborts
that stream mid-read and checks that nothing escapes as an unhandled rejection; it also checks that
the footer status row is never written and that cloud request payloads come back untouched.

## Development

The extension is a single TypeScript file; pi loads it through its own loader, so an edit only
needs a pi restart.

```bash
# types only come from the running pi installation; the link is gitignored
mkdir -p node_modules/@earendil-works
ln -sfn ~/.local/lib/node_modules/@earendil-works/pi-coding-agent node_modules/@earendil-works/pi-coding-agent

npm run check   # tsc --noEmit, strict
npm test        # harness: cloud, prefill, generation, tool calls, model loading, abort, off
                # also asserts no footer writes and no unhandled rejection after an aborted stream
LLAMA_SPEED_DEBUG=1 pi --print "hello"   # live log of what was counted vs reported by the server
```
