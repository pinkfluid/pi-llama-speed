// Synthetic pi events for the extension: no server, no network.
// Run with: node --experimental-strip-types test/harness.mjs

const handlers = new Map();
const commands = new Map();
const notices = [];
const statusCalls = [];
let working = "(default)";

const pi = {
  on(event, handler) {
    handlers.set(event, handler);
  },
  registerCommand(name, options) {
    commands.set(name, options);
  },
};

function makeCtx(model) {
  return {
    hasUI: true,
    mode: "tui",
    model,
    ui: {
      setStatus: (key, value) => statusCalls.push(`${key}=${value ?? "(cleared)"}`),
      setWorkingMessage: (message) => {
        working = message ?? "(default)";
      },
      notify: (message) => notices.push(message),
    },
  };
}

const mod = await import("../index.ts");
mod.default(pi);

const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));
const show = (label) => console.log(`${label.padEnd(30)} | ${working ?? "(default)"}`);
const request = (model, payload = { model: model.id, stream: true }) =>
  handlers.get("before_provider_request")({ type: "before_provider_request", payload }, makeCtx(model));
const chunk = (data) =>
  handlers.get("provider_stream_event")({ type: "provider_stream_event", data }, cloudCtx);
const delta = (text, type = "text_delta") =>
  handlers.get("message_update")({
    type: "message_update",
    message: { role: "assistant" },
    assistantMessageEvent: { type, delta: text },
  });
const end = (usage, stopReason = "stop") =>
  handlers.get("message_end")({ type: "message_end", message: { role: "assistant", usage, stopReason } }, cloudCtx);
const agentEnd = () => handlers.get("agent_end")({ type: "agent_end" }, cloudCtx);

const cloud = { provider: "openai", id: "gpt-5", api: "openai-completions", baseUrl: "https://api.openai.com/v1" };
const copilot = { provider: "github-copilot", id: "gpt-5-mini", api: "openai-responses", baseUrl: "https://api.githubcopilot.com" };
const anthropic = { provider: "anthropic", id: "claude", api: "anthropic-messages", baseUrl: "https://api.anthropic.com" };
const llama = { provider: "brain", id: "qwen-flash-next", api: "openai-completions", baseUrl: "https://lama:9080/v1" };
const cloudCtx = makeCtx(cloud);

// 1. cloud endpoint: payload untouched, but waiting and generation speed still show
const cloudPayload = request(cloud);
console.log("cloud payload untouched:", !JSON.stringify(cloudPayload).includes("return_progress"));
await sleep(250);
show("cloud, waiting");
for (let i = 0; i < 40; i++) {
  delta(`word ${i} `);
  await sleep(30);
}
await sleep(250);
show("cloud, generating");
end({ input: 500, output: 130, cacheRead: 900 });
show("cloud, after end");

// 2. Copilot (responses API) and Anthropic: same treatment, nothing injected
const copilotPayload = request(copilot);
const anthropicPayload = request(anthropic);
console.log(
  "copilot/anthropic untouched:",
  !JSON.stringify(copilotPayload).includes("return_progress") && !JSON.stringify(anthropicPayload).includes("return_progress"),
);
for (let i = 0; i < 20; i++) delta("token ");
await sleep(250);
show("copilot, generating");
end({ input: 100, output: 60, cacheRead: 0 });

// 3. first llama.cpp request: server samples bring the prefill line
const llamaPayload = request(llama);
console.log("llama payload injected:", JSON.stringify(llamaPayload).includes("return_progress"));
for (const p of [
  // the cached prefix arrives in bulk; llama.cpp still calls that sample 0%
  { total: 9000, cache: 6000, processed: 6000, time_ms: 200 },
  { total: 9000, cache: 6000, processed: 7300, time_ms: 1100 },
  { total: 9000, cache: 6000, processed: 8600, time_ms: 1600 },
]) {
  chunk({ prompt_progress: p });
  await sleep(70);
}
await sleep(250);
show("llama, processing prompt");
const promptLine = working;
console.log(
  "prefill rate is plausible:",
  /\d+(?:\.\d+)?k? t\/s/.test(promptLine) && Number((promptLine.match(/([\d.]+)k? t\/s/)||[])[1]) < 3000,
  `| ${promptLine}`
);
const streamed = [];
for (let i = 0; i < 60; i++) {
  const piece = `tok${String(i).padStart(2, "0")}`;
  streamed.push(piece);
  chunk({ choices: [{ delta: { content: piece } }] });
  delta(piece); // pi's own event for the same text: must not be counted twice
  await sleep(25);
}
chunk({ usage: { prompt_tokens: 9000, completion_tokens: 61 } });
await sleep(250);
show("llama, generating");
end({ input: 3000, output: 61, cacheRead: 6000 });
show("llama, after end");

// 3a2. a cached prompt starts at 0%, not at cache/total
request(llama);
chunk({ prompt_progress: { total: 3054, cache: 2575, processed: 2575, time_ms: 931 } });
await sleep(250);
show("llama, cached prefix applied");
const cacheStart = working;
chunk({ prompt_progress: { total: 3054, cache: 2575, processed: 3054, time_ms: 1531 } });
await sleep(250);
show("llama, rest computed");
const cacheDone = working;
end({ input: 479, output: 0, cacheRead: 2575 }, "aborted");
agentEnd();
console.log("cached prompt starts at 0%:", / 0% /.test(cacheStart), "|", cacheStart);
console.log("and reaches 100%:", / 100% /.test(cacheDone), "|", cacheDone);

// 3b. one slow sample: percent, rate and the server's own processing time
request(llama);
chunk({ prompt_progress: { total: 9450, cache: 0, processed: 1323, time_ms: 4200 } });
await sleep(250);
show("llama, slow prefill");
const slowLine = working;
end({ input: 0, output: 0, cacheRead: 0 }, "aborted");
agentEnd();
await commands.get("speed").handler("", makeCtx(llama));
const slowRecap = (notices.at(-1) || "").match(/last: (.*)$/m)?.[1];
console.log("prefill line:", slowLine === "⚡ ▓░░░░░░░ 14% 315 t/s 4.2s", "|", slowLine);
console.log("prefill recap:", slowRecap);

// 3c. cached prompt tokens arrive in bulk: they must not fake the rate
request(llama);
chunk({ prompt_progress: { total: 60000, cache: 49000, processed: 49000, time_ms: 200 } });
await sleep(60);
chunk({ prompt_progress: { total: 60000, cache: 49000, processed: 50500, time_ms: 3200 } });
await sleep(250);
show("llama, cache then compute");
const burstLine = working;
end({ input: 11000, output: 0, cacheRead: 49000 }, "aborted");
agentEnd();
const burstRate = Number((burstLine.match(/([\d.]+) t\/s/) || [])[1]);
console.log("cache burst line:", burstLine);
console.log("rate after a cache burst stays sane:", Number.isFinite(burstRate) && burstRate < 1000, `| ${burstRate} t/s`);

// 4. no double counting: an aborted run without usage estimates from the stream,
//    and must land on the same number with or without pi's own events.
const estimateOf = async (withPiEvents) => {
  request(llama);
  chunk({ prompt_progress: { total: 400, cache: 0, processed: 400, time_ms: 100 } });
  for (const piece of streamed.slice(0, 30)) {
    chunk({ choices: [{ delta: { content: piece } }] });
    if (withPiEvents) delta(piece);
    await sleep(15);
  }
  end({ input: 0, output: 0, cacheRead: 0 }, "aborted");
  agentEnd();
  await commands.get("speed").handler("", makeCtx(llama));
  // compare the token estimate only; the rate jitters with scheduling
  return (notices.at(-1) || "").match(/🔥 [^\n]+? (~?\d+(?:\.\d+)?k?) tok/)?.[1];
};
const providerOnly = await estimateOf(false);
const bothSources = await estimateOf(true);
console.log("no double counting:", providerOnly === bothSources, `| counted ${providerOnly} tok from the stream either way`);

// 5. writing code through a tool call counts as generation
request(llama);
chunk({ prompt_progress: { total: 1000, cache: 900, processed: 1000, time_ms: 60 } });
for (const args of ['{"path":"/home/mitja/demo.py","content":"', "def fib(n):\n", "  return fib(n-1)+fib(n-2)\n", '"}']) {
  chunk({ choices: [{ delta: { tool_calls: [{ index: 0, function: { arguments: args } }] } }] });
  await sleep(90);
}
await sleep(250);
show("writing code via tool");
end({ input: 100, output: 42, cacheRead: 900 }, "toolUse");

// 6. thinking may arrive in pi's events only (endpoints without reasoning chunks)
request(cloud);
for (let i = 0; i < 25; i++) delta("pondering ", "thinking_delta");
await sleep(250);
show("thinking via pi events");
end({ input: 0, output: 35, cacheRead: 0 });

// 7. model loading: status from /models, percentage from the state stream
let modelState = "loading";
const fetched = [];
const frames = [
  { status: "loading", progress: { stages: ["text_model"], current: "text_model", value: 0.2 } },
  { status: "loading", progress: { stages: ["text_model"], current: "spec_model", value: 0.6 } },
  { status: "loaded", progress: {} },
];
let frameIndex = 0;
const encoder = new TextEncoder();
globalThis.fetch = async (url) => {
  const target = String(url);
  fetched.push(target);
  if (target.endsWith("/models/sse")) {
    return {
      ok: true,
      status: 200,
      body: {
        getReader: () => ({
          read: async () => {
            if (frameIndex >= frames.length) return { done: true, value: undefined };
            await sleep(300);
            const frame = frames[frameIndex++];
            const text = `data: {"model":"${llama.id}","event":"status_change","data":${JSON.stringify(frame)}}\n\n`;
            return { done: false, value: encoder.encode(text) };
          },
        }),
      },
    };
  }
  return { ok: true, status: 200, json: async () => ({ data: [{ id: llama.id, status: { value: modelState } }] }) };
};
request(llama);
await sleep(1500); // late enough that the model list is read
show("llama, model loading");
const loadLine = working;
await sleep(500);
show("llama, load progress");
const progressLine = working;
modelState = "loaded";
await sleep(900);
show("llama, model ready");
chunk({ prompt_progress: { total: 500, cache: 0, processed: 500, time_ms: 200 } });
end({ input: 0, output: 0, cacheRead: 0 }, "aborted");
agentEnd();
await commands.get("speed").handler("", makeCtx(llama));
console.log("loading state:", loadLine.startsWith("\ud83e\udd16"), "|", loadLine);
console.log("with progress:", /\d%/.test(progressLine), "|", progressLine);
console.log("asked:", [...new Set(fetched)].join(" "));
console.log("recap:", (notices.at(-1) || "").match(/last: (.*)$/m)?.[1]);

// cloud endpoints are never asked anything
fetched.length = 0;
request(cloud);
await sleep(1500);
console.log("cloud endpoint polled the server:", fetched.length === 0 ? "no" : "YES (wrong)");
end({ input: 0, output: 0, cacheRead: 0 });

// 8. closing the state stream mid-flight must not leave a rejection behind
let unhandled = 0;
process.on("unhandledRejection", () => unhandled++);
process.on("uncaughtException", () => unhandled++);
globalThis.fetch = async (url, options) => {
  const target = String(url);
  if (target.endsWith("/models/sse")) {
    const signal = options?.signal;
    return {
      ok: true,
      status: 200,
      body: {
        getReader: () => ({
          read: () =>
            new Promise((resolve, reject) => {
              const timer = setTimeout(() => {
                const frame = { status: "loading", progress: { current: "text_model", value: 0.5 } };
                resolve({ done: false, value: encoder.encode(`data: {"model":"${llama.id}","data":${JSON.stringify(frame)}}\n\n`) });
              }, 400);
              signal?.addEventListener("abort", () => {
                clearTimeout(timer);
                // This is what undici does when the request is aborted.
                reject(new DOMException("This operation was aborted", "AbortError"));
              });
            }),
        }),
      },
    };
  }
  return { ok: true, status: 200, json: async () => ({ data: [{ id: llama.id, status: { value: "loading" } }] }) };
};
request(llama);
await sleep(2000); // model list read, state stream open and mid-frame
const midLine = working;
end({ input: 0, output: 0, cacheRead: 0 }, "aborted");
agentEnd(); // this closes the stream while a read is pending
await sleep(600);
console.log("mid-load line:", midLine.startsWith("\ud83e\udd16"), "|", midLine);
console.log("unhandled rejections after abort:", unhandled);

// 8. command and shutdown
await commands.get("speed").handler("", makeCtx(llama));
console.log("notify:", (notices.at(-1) || "").replace(/\n/g, " | "));
await commands.get("speed").handler("off", makeCtx(llama));
show("after /speed off");
request(cloud);
show("cloud after off");
await commands.get("speed").handler("on", makeCtx(llama));
handlers.get("session_shutdown")({ type: "session_shutdown" }, makeCtx(llama));
show("after shutdown");
console.log("setStatus calls:", statusCalls.length === 0 ? "none (footer untouched)" : statusCalls.join(", "));
console.log("harness exits on its own when timers are cleared");
