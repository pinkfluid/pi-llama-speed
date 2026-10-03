/**
 * llama-speed: how fast the current model request is going, on the "Working"
 * line above the chat input.
 *
 *   ⚡ 0.4s                          waiting for the first token
 *   🤖 ▓▓░░░░░░ 25% 8.4s             the model itself is being loaded (llama.cpp)
 *   ⚡ ▓▓▓▓░░░░ 54% 3078 t/s 1.6s     prompt processing, llama.cpp only
 *   🔥 38.1 t/s 1.6s                 the model is answering
 *
 * Prefill and generation are never shown together, and when the request finishes
 * pi's own "Working" text comes straight back. Token totals, cache rate and cost
 * are left out on purpose because pi's footer already shows them.
 *
 * llama.cpp is the focus: those endpoints are asked for prompt_progress, which
 * gives the server's own prefill numbers, and their stream is read directly. When
 * a reply is late, the server's model list says whether the model is in memory,
 * and if it is not, its model state stream supplies the load percentage. Both are
 * plain GETs to that same host, and both are closed through an AbortController
 * inside try/catch, so nothing can reject unhandled. No
 * other provider publishes that, so every other endpoint (OpenAI, Copilot,
 * Anthropic, OpenRouter, ...) gets the same line minus the prefill detail,
 * counted from pi's own stream events: no extra request fields, no extra calls.
 *
 * It reads pi's own events and never touches fetch(), Response bodies, or
 * ReadableStreams, so aborting a request cannot leak a rejected promise.
 *
 * Commands:
 *   /speed            detailed numbers for the current and last request
 *   /speed off        stop updating the working line
 *   /speed on         start updating it again
 *   /speed progress   toggle sending return_progress to local servers
 *   /speed load       toggle reading the server's model list
 */

import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";

/** Minimum time between working line updates, in milliseconds. */
const REPAINT_MS = 200;
/** How many prefill samples to average for the live rate. */
const PREFILL_WINDOW = 6;
/** Sliding window used for the live generation rate. */
const GEN_WINDOW_MS = 3000;
/** Only look at model state once a reply is visibly late. */
const LOAD_POLL_AFTER_MS = 1200;
/** And at most this often while it stays late. */
const LOAD_POLL_EVERY_MS = 2000;
/** Give up on a slow answer from the model list. */
const LOAD_FETCH_TIMEOUT_MS = 2500;

type PromptProgress = {
  total?: number;
  processed?: number;
  time_ms?: number;
  cache?: number;
};

type ChunkShape = {
  prompt_progress?: PromptProgress;
  usage?: { prompt_tokens?: number; completion_tokens?: number };
  choices?: Array<{
    delta?: {
      content?: string;
      reasoning_content?: string;
      reasoning?: string;
      reasoning_text?: string;
      tool_calls?: Array<{ function?: { name?: string; arguments?: string } }>;
    };
  }>;
};

type Usage = { input: number; output: number; cacheRead: number };

type Request = {
  modelLabel: string;
  baseUrl: string;
  startedAt: number;
  firstDeltaAt?: number;
  lastDeltaAt?: number;
  /** Latest prompt_progress from the server, plus the previous one for rates. */
  progress?: PromptProgress;
  prevProcessed?: number;
  prevTimeMs?: number;
  prefillRates: number[];
  /** Streamed text characters, used to estimate tokens before usage arrives. */
  outTextChars: number;
  /** Tool call argument characters and non-empty fragments, counted separately. */
  outArgChars: number;
  outArgPieces: number;
  /** Real output token count once the server reports usage. */
  outTokensReal?: number;
  genSamples: Array<{ at: number; tokens: number }>;
  /** Set as soon as the server sends prompt_progress; it identifies llama.cpp. */
  sawServerProgress: boolean;
  /** Count from pi's stream events instead of the provider's own chunks. */
  meterFromPi: boolean;
  /** Set once the provider chunks gave us something to count. */
  meteredFromProvider: boolean;
  /** Model id as pi asks for it, used for the server's model state endpoints. */
  modelId: string;
  /** The name the server itself uses for that model, once known. */
  modelName?: string;
  /** Model state from the server's own model list, e.g. "loading" or "loaded". */
  modelStatus?: string;
  /** Load progress as a fraction, from the server's model state stream. */
  loadFraction?: number;
  /** Which part of the load is running, e.g. "text_model" or "spec_model". */
  loadStage?: string;
  /** First moment the server said the model was not in memory yet. */
  loadSeenAt?: number;
  /** Moment it became ready. */
  loadReadyAt?: number;
  /** When we last asked the server for model state. */
  statusCheckAt?: number;
  /** Stop asking once the model is known to be ready for this request. */
  statusDone?: boolean;
  usage?: Usage;
  stopReason?: string;
};

/** Hosts we are willing to send llama.cpp-only request fields to. */
function isPrivateHost(value: string | undefined): boolean {
  if (!value) return false;
  let host: string;
  try {
    host = new URL(value).hostname.toLowerCase();
  } catch {
    return false;
  }
  if (host === "localhost" || host.endsWith(".localhost") || host.endsWith(".local") || host.endsWith(".lan")) return true;
  if (host === "::1" || host === "0.0.0.0") return true;
  // A name with no dot at all can only be resolved by the local network, so it
  // is a machine on the LAN or a VPN name such as "lama".
  if (!host.includes(".") && host.length > 0) return true;
  const parts = host.split(".").map((part) => Number(part));
  if (parts.length !== 4 || parts.some((part) => !Number.isInteger(part))) return false;
  return (
    parts[0] === 127 ||
    parts[0] === 10 ||
    (parts[0] === 192 && parts[1] === 168) ||
    (parts[0] === 172 && parts[1] >= 16 && parts[1] <= 31)
  );
}

/** Endpoints that may be llama.cpp: OpenAI-compatible chat on a local host. */
function isLlamaCandidate(model: { api?: string; baseUrl?: string } | undefined): boolean {
  return !!model && model.api === "openai-completions" && isPrivateHost(model.baseUrl);
}

function newRequest(modelLabel: string, modelId: string, baseUrl: string, meterFromPi: boolean): Request {
  return {
    modelLabel,
    modelId,
    baseUrl,
    startedAt: Date.now(),
    prefillRates: [],
    outTextChars: 0,
    outArgChars: 0,
    outArgPieces: 0,
    genSamples: [],
    sawServerProgress: false,
    meterFromPi,
    meteredFromProvider: false,
  };
}

function fmtTokens(value: number): string {
  if (value >= 10000) return `${Math.round(value / 1000)}k`;
  if (value >= 1000) return `${(value / 1000).toFixed(1)}k`;
  return `${value}`;
}

function fmtRate(value: number): string {
  return value >= 100 ? `${Math.round(value)}` : value.toFixed(1);
}

function fmtDuration(seconds: number): string {
  if (seconds < 10) return `${seconds.toFixed(1)}s`;
  if (seconds < 60) return `${Math.round(seconds)}s`;
  return `${Math.floor(seconds / 60)}m${String(Math.round(seconds % 60)).padStart(2, "0")}s`;
}

function progressBar(fraction: number, width = 8): string {
  const filled = Math.max(0, Math.min(width, Math.round(fraction * width)));
  return "▓".repeat(filled) + "░".repeat(width - filled);
}

/** Only keep rates that are real numbers. */
function usable(value: number | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value) && value > 0;
}

/** Live rate from the server's prompt_progress samples. */
function prefillRate(req: Request): number | undefined {
  if (req.prefillRates.length > 0) {
    const recent = req.prefillRates.slice(-PREFILL_WINDOW);
    return recent.reduce((sum, value) => sum + value, 0) / recent.length;
  }
  const progress = req.progress;
  if (progress && typeof progress.processed === "number" && typeof progress.time_ms === "number" && progress.time_ms > 0) {
    return (progress.processed / progress.time_ms) * 1000;
  }
  return undefined;
}

/** Output tokens so far: server-reported first, then our own estimate. */
function outputTokens(req: Request): number {
  return req.outTokensReal ?? estimateTokens(req);
}

/**
 * Our own count from what actually arrived on the wire. Prose and thinking are
 * roughly four characters per token; tool call arguments carry framing tokens
 * that never show up as bytes, so each fragment counts as at least one.
 */
const TEXT_CHARS_PER_TOKEN = 4.5;
// Tool call arguments are JSON with escapes, and the server adds call framing
// tokens that never appear as bytes here, so they cost more tokens per byte.
const ARG_CHARS_PER_TOKEN = 1.7;

function estimateTokens(req: Request): number {
  const text = Math.round(req.outTextChars / TEXT_CHARS_PER_TOKEN);
  const args = Math.max(req.outArgPieces, Math.round(req.outArgChars / ARG_CHARS_PER_TOKEN));
  return text + args;
}

/** True while the shown token count is still an estimate. */
function estimated(req: Request): boolean {
  return req.outTokensReal === undefined;
}

/** Live generation rate over a sliding window, corrected by real usage when known. */
function generationRate(req: Request): number | undefined {
  const now = Date.now();
  const tokens = outputTokens(req);
  if (!req.firstDeltaAt || tokens <= 0) return undefined;
  const samples = [...req.genSamples, { at: now, tokens }];
  const windowStart = now - GEN_WINDOW_MS;
  let oldest = samples[0];
  for (const sample of samples) {
    if (sample.at >= windowStart) break;
    oldest = sample;
  }
  const newest = samples[samples.length - 1];
  const span = (newest.at - oldest.at) / 1000;
  if (span >= 0.4 && newest.tokens > oldest.tokens) return (newest.tokens - oldest.tokens) / span;
  const total = (now - req.firstDeltaAt) / 1000;
  if (total >= 0.2) return tokens / total;
  return undefined;
}

/** Cache hit share from the server's progress sample. */
function cacheShare(progress: PromptProgress | undefined): number | undefined {
  if (!progress || typeof progress.cache !== "number" || progress.cache <= 0) return undefined;
  const total = typeof progress.total === "number" && progress.total > 0 ? progress.total : progress.processed;
  if (!total || total <= 0) return undefined;
  return Math.min(1, progress.cache / total);
}

function cacheFromUsage(req: Request): number | undefined {
  const usage = req.usage;
  if (!usage || typeof usage.cacheRead !== "number" || usage.cacheRead <= 0) return undefined;
  const fresh = typeof usage.input === "number" && usage.input > 0 ? usage.input : 0;
  return Math.min(1, usage.cacheRead / (usage.cacheRead + fresh));
}

/**
 * The model list sits next to the completions endpoint, so this only swaps the
 * path. Used for llama.cpp style hosts only.
 */
function modelListUrl(baseUrl: string): string | undefined {
  try {
    const url = new URL(baseUrl);
    url.pathname = "/models";
    url.search = "";
    return url.toString();
  } catch {
    return undefined;
  }
}

/**
 * States in which the server cannot answer yet because the model is not in
 * memory. "sleeping" counts too: waking it up reloads the weights.
 */
const NOT_READY = new Set(["downloading", "unloaded", "loading", "sleeping"]);

function modelIsLoading(req: Request): boolean {
  return req.modelStatus !== undefined && NOT_READY.has(req.modelStatus);
}

/** The model state stream sits next to the completions endpoint. */
function eventStreamUrl(baseUrl: string): string | undefined {
  try {
    const url = new URL(baseUrl);
    url.pathname = "/models/sse";
    url.search = "";
    return url.toString();
  } catch {
    return undefined;
  }
}

/**
 * Pull a fraction out of the server's progress payload. Loading reports
 * {stages, current, value}; a download reports {downloaded, total} instead.
 */
function loadFractionOf(progress: unknown): number | undefined {
  if (!progress || typeof progress !== "object") return undefined;
  const record = progress as Record<string, unknown>;
  if (typeof record.value === "number" && Number.isFinite(record.value)) {
    return Math.max(0, Math.min(1, record.value));
  }
  if (typeof record.downloaded === "number" && typeof record.total === "number" && record.total > 0) {
    return Math.max(0, Math.min(1, record.downloaded / record.total));
  }
  for (const value of Object.values(record)) {
    if (value && typeof value === "object") {
      const nested = value as Record<string, unknown>;
      if (typeof nested.downloaded === "number" && typeof nested.total === "number" && nested.total > 0) {
        return Math.max(0, Math.min(1, nested.downloaded / nested.total));
      }
    }
  }
  return undefined;
}

/**
 * Status line for a request in flight. Prefill and generation are shown one at
 * a time, never together, and both endpoint kinds use the same shape: a rate and
 * an elapsed time. Token counts, cache rate and cost stay in pi's footer.
 */
function statusText(req: Request): string {
  const now = Date.now();

  // The model has not produced anything yet: show the server's own numbers.
  if (!req.firstDeltaAt) {
    const progress = req.progress;
    if (!progress || typeof progress.processed !== "number") {
      const waited = fmtDuration((now - req.startedAt) / 1000);
      if (modelIsLoading(req)) {
        // Third state: the model itself is coming into memory.
        const bits = ["🤖"];
        if (req.loadFraction !== undefined) {
          bits.push(`${progressBar(req.loadFraction)} ${Math.round(req.loadFraction * 100)}%`);
        }
        bits.push(waited);
        if (req.loadStage && req.loadStage !== "text_model") bits.push(req.loadStage);
        return bits.join(" ");
      }
      return `⚡ ${waited}`;
    }
    const bits = ["⚡"];
    const total = typeof progress.total === "number" && progress.total > 0 ? progress.total : undefined;
    if (total) bits.push(`${progressBar(Math.min(1, progress.processed / total))} ${Math.round((progress.processed / total) * 100)}%`);
    const rate = prefillRate(req);
    if (usable(rate)) bits.push(`${fmtRate(rate)} t/s`);
    // Prefer the server's own prompt processing time; it belongs with its rate.
    const elapsed = typeof progress.time_ms === "number" && progress.time_ms > 0 ? progress.time_ms : now - req.startedAt;
    bits.push(fmtDuration(elapsed / 1000));
    return bits.join(" ");
  }

  // The model is answering: show generation speed only.
  const rate = generationRate(req);
  const bits = ["🔥", usable(rate) ? `${fmtRate(rate)} t/s` : "measuring"];
  bits.push(fmtDuration((now - (req.firstDeltaAt ?? now)) / 1000));
  return bits.join(" ");
}

/** Summary of a finished request; only used by /speed and the debug log. */
function recapText(req: Request): string | undefined {
  const bits: string[] = [];
  if (req.firstDeltaAt) {
    const usage = req.usage;
    const out = usage && typeof usage.output === "number" && usage.output > 0 ? usage.output : outputTokens(req);
    const rate =
      usage && typeof usage.output === "number" && usage.output > 0 && req.lastDeltaAt && req.lastDeltaAt > req.firstDeltaAt
        ? (usage.output / (req.lastDeltaAt - req.firstDeltaAt)) * 1000
        : generationRate(req);
    bits.push("🔥");
    bits.push(usable(rate) ? `${fmtRate(rate)} t/s` : "?");
    bits.push(`${usage && typeof usage.output === "number" && usage.output > 0 ? "" : "~"}${fmtTokens(out)} tok`);
  } else {
    const rate = prefillRate(req);
    const progress = req.progress;
    const processed = progress?.processed;
    if (!usable(rate) && !processed) return undefined;
    bits.push("⚡");
    if (usable(rate)) bits.push(`${fmtRate(rate)} t/s`);
    if (processed) bits.push(`${fmtTokens(processed)} tok`);
    if (progress && typeof progress.time_ms === "number" && progress.time_ms > 0) {
      bits.push(fmtDuration(progress.time_ms / 1000));
    }
  }
  if (req.loadSeenAt && req.loadReadyAt && req.loadReadyAt > req.loadSeenAt) {
    bits.push(`model ready in ${fmtDuration((req.loadReadyAt - req.loadSeenAt) / 1000)}`);
  }
  const cached = cacheFromUsage(req) ?? cacheShare(req.progress);
  // Only a cancelled or failed request is worth saying out loud.
  if (req.stopReason === "aborted" || req.stopReason === "error") bits.push(req.stopReason);
  else if (cached !== undefined) bits.push(`cache ${Math.round(cached * 100)}%`);
  return bits.join(" ");
}

export default function (pi: ExtensionAPI) {
  let req: Request | undefined;
  let timer: ReturnType<typeof setInterval> | undefined;
  /** Last request summary, kept for /speed only; it is never shown by itself. */
  let recap: string | undefined;
  let enabled = true;
  let injectProgress = true;
  /** Whether reading the server's model list is allowed. */
  let pollModel = true;
  let pollInFlight = false;
  /** Hosts that answered badly, so we stop asking them. */
  const statusFailures = new Map<string, number>();
  const streamFailures = new Map<string, number>();
  /** One open model state stream at a time, closed through this controller. */
  let stateStream: AbortController | undefined;
  /** Endpoints that have already proven they send prompt_progress. */
  const llamaEndpoints = new Map<string, boolean>();
  const debug = !!process.env.LLAMA_SPEED_DEBUG;
  const log = (...args: unknown[]) => {
    if (debug) console.error("[llama-speed]", ...args);
  };

  const stopTimer = () => {
    if (timer !== undefined) {
      clearInterval(timer);
      timer = undefined;
    }
  };

  /** Live numbers replace the "Working" text above the input. */
  const paintLive = (ctx: ExtensionContext | undefined, text: string) => {
    if (!enabled || !ctx?.hasUI) return;
    try {
      ctx.ui.setWorkingMessage?.(text);
    } catch (error) {
      log("setWorkingMessage failed", error);
    }
  };

  /** Put pi's own "Working" text back. */
  const clearLive = (ctx: ExtensionContext | undefined) => {
    if (!ctx?.hasUI) return;
    try {
      ctx.ui.setWorkingMessage?.();
    } catch {
      // The UI may already be gone.
    }
  };

  /**
   * When a reply is late, ask the server whether the model is still being loaded.
   * One small GET against an endpoint we already know is llama.cpp, only while the
   * request has produced nothing, and never more than once at a time. Every
   * failure is swallowed: a server without that endpoint is simply not polled
   * again. This never touches the running stream.
   */
  /** Close the model state stream. Safe to call at any time. */
  const stopModelState = () => {
    const current = stateStream;
    stateStream = undefined;
    try {
      current?.abort();
    } catch {
      // The abort reason is a DOMException we deliberately drop; it must never
      // reach an unhandled rejection, which is what used to kill pi.
    }
  };

  const maybeCheckModelStatus = (ctx: ExtensionContext | undefined) => {
    // Only endpoints that look like a local llama.cpp server are ever asked.
    // Skip it while the server is already processing the prompt: that proves the
    // model is in memory, so the only case left worth asking about is silence.
    if (!req || req.meterFromPi || pollInFlight || req.statusDone || req.firstDeltaAt || req.progress || !pollModel) return;
    const now = Date.now();
    if (now - req.startedAt < LOAD_POLL_AFTER_MS) return;
    if (req.statusCheckAt && now - req.statusCheckAt < LOAD_POLL_EVERY_MS) return;
    const model = ctx?.model;
    const url = modelListUrl(model?.baseUrl ?? "");
    if (!url || (statusFailures.get(url) ?? 0) >= 2) return;
    req.statusCheckAt = now;
    pollInFlight = true;
    void (async () => {
      const response = await fetch(url, {
        signal: AbortSignal.timeout(LOAD_FETCH_TIMEOUT_MS),
        headers: { accept: "application/json" },
      });
      if (!response.ok) throw new Error(`status ${response.status}`);
      const body: unknown = await response.json();
      const entries = Array.isArray((body as { data?: unknown })?.data) ? (body as { data: unknown[] }).data : [];
      const wanted = model?.id ?? "";
      for (const entry of entries) {
        const record = entry as { id?: string; aliases?: unknown; status?: { value?: unknown } };
        const aliases = Array.isArray(record?.aliases) ? (record.aliases as unknown[]).filter((a): a is string => typeof a === "string") : [];
        if (record?.id !== wanted && !aliases.includes(wanted)) continue;
        // The state stream names models the server's own way, so remember it.
        req.modelName = record.id;
        const value = typeof record.status?.value === "string" ? record.status.value : undefined;
        if (value) applyModelStatus(value, now);
        return;
      }
    })()
      .catch((error: unknown) => {
        statusFailures.set(url, (statusFailures.get(url) ?? 0) + 1);
        log("model status unavailable", String((error as Error)?.message ?? error));
      })
      .finally(() => {
        pollInFlight = false;
      });
  };

  /**
   * Follow the server's model state stream while the model is coming into
   * memory; that is the only place llama.cpp reports load progress. Everything is
   * inside try/catch and the stream is closed with an AbortController, so no
   * rejection can escape unhandled - the failure class that used to crash pi.
   */
  const watchModelState = () => {
    const url = eventStreamUrl(req?.baseUrl ?? "");
    if (!url || stateStream || (streamFailures.get(url) ?? 0) >= 2) return;
    const controller = new AbortController();
    stateStream = controller;
    void (async () => {
      const response = await fetch(url, {
        signal: controller.signal,
        headers: { accept: "text/event-stream" },
      });
      if (!response.ok) throw new Error(`status ${response.status}`);
      if (!response.body) throw new Error("no body");
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const chunk = await reader.read();
        if (chunk.done) break;
        buffer += decoder.decode(chunk.value, { stream: true });
        let cut = buffer.indexOf("\n\n");
        while (cut >= 0) {
          applyStateFrame(buffer.slice(0, cut), req?.modelId ?? "");
          buffer = buffer.slice(cut + 2);
          cut = buffer.indexOf("\n\n");
        }
        // Stop as soon as the model is ready or the answer started streaming.
        if (!req || req.firstDeltaAt || !modelIsLoading(req)) break;
      }
    })()
      .catch((error: unknown) => {
        streamFailures.set(url, (streamFailures.get(url) ?? 0) + 1);
        log("model state stream unavailable", String((error as Error)?.message ?? error));
      })
      .finally(() => {
        stopModelState();
      });
  };

  /** One "data: {...}" frame from the model state stream. */
  const applyStateFrame = (frame: string, modelId: string) => {
    const line = frame.split("\n").find((entry) => entry.startsWith("data:"));
    if (!line) return;
    let parsed: unknown;
    try {
      parsed = JSON.parse(line.slice(5).trim());
    } catch {
      return;
    }
    const event = parsed as { model?: string; data?: { status?: unknown; progress?: unknown } };
    const known = [modelId, req?.modelName].filter((name): name is string => !!name);
    if (!event || (event.model && !known.includes(event.model))) return;
    const status = typeof event.data?.status === "string" ? event.data.status : undefined;
    if (status) applyModelStatus(status, Date.now(), event.data?.progress);
  };

  /** Record what the server said about its model, and when it changed. */
  const applyModelStatus = (value: string, at: number, progress?: unknown) => {
    if (!req) return;
    req.modelStatus = value;
    if (modelIsLoading(req)) {
      if (req.loadSeenAt === undefined) req.loadSeenAt = at;
      const fraction = loadFractionOf(progress);
      if (fraction !== undefined) req.loadFraction = fraction;
      const stage = (progress as { current?: unknown } | undefined)?.current;
      if (typeof stage === "string") req.loadStage = stage;
      // The model list has no percentages, so go and watch the state stream.
      watchModelState();
      return;
    }
    if (value === "loaded" || value === "downloaded") {
      req.statusDone = true;
      stopModelState();
      if (req.loadSeenAt !== undefined && req.loadReadyAt === undefined) {
        req.loadReadyAt = at;
        log(`model ready ${value} after ${at - req.loadSeenAt}ms`);
      }
    }
  };

  const startTimer = (ctx: ExtensionContext | undefined) => {
    if (!ctx?.hasUI) return;
    stopTimer();
    timer = setInterval(() => {
      try {
        if (req) {
          // Once the model answers there is nothing left to learn about loading.
          if (req.firstDeltaAt) stopModelState();
          else maybeCheckModelStatus(ctx);
          paintLive(ctx, statusText(req));
        }
      } catch (error) {
        log("repaint failed", error);
      }
    }, REPAINT_MS);
  };

  const observeChunk = (data: unknown) => {
    if (!req || !data || typeof data !== "object") return;
    const chunk = data as ChunkShape;
    const progress = chunk.prompt_progress;
    if (progress && typeof progress === "object") {
      if (!req.sawServerProgress) {
        req.sawServerProgress = true;
        llamaEndpoints.set(req.baseUrl, true);
        log("llama.cpp confirmed", req.baseUrl);
      }
      if (typeof progress.processed === "number" && typeof progress.time_ms === "number") {
        const dt = progress.time_ms - (req.prevTimeMs ?? 0);
        const dp = progress.processed - (req.prevProcessed ?? 0);
        if (dt > 0 && dp > 0) {
          req.prefillRates.push((dp / dt) * 1000);
          if (req.prefillRates.length > 20) req.prefillRates.shift();
        }
        req.prevProcessed = progress.processed;
        req.prevTimeMs = progress.time_ms;
      }
      req.progress = progress;
      return;
    }
    if (chunk.usage && typeof chunk.usage.completion_tokens === "number") {
      req.outTokensReal = chunk.usage.completion_tokens;
    }
    const delta = chunk.choices?.[0]?.delta;
    if (!delta) return;
    // Everything the model generates counts: prose, thinking, and the arguments
    // it streams into a tool call (that is where written code goes). Thinking
    // uses whichever reasoning field this endpoint fills, same as pi does, so a
    // server that sends two of them is not counted twice.
    const thinking = [delta.reasoning_content, delta.reasoning, delta.reasoning_text].find(
      (value) => typeof value === "string" && value.length > 0,
    );
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    let argChars = 0;
    let argPieces = 0;
    for (const call of calls) {
      const fragment = `${call?.function?.name ?? ""}${call?.function?.arguments ?? ""}`;
      if (fragment.length > 0) {
        argChars += fragment.length;
        argPieces += 1;
      }
    }
    const textChars = `${delta.content ?? ""}${thinking ?? ""}`.length;
    if (textChars === 0 && argPieces === 0) return;
    req.meteredFromProvider = true;
    const now = Date.now();
    if (!req.firstDeltaAt) req.firstDeltaAt = now;
    req.lastDeltaAt = now;
    req.outTextChars += textChars;
    req.outArgChars += argChars;
    req.outArgPieces += argPieces;
    req.genSamples.push({ at: now, tokens: outputTokens(req) });
    if (req.genSamples.length > 400) req.genSamples.shift();
  };

  pi.on("before_provider_request", (event, ctx) => {
    const model = ctx.model;
    const candidate = isLlamaCandidate(model);
    const baseUrl = model?.baseUrl ?? "";
    recap = undefined;
    stopModelState();
    // Every endpoint is metered; llama.cpp candidates are read from the provider's
    // own chunks, everything else from pi's stream events.
    req = newRequest(model ? `${model.provider}/${model.id}` : "model", model?.id ?? "", baseUrl, !candidate);
    paintLive(ctx, statusText(req));
    startTimer(ctx);
    if (!candidate || !injectProgress || !model) return event.payload;
    const payload = event.payload;
    if (!payload || typeof payload !== "object" || Array.isArray(payload)) return payload;
    const body = payload as Record<string, unknown>;
    if (body.return_progress === true) return payload;
    return { ...body, return_progress: true };
  });

  pi.on("provider_stream_event", (event) => {
    if (!req || req.meterFromPi) return;
    observeChunk(event.data);
  });

  // Endpoints that do not expose a parseable chunk carry the same information in
  // pi's own stream events, so count from there instead. Only used while the
  // provider chunks gave us nothing, so nothing is ever counted twice.
  pi.on("message_update", (event) => {
    if (!req || req.meteredFromProvider) return;
    const part = event.assistantMessageEvent as { type?: string; delta?: unknown };
    if (part.type !== "text_delta" && part.type !== "thinking_delta" && part.type !== "toolcall_delta") return;
    if (typeof part.delta !== "string" || part.delta.length === 0) return;
    const now = Date.now();
    if (!req.firstDeltaAt) req.firstDeltaAt = now;
    req.lastDeltaAt = now;
    if (part.type === "toolcall_delta") {
      req.outArgChars += part.delta.length;
      req.outArgPieces += 1;
    } else {
      req.outTextChars += part.delta.length;
    }
    req.genSamples.push({ at: now, tokens: outputTokens(req) });
    if (req.genSamples.length > 400) req.genSamples.shift();
  });

  pi.on("message_end", (event, ctx) => {
    if (!req) return;
    const message = event.message as { role?: string; usage?: Usage; stopReason?: string };
    if (message.role !== "assistant") return;
    if (message.usage) req.usage = message.usage;
    req.stopReason = message.stopReason;
    stopTimer();
    stopModelState();
    recap = recapText(req);
    clearLive(ctx);
    const real = message.usage && typeof message.usage.output === "number" ? message.usage.output : undefined;
    log("request end", req.modelLabel, recap, `counted ${estimateTokens(req)} from the stream, server reported ${real}`);
  });

  pi.on("agent_end", (_event, ctx) => {
    stopTimer();
    stopModelState();
    clearLive(ctx);
  });

  pi.on("session_shutdown", (_event, ctx) => {
    stopTimer();
    stopModelState();
    clearLive(ctx);
    req = undefined;
  });

  pi.registerCommand("speed", {
    description: "Show llama.cpp request speed details",
    handler: async (args, ctx) => {
      const arg = (args || "").trim().toLowerCase();
      if (arg === "off") {
        enabled = false;
        stopTimer();
        stopModelState();
        clearLive(ctx);
        ctx.ui.notify("llama-speed: working line off", "info");
        return;
      }
      if (arg === "on") {
        enabled = true;
        ctx.ui.notify("llama-speed: working line on", "info");
        return;
      }
      if (arg === "progress") {
        injectProgress = !injectProgress;
        ctx.ui.notify(`llama-speed: return_progress ${injectProgress ? "enabled" : "disabled"} for local servers`, "info");
        return;
      }
      if (arg === "load") {
        pollModel = !pollModel;
        ctx.ui.notify(`llama-speed: model loading status ${pollModel ? "on" : "off"}`, "info");
        return;
      }
      const lines = [
        `llama-speed: ${enabled ? "on" : "off"}, return_progress ${injectProgress ? "on" : "off"}, model status ${pollModel ? "on" : "off"}`,
      ];
      if (llamaEndpoints.size === 0) {
        lines.push("no llama.cpp endpoint seen yet this session");
      } else {
        for (const url of llamaEndpoints.keys()) lines.push(`confirmed: ${url}`);
      }
      if (req && !req.stopReason) lines.push(`current: ${statusText(req)}`);
      if (recap) lines.push(`last: ${recap}`);
      ctx.ui.notify(lines.join("\n"), "info");
    },
  });
}
