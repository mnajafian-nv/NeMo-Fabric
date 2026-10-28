// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import test from "node:test";
import { readFileSync } from "node:fs";
import { findPackageJSON } from "node:module";
import { fileURLToPath, pathToFileURL } from "node:url";
import { ClineInvocationUsage, clineUsageHooks } from "../dist/usage.js";

// Qualify the supported harness package when installed, without provider calls.
function packageEntry(name, base) {
  const path = findPackageJSON(name, base);
  assert.ok(path, `Cannot find ${name}`);
  const manifest = JSON.parse(readFileSync(path, "utf8"));
  assert.equal(manifest.version, "0.0.83", "source qualification requires the supported Cline version");
  return new URL(manifest.exports["."].import, pathToFileURL(path));
}
let source = process.env.NEMO_FABRIC_TEST_CLINE_AGENTS;
if (source === undefined) {
  try {
    source = fileURLToPath(packageEntry("@cline/agents", import.meta.url));
  } catch (error) {
    if (error.code !== "ERR_MODULE_NOT_FOUND" && error.code !== "MODULE_NOT_FOUND") throw error;
  }
}
if (source !== undefined) {
  const manifest = JSON.parse(readFileSync(new URL("../package.json", pathToFileURL(source)), "utf8"));
  assert.equal(manifest.version, "0.0.83", "source qualification requires the supported Cline version");
}
const key = "nvidia.fabric.cline.usage_evidence.v1";

for (const [label, first, second] of [
  ["full", { inputTokens: 40, outputTokens: 8, cacheReadTokens: 6 }, { inputTokens: 20, outputTokens: 4, cacheReadTokens: 2 }],
  ["omitted output", { inputTokens: 40, outputTokens: 8 }, { inputTokens: 20 }],
  ["reported zero", { inputTokens: 40, outputTokens: 0 }, { inputTokens: 20, outputTokens: 0 }],
]) {
  test(`pinned AgentRuntime: ${label}, independent per-call metrics`, { skip: !source }, async () => {
    const { AgentRuntime } = await import(pathToFileURL(source).href);
    let index = 0;
    const runtime = new AgentRuntime({
      model: { async *stream() {
        yield { type: "text-delta", text: "done" };
        yield { type: "usage", usage: [first, second][index++] };
        yield { type: "finish", reason: "stop" };
      } },
      messageModelInfo: { id: "test-model", provider: "openai" },
      hooks: clineUsageHooks,
      tools: [],
    });
    const evidence = new ClineInvocationUsage();
    // Two execute calls in one Fabric owner prove runId/iteration are not call IDs.
    await evidence.run(async () => {
      await runtime.run("one");
      await runtime.continue("two");
    });
    const result = evidence.finish();
    assert.equal(result.input_tokens, undefined);
    assert.equal(result.extensions["nvidia.fabric.cline.usage_evidence.v1"].observed.input_tokens, 60);
    assert.equal(result.output_tokens, undefined);
    assert.equal(result.extensions[key].observed.output_tokens, label === "full" ? 12 : label === "omitted output" ? 8 : undefined);
  });
}

test("pinned AgentRuntime: a stream that fails before afterModel leaves owned work unfinished", { skip: !source }, async () => {
  const { AgentRuntime } = await import(pathToFileURL(source).href);
  const runtime = new AgentRuntime({
    model: { async *stream() { throw new Error("connection dropped"); } },
    messageModelInfo: { id: "test-model", provider: "openai" }, hooks: clineUsageHooks, tools: [],
  });
  const evidence = new ClineInvocationUsage();
  await evidence.run(() => runtime.run("one"));
  const result = evidence.finish();
  assert.equal(result.input_tokens, undefined);
  assert.ok(result.extensions[key].gaps.some(gap => gap === "unfinished_work"));
});


test("pinned AgentRuntime: reported usage survives a stream failure as observations", { skip: !source }, async () => {
  const { AgentRuntime } = await import(pathToFileURL(source).href);
  const runtime = new AgentRuntime({
    model: { async *stream() {
      yield { type: "usage", usage: { inputTokens: 40, outputTokens: 8 } };
      throw new Error("connection dropped after usage");
    } },
    messageModelInfo: { id: "test-model", provider: "openai" }, hooks: clineUsageHooks, tools: [],
  });
  const evidence = new ClineInvocationUsage();
  await evidence.run(() => runtime.run("one"));
  const result = evidence.finish();
  assert.equal(result.input_tokens, undefined);
  assert.equal(result.extensions[key].observed.input_tokens, 40);
  assert.equal(result.extensions[key].observed.output_tokens, 8);
});

test("pinned AgentRuntime: failed completion retains observed usage without inventing unfinished work", { skip: !source }, async () => {
  const { AgentRuntime } = await import(pathToFileURL(source).href);
  const runtime = new AgentRuntime({
    model: { async *stream() {
      yield { type: "usage", usage: { inputTokens: 40, outputTokens: 8 } };
      yield { type: "finish", reason: "error", error: "provider stopped", errorRetryable: false };
    } },
    messageModelInfo: { id: "test-model", provider: "openai" }, hooks: clineUsageHooks, tools: [],
  });
  const evidence = new ClineInvocationUsage();
  const terminal = await evidence.run(() => runtime.run("one"));
  assert.equal(terminal.status, "failed");
  const result = evidence.finish();
  assert.equal(result.input_tokens, undefined);
  assert.equal(result.extensions["nvidia.fabric.cline.usage_evidence.v1"].observed.input_tokens, 40);
  assert.equal(result.output_tokens, undefined);
  assert.equal(result.extensions["nvidia.fabric.cline.usage_evidence.v1"].observed.output_tokens, 8);
  assert.ok(!result.extensions[key].gaps.some(gap => gap === "unfinished_work"));
});

test("pinned AgentRuntime: overflow recovery keeps both calls in the same iteration", { skip: !source }, async () => {
  const { AgentRuntime } = await import(pathToFileURL(source).href);
  let calls = 0;
  const runtime = new AgentRuntime({
    model: { async *stream() {
      calls++;
      yield { type: "usage", usage: { inputTokens: calls === 1 ? 40 : 20, outputTokens: calls === 1 ? 8 : 4 } };
      if (calls === 1) yield { type: "finish", reason: "error", error: "context window exceeded", errorClass: "context_window_exceeded" };
      else { yield { type: "text-delta", text: "done" }; yield { type: "finish", reason: "stop" }; }
    } },
    initialMessages: [
      { id: "history-user", role: "user", content: [{ type: "text", text: "old question" }], createdAt: 1 },
      { id: "history-assistant", role: "assistant", content: [{ type: "text", text: "old answer" }], createdAt: 2 },
    ],
    prepareTurn(context) { return context.overflowRecovery ? { messages: context.messages.slice(-1) } : undefined; },
    messageModelInfo: { id: "test-model", provider: "openai" }, hooks: clineUsageHooks, tools: [],
  });
  const evidence = new ClineInvocationUsage();
  await evidence.run(() => runtime.run("one"));
  const result = evidence.finish();
  assert.equal(calls, 2);
  assert.equal(result.input_tokens, undefined);
  assert.equal(result.extensions["nvidia.fabric.cline.usage_evidence.v1"].observed.input_tokens, 60);
  assert.equal(result.output_tokens, undefined);
  assert.equal(result.extensions["nvidia.fabric.cline.usage_evidence.v1"].observed.output_tokens, 12);
});

for (const retry of ["none", "503", "empty"]) {
  test(`pinned llms ${retry} retry keeps available usage without claiming request coverage`, {skip: !source}, async () => {
    const {AgentRuntime} = await import(pathToFileURL(source).href);
    const {createGateway} = await import(packageEntry("@cline/llms", pathToFileURL(source)).href);
    let requests = 0;
    // Test-only supported Gateway DI exposes the physical sends. Core's public
    // localRuntime bootstrap does not expose this gateway injection to Fabric.
    const gateway = createGateway({providerConfigs: [{providerId: "openai-compatible", apiKey: "fixture-only", baseUrl: "https://fixture.invalid/v1", fetch: async () => {
      requests++;
      if (retry === "503" && requests === 1) return new Response(JSON.stringify({error: {message: "controlled retry", type: "server_error"}}), {status: 503, headers: {"content-type": "application/json", "retry-after": "0"}});
      const empty = retry === "empty" && requests === 1;
      const frame = {id: "response-fixture", object: "chat.completion.chunk", created: 1, model: "gpt-4.1-mini", choices: [{index: 0, delta: {role: "assistant", content: empty ? "" : "done"}, finish_reason: null}]};
      const final = {...frame, choices: [{index: 0, delta: {}, finish_reason: "stop"}], usage: {prompt_tokens: empty ? 20 : 40, completion_tokens: empty ? 4 : 8, total_tokens: empty ? 24 : 48, prompt_tokens_details: {cached_tokens: empty ? 2 : 6}}};
      return new Response(`data: ${JSON.stringify(frame)}\n\ndata: ${JSON.stringify(final)}\n\ndata: [DONE]\n\n`, {headers: {"content-type": "text/event-stream"}});
    }}]});
    const runtime = new AgentRuntime({model: gateway.createAgentModel({providerId: "openai-compatible", modelId: "gpt-4.1-mini"}), messageModelInfo: {id: "gpt-4.1-mini", provider: "openai-compatible"}, hooks: clineUsageHooks, tools: []});
    const evidence = new ClineInvocationUsage();
    const result = await evidence.run(() => runtime.run("reply done"));
    const usage = evidence.finish();
    assert.equal(result.status, "completed");
    assert.equal(requests, retry === "none" ? 1 : 2);
    assert.equal(usage.input_tokens, undefined);
    assert.equal(usage.output_tokens, undefined);
    assert.equal(usage.extensions[key].observed.input_tokens, retry === "empty" ? 60 : 40);
    assert.equal(usage.extensions[key].observed.output_tokens, retry === "empty" ? 12 : 8);
    assert.equal(usage.extensions[key].observed.cached_input_tokens, retry === "empty" ? 8 : 6);
    assert.ok(usage.extensions[key].gaps.some(gap => gap === "scope_unverified"));
  });
}


test("pinned AgentRuntime: reasoning usage survives a retry followed by stream failure", { skip: !source }, async () => {
  const { AgentRuntime } = await import(pathToFileURL(source).href);
  let requests = 0;
  const runtime = new AgentRuntime({
    model: { async *stream() {
      requests++;
      if (requests === 1) {
        yield { type: "finish", reason: "error", error: "controlled transient error", errorRetryable: true };
        return;
      }
      yield { type: "usage", usage: { inputTokens: 40, outputTokens: 8, reasoningTokenCount: 3 } };
      throw new Error("controlled stream failure after usage");
    } },
    messageModelInfo: { id: "test-model", provider: "openai" }, hooks: clineUsageHooks, tools: [],
  });
  const evidence = new ClineInvocationUsage();
  const result = await evidence.run(() => runtime.run("one"));
  assert.equal(requests, 2);
  assert.equal(result.status, "failed");
  assert.deepEqual(evidence.finish().extensions[key].observed, {
    input_tokens: 40, output_tokens: 8, reasoning_tokens: 3,
  });
  assert.ok(evidence.finish().extensions[key].gaps.includes("scope_unverified"));
});
