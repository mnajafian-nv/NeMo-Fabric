// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { createRequire } from "node:module";
import test from "node:test";
import { ClineInvocationUsage, clineUsageHooks as hooks } from "../dist/usage.js";

const key = "nvidia.fabric.cline.usage_evidence.v1";
const evidence = value => value.extensions[key];
function completed(id, metrics) {
  hooks.beforeModel({ snapshot: { usage: {} } });
  hooks.afterModel({ assistantMessage: { id, metrics } });
}

test("logical calls add observations once while cache and reasoning remain breakdowns", async () => {
  const usage = new ClineInvocationUsage();
  await usage.run(async () => {
    completed("one", { inputTokens: 40, outputTokens: 8, cacheReadTokens: 6, cacheWriteTokens: 2, reasoningTokenCount: 3 });
    hooks.afterModel({ assistantMessage: { id: "one", metrics: { inputTokens: 40, outputTokens: 8 } } });
    completed("two", { inputTokens: 20, outputTokens: 4, cacheReadTokens: 1 });
  });
  assert.deepEqual(evidence(usage.finish()), {
    observed: { input_tokens: 60, output_tokens: 12, cached_input_tokens: 7, cache_write_input_tokens: 2, reasoning_tokens: 3 },
    gaps: ["scope_unverified"],
  });
});

test("zero-filled, omitted, and invalid counters do not become observations", async () => {
  const usage = new ClineInvocationUsage();
  await usage.run(async () => completed("one", {
    inputTokens: 0, outputTokens: -1, cacheReadTokens: 1.5, cacheWriteTokens: Number.MAX_SAFE_INTEGER + 1,
  }));
  assert.deepEqual(evidence(usage.finish()), { observed: {}, gaps: ["scope_unverified", "missing_usage"] });
});

test("unfinished calls subtract the before-model snapshot from cumulative updates", async () => {
  const usage = new ClineInvocationUsage();
  await usage.run(async () => {
    completed("one", { inputTokens: 40, outputTokens: 8 });
    hooks.beforeModel({ snapshot: { usage: { inputTokens: 40, outputTokens: 8 } } });
    hooks.onEvent({ type: "usage-updated", usage: { inputTokens: 60, outputTokens: 12 } });
  });
  assert.deepEqual(evidence(usage.finish({ inputTokens: 0, outputTokens: 0 })), {
    observed: { input_tokens: 60, output_tokens: 12 }, gaps: ["scope_unverified", "unfinished_work"],
  });
});

test("unfinished later calls without a baseline do not repeat earlier observations", async () => {
  const usage = new ClineInvocationUsage();
  await usage.run(async () => {
    completed("one", { inputTokens: 40, outputTokens: 8, reasoningTokenCount: 3 });
    hooks.beforeModel();
    hooks.onEvent({ type: "usage-updated", usage: { inputTokens: 60, outputTokens: 12, reasoningTokenCount: 5 } });
  });
  assert.deepEqual(evidence(usage.finish()).observed, { input_tokens: 40, output_tokens: 8, reasoning_tokens: 3 });
  assert.ok(evidence(usage.finish()).gaps.includes("unfinished_work"));
});

test("an ID-less completion closes its pending call and retains the observed metrics", async () => {
  const usage = new ClineInvocationUsage();
  await usage.run(async () => completed(undefined, { inputTokens: 40, outputTokens: 8 }));
  assert.deepEqual(evidence(usage.finish()), {
    observed: { input_tokens: 40, output_tokens: 8 }, gaps: ["scope_unverified"],
  });
});

test("counter overflow omits only the affected observation", async () => {
  const usage = new ClineInvocationUsage();
  await usage.run(async () => {
    completed("one", { inputTokens: Number.MAX_SAFE_INTEGER, outputTokens: 8 });
    completed("two", { inputTokens: 1, outputTokens: 4 });
  });
  assert.deepEqual(evidence(usage.finish()), {
    observed: { output_tokens: 12 }, gaps: ["scope_unverified", "missing_usage"],
  });
});

test("callbacks created by an old invocation cannot change its result or the next invocation", async () => {
  const oldUsage = new ClineInvocationUsage();
  let release;
  let late;
  await oldUsage.run(async () => {
    completed("old", { inputTokens: 40, outputTokens: 8 });
    late = new Promise(resolve => { release = resolve; }).then(() => completed("late", { inputTokens: 900, outputTokens: 900 }));
  });
  const saved = oldUsage.finish();
  const expected = structuredClone(saved);
  const next = new ClineInvocationUsage();
  await next.run(async () => {
    completed("new", { inputTokens: 20, outputTokens: 4 });
    release();
    await late;
  });
  assert.deepEqual(saved, expected);
  assert.deepEqual(evidence(next.finish()).observed, { input_tokens: 20, output_tokens: 4 });
});

test("descriptor accepts observed evidence and rejects zero-filled counts", async () => {
  const require = createRequire(import.meta.url);
  const Ajv = require("ajv/dist/2020.js").default;
  const descriptor = JSON.parse(await readFile(new URL("../cline.fabric-adapter.json", import.meta.url)));
  const validate = new Ajv().compile(descriptor.extension_schemas.usage);
  const usage = new ClineInvocationUsage().finish({ inputTokens: 40, outputTokens: 8 });
  assert.equal(validate(usage.extensions), true, JSON.stringify(validate.errors));
  assert.equal(validate({ [key]: { observed: { input_tokens: 0 }, gaps: ["scope_unverified"] } }), false);
});
