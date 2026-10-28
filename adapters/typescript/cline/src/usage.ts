// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import { AsyncLocalStorage } from "node:async_hooks";
import type { AgentUsage, JsonObject } from "nemo-fabric-adapter-contract";

type Metrics = Record<string, unknown>;
type Counters = Record<string, number>;
const key = "nvidia.fabric.cline.usage_evidence.v1";
const fields = [
  ["inputTokens", "input_tokens"],
  ["outputTokens", "output_tokens"],
  ["cacheReadTokens", "cached_input_tokens"],
  ["cacheWriteTokens", "cache_write_input_tokens"],
  ["reasoningTokenCount", "reasoning_tokens"],
] as const;
const owner = new AsyncLocalStorage<ClineInvocationUsage>();

function metrics(value: unknown): Metrics {
  return typeof value === "object" && value !== null ? value as Metrics : {};
}

function observedCounters(value: unknown): Counters {
  const source = metrics(value);
  const counters: Counters = {};
  for (const [field, destination] of fields) {
    const count = source[field];
    // Cline 0.0.83 replaces missing counts with zero before its public hooks.
    if (typeof count === "number" && Number.isSafeInteger(count) && count > 0) {
      counters[destination] = count;
    }
  }
  return counters;
}

export const clineUsageHooks = {
  beforeModel(context?: { snapshot?: { usage?: unknown } }): void {
    owner.getStore()?.started(context?.snapshot?.usage);
  },
  afterModel(context: { assistantMessage: unknown }): void {
    owner.getStore()?.completed(context.assistantMessage);
  },
  onEvent(event: { type: string; usage?: unknown }): void {
    if (event.type === "usage-updated") owner.getStore()?.updated(event.usage);
  },
};

/** Retain hook observations without claiming coverage of physical requests. */
export class ClineInvocationUsage {
  private readonly completedCalls = new Map<string, Counters>();
  private readonly unidentifiedCalls: Counters[] = [];
  private readonly pending: { baseline: Metrics; latest?: Metrics; first: boolean }[] = [];
  private startedCount = 0;
  private snapshot?: AgentUsage;

  run<T>(operation: () => Promise<T>): Promise<T> {
    return owner.run(this, operation);
  }

  started(value?: unknown): void {
    if (this.snapshot !== undefined) return;
    this.pending.push({ baseline: { ...metrics(value) }, first: this.startedCount++ === 0 });
  }

  updated(value: unknown): void {
    if (this.snapshot !== undefined) return;
    const call = this.pending.at(-1);
    if (call !== undefined) call.latest = { ...metrics(value) };
  }

  completed(value: unknown): void {
    if (this.snapshot !== undefined) return;
    const message = metrics(value);
    const id = typeof message.id === "string" && message.id.length > 0 ? message.id : undefined;
    if (id !== undefined && this.completedCalls.has(id)) return;
    // An ID-less completion can close a pending call, but cannot be deduplicated.
    if (this.pending.length === 0) return;
    this.pending.pop();
    const counters = observedCounters(message.metrics);
    if (id === undefined) this.unidentifiedCalls.push(counters);
    else this.completedCalls.set(id, counters);
  }

  finish(terminalUsage?: unknown): AgentUsage {
    if (this.snapshot !== undefined) return this.snapshot;
    const calls = [...this.completedCalls.values(), ...this.unidentifiedCalls];
    const gaps = new Set(["scope_unverified"]);
    if (calls.length === 0 && this.pending.length === 0) {
      calls.push(observedCounters(terminalUsage));
    }
    for (const call of this.pending) {
      const delta: Metrics = {};
      // Cline's initial snapshot omits reasoning until its first usage update.
      // A retry before that update still has a known zero reasoning baseline.
      const initialSnapshot = ["inputTokens", "outputTokens", "cacheReadTokens", "cacheWriteTokens"]
        .every(field => call.baseline[field] === 0);
      // usage-updated is cumulative within one execute(). Subtract only a
      // supported baseline so an unfinished call cannot repeat earlier usage.
      for (const [field] of fields) {
        const startsAtZero = call.first || (field === "reasoningTokenCount" && initialSnapshot);
        const before = call.baseline[field] ?? (startsAtZero ? 0 : undefined);
        const after = call.latest?.[field];
        if (typeof before === "number" && Number.isSafeInteger(before) && before >= 0 &&
            typeof after === "number" && Number.isSafeInteger(after) && after >= before) {
          delta[field] = after - before;
        }
      }
      calls.push(observedCounters(delta));
      gaps.add("unfinished_work");
    }
    const observed: JsonObject = {};
    const overflow = new Set<string>();
    for (const counters of calls) {
      if (counters.input_tokens === undefined || counters.output_tokens === undefined) gaps.add("missing_usage");
      for (const [field, count] of Object.entries(counters)) {
        if (overflow.has(field)) continue;
        const sum = Number(observed[field] ?? 0) + count;
        if (Number.isSafeInteger(sum)) observed[field] = sum;
        else {
          delete observed[field];
          overflow.add(field);
          gaps.add("missing_usage");
        }
      }
    }
    this.snapshot = { extensions: { [key]: { observed, gaps: [...gaps] } } };
    return this.snapshot;
  }
}
