import { readFileSync } from "node:fs";
import { test } from "node:test";
import assert from "node:assert/strict";
import ts from "typescript";

const source = readFileSync(new URL("../src/lib/researchProgress.ts", import.meta.url), "utf8");
const compiled = ts.transpileModule(source, { compilerOptions: { target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ES2022 } }).outputText;
const { researchProgressParts, friendlySimulationError, simulationNeedsNewRun } = await import(`data:text/javascript;base64,${Buffer.from(compiled).toString("base64")}`);

const start = { type: "agent_step", phase: "hypotheses", verdict: "running", note: "正在提炼…" };
test("empty and failed extraction replaces the running label, including history replay", () => {
  for (const verdict of ["empty", "completed", "failed", "timeout"]) {
    const end = { ...start, verdict, note: "阶段已结束" };
    assert.deepEqual(researchProgressParts([start, end], false), [end]);
    assert.deepEqual(researchProgressParts([start, end], true), [end]);
  }
});

test("stream termination without a result does not pretend extraction succeeded", () => {
  assert.deepEqual(researchProgressParts([start], true), [start]);
  const parts = researchProgressParts([start], false);
  assert.equal(parts[0].verdict, "interrupted");
  assert.match(parts[0].note, /未收到/);
});

test("old 13-character thinking label stops when its response has finished", () => {
  const old = { type: "thinking", agent: "router", text: "正在提炼可证伪的研究假设…" };
  assert.deepEqual(researchProgressParts([old], true), [old]);
  assert.equal(researchProgressParts([old], false)[0].verdict, "ended");
  assert.doesNotMatch(researchProgressParts([old], false)[0].note, /正在/);
});

test("both simulation views explain nested budget failures without filesystem paths", () => {
  const message = "RuntimeError: structured decision collection failed: local smoke token budget exhausted at 211293 tokens";
  assert.match(friendlySimulationError(message), /用量上限/);
  assert.match(friendlySimulationError(message), /重新推演/);
  assert.doesNotMatch(friendlySimulationError(message), /FileNotFoundError|211293/);
  assert.equal(simulationNeedsNewRun(message), true);
  assert.equal(simulationNeedsNewRun("temporary network failure"), false);
});
