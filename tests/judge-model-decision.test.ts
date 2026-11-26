/**
 * 决策模型判定：Pi 的 classifier 类型模型（例如 TypeSafe 的 jev-latest）不生成文本，
 * 只回答类型化问题。这里验证它如何被查找、如何被提问、答案如何换算成判定。
 */
import assert from "node:assert/strict";
import { test } from "node:test";
import type { TurnSnapshot } from "../src/session-context.ts";
import { i18n } from "../src/i18n.ts";
import {
  createJudgeModelInvoker,
  createJudgeModelSource,
  isDecisionModel,
  piJudgeCompletion,
  type JudgeClassifier,
  type JudgeCompletion,
  type JudgeDecisionResult,
  type JudgeModelSource,
  type PiModel,
} from "../src/judge-model.ts";
import { parseJudgeVerdict } from "../src/verdict.ts";
import { parseConfig } from "../src/config.ts";

const SNAPSHOT: TurnSnapshot = { userRequest: "任务", userAnswers: [], finalOutput: "做了一半", toolTrace: [] };

/** 测试用对话模型。 */
const CHAT_MODEL: PiModel = {
  id: "judge-model",
  name: "Judge Model",
  api: "openai-completions",
  provider: "test",
  baseUrl: "https://example.invalid",
  reasoning: false,
  input: ["text"],
  cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
  contextWindow: 8192,
  maxTokens: 4096,
};

/** 测试用决策模型：Pi 给它的类型是 classifier。 */
const DECISION_MODEL = { ...CHAT_MODEL, id: "jev-latest", provider: "typesafe", type: "classifier" } as unknown as PiModel;

/** 一次决策模型调用的可观察入参。 */
interface CapturedCall {
  model: PiModel;
  state: Record<string, unknown>;
  questions: Record<string, { type: string; instructions: string; criteria: Record<string, string> }>;
  apiKey?: string;
}

/** 记录每次调用的决策模型替身；answers 决定它这次返回什么。 */
function capturingClassifier(answers: JudgeDecisionResult["answers"], stopReason = "stop", errorMessage?: string) {
  const calls: CapturedCall[] = [];
  const classify: JudgeClassifier = async (model, context, options) => {
    calls.push({ model, state: context.state, questions: context.questions, apiKey: options.apiKey });
    return { provider: "typesafe", model: "jev-latest", answers, stopReason, errorMessage };
  };
  return { classify, calls };
}

/** 组装判定模型来源：默认解析出决策模型并带上给定的决策模型替身。 */
function decisionSource(classify?: JudgeClassifier, overrides: Partial<JudgeModelSource> = {}): JudgeModelSource {
  return {
    configuredModel: "typesafe/jev-latest",
    sessionModel: CHAT_MODEL,
    findModel: () => DECISION_MODEL,
    resolveAuth: async () => ({ ok: true, apiKey: "key" }),
    classify,
    ...overrides,
  };
}

function choiceAnswer(decision: string, confidence: number, rule: string) {
  const other = Number((1 - confidence).toFixed(2));
  return {
    decision: {
      type: "choice" as const,
      choice: decision,
      confidence,
      probabilities: { stop: decision === "stop" ? confidence : other, continue: decision === "continue" ? confidence : other },
    },
    rule: { type: "choice" as const, choice: rule },
  };
}

test("isDecisionModel 只认 Pi 的 classifier 类型", () => {
  assert.equal(isDecisionModel(DECISION_MODEL), true);
  assert.equal(isDecisionModel(CHAT_MODEL), false);
});

test("createJudgeModelSource 在 find 找不到时改用 findOfType(classifier)", () => {
  const lookups: string[] = [];
  const ctx = {
    model: CHAT_MODEL,
    modelRegistry: {
      find: (provider: string, modelId: string) => {
        lookups.push(`find ${provider}/${modelId}`);
        return provider === "test" ? CHAT_MODEL : undefined;
      },
      findOfType: (type: string, provider: string, modelId: string) => {
        lookups.push(`findOfType ${type} ${provider}/${modelId}`);
        return type === "classifier" && provider === "typesafe" && modelId === "jev-latest" ? DECISION_MODEL : undefined;
      },
      getApiKeyAndHeaders: async () => ({ ok: true as const, apiKey: "key" }),
    },
    sessionManager: { getSessionId: () => "session-decision" },
  };
  const source = createJudgeModelSource(
    ctx as unknown as Parameters<typeof createJudgeModelSource>[0],
    parseConfig({ model: "typesafe/jev-latest" }),
  );

  // 对话模型仍由 find 解析，不多查一次。
  assert.equal(lookup(source, "test/judge-model")?.id, "judge-model");
  assert.deepEqual(lookups, ["find test/judge-model"]);
  lookups.length = 0;

  // 决策模型 find 查不到，必须落到 findOfType 才能找到。
  assert.equal(lookup(source, "typesafe/jev-latest")?.id, "jev-latest");
  assert.deepEqual(lookups, ["find typesafe/jev-latest", "findOfType classifier typesafe/jev-latest"]);
});

function lookup(source: JudgeModelSource, reference: string): PiModel | undefined {
  const [provider, modelId] = reference.split("/");
  return source.findModel(provider, modelId);
}

test("决策模型：一次调用问两个问题，答案换算成判定，所选规则进入理由", async () => {
  const { classify, calls } = capturingClassifier(choiceAnswer("continue", 0.91, "plan_not_executed"));
  const invoke = createJudgeModelInvoker({ source: decisionSource(classify) });

  const response = await invoke({ snapshot: SNAPSHOT, signal: undefined });
  const verdict = parseJudgeVerdict(response.text);

  assert.deepEqual(verdict, {
    decision: "continue",
    confidence: 0.91,
    reason: i18n.t("judgeDecisionContinueReason", {
      rule: i18n.t("judgeRulePlanNotExecuted"),
      stop: "0.09",
      continue: "0.91",
    }),
  });
  assert.equal(response.stopReason, "stop");

  // 同一次调用里问「是否可停」与「适用哪条规则」；规则与提示词里的条目一一对应。
  assert.equal(calls.length, 1);
  assert.deepEqual(Object.keys(calls[0].questions), ["decision", "rule"]);
  assert.deepEqual(Object.keys(calls[0].questions.decision.criteria), ["stop", "continue"]);
  assert.equal(Object.keys(calls[0].questions.rule.criteria).length, 11);
  // 规则全文与本轮内容都交给决策模型，鉴权头随请求传入。
  assert.ok(String(calls[0].state.rules).includes(i18n.t("judgeSystemPrompt")));
  assert.match(String(calls[0].state.turn), /做了一半/);
  assert.equal(calls[0].apiKey, "key");
});

test("决策模型：stop 结论带匹配的规则", async () => {
  const { classify } = capturingClassifier(choiceAnswer("stop", 0.97, "waiting_background"));
  const invoke = createJudgeModelInvoker({ source: decisionSource(classify) });

  const verdict = parseJudgeVerdict((await invoke({ snapshot: SNAPSHOT, signal: undefined })).text);
  assert.equal(verdict?.decision, "stop");
  assert.equal(verdict?.confidence, 0.97);
  assert.ok((verdict?.reason ?? "").includes(i18n.t("judgeRuleWaitingBackground")));
});

test("决策模型：所选规则与结论矛盾时不用它，改用概括的说法", async () => {
  // 结论是继续，却选了「可以停下」的规则：催促不能带着「请求已完成」这种错误原因。
  const premature = capturingClassifier(choiceAnswer("continue", 0.9, "request_complete"));
  const first = parseJudgeVerdict(
    (await createJudgeModelInvoker({ source: decisionSource(premature.classify) })({ snapshot: SNAPSHOT, signal: undefined })).text,
  );
  assert.equal(first?.decision, "continue");
  assert.match(first?.reason ?? "", /\{rule\}|stopped with steps of the request still unfinished|请求仍有步骤未完成/);

  // 结论是停下，却选了「必须继续」的规则：同样不用它。
  const finished = capturingClassifier(choiceAnswer("stop", 0.9, "partial"));
  const second = parseJudgeVerdict(
    (await createJudgeModelInvoker({ source: decisionSource(finished.classify) })({ snapshot: SNAPSHOT, signal: undefined })).text,
  );
  assert.equal(second?.decision, "stop");
  assert.doesNotMatch(second?.reason ?? "", /Only part of a multi-step request|多步请求只完成了一部分/);
});

test("决策模型：置信度取所选选项的概率，而不是分布集中度", async () => {
  // 两选项时模型的 confidence 是 2 x P(top) - 1：0.36 对应 0.68。直接用它会把阈值翻倍。
  const { classify } = capturingClassifier({
    decision: { type: "choice", choice: "continue", confidence: 0.36, probabilities: { stop: 0.32, continue: 0.68 } },
    rule: { type: "choice", choice: "partial" },
  });
  const invoke = createJudgeModelInvoker({ source: decisionSource(classify) });
  const verdict = parseJudgeVerdict((await invoke({ snapshot: SNAPSHOT, signal: undefined })).text);
  assert.equal(verdict?.confidence, 0.68);

  // 没有概率时，从集中度换算回概率。
  const marginOnly = capturingClassifier({
    decision: { type: "choice", choice: "stop", confidence: 0.5 },
    rule: { type: "choice", choice: "blocked" },
  });
  const fromMargin = parseJudgeVerdict(
    (await createJudgeModelInvoker({ source: decisionSource(marginOnly.classify) })({ snapshot: SNAPSHOT, signal: undefined })).text,
  );
  assert.equal(fromMargin?.confidence, 0.75);
});

test("决策模型：出错或没有返回选择时按失败上报，不算作判定", async () => {
  const failed = capturingClassifier({}, "error", "401 unauthorized");
  const errorResponse = await createJudgeModelInvoker({ source: decisionSource(failed.classify) })({
    snapshot: SNAPSHOT,
    signal: undefined,
  });
  assert.equal(errorResponse.stopReason, "error");
  assert.equal(errorResponse.text, "");
  assert.equal(errorResponse.errorMessage, "401 unauthorized");

  const noChoice = capturingClassifier({ decision: { type: "score", score: 1, confidence: 1 } });
  const emptyResponse = await createJudgeModelInvoker({ source: decisionSource(noChoice.classify) })({
    snapshot: SNAPSHOT,
    signal: undefined,
  });
  assert.equal(parseJudgeVerdict(emptyResponse.text), undefined);
  assert.match(emptyResponse.errorMessage ?? "", /no choice|没有返回选项/);
});

test("决策模型：运行时没有 classify 时明确报错，不静默换模型", async () => {
  await assert.rejects(
    piJudgeCompletion({
      model: DECISION_MODEL,
      systemPrompt: "rules",
      userPrompt: "turn",
      auth: { ok: true, apiKey: "key" },
      maxTokens: 0,
    }),
    /no classify\(\) call|没有提供 classify\(\)/,
  );
});

test("决策模型：没有输出预算，也不走截断重试", async () => {
  let completions = 0;
  const truncated: JudgeCompletion = async () => {
    completions += 1;
    return { text: "", stopReason: "length", partTypes: [] };
  };
  // 即便替身返回「被截断」，也只调用一次：决策模型没有可以加大的输出预算。
  await createJudgeModelInvoker({ source: decisionSource(), completion: truncated })({ snapshot: SNAPSHOT, signal: undefined });
  assert.equal(completions, 1);

  // 对话模型保持原有行为：截断且无文本时用翻倍预算重试一次。
  let chatCompletions = 0;
  const chatBudgets: number[] = [];
  const chatTruncated: JudgeCompletion = async (options) => {
    chatCompletions += 1;
    chatBudgets.push(options.maxTokens);
    return { text: "", stopReason: "length", partTypes: [] };
  };
  await createJudgeModelInvoker({
    source: { ...decisionSource(undefined, { configuredModel: "test/judge-model" }), findModel: () => CHAT_MODEL },
    completion: chatTruncated,
    // 给出小于模型上限的预算，截断后才有空间翻倍重试。
    maxTokens: 1000,
  })({ snapshot: SNAPSHOT, signal: undefined });
  assert.equal(chatCompletions, 2);
  assert.deepEqual(chatBudgets, [1000, 2000]);
});
