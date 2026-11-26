/**
 * 判定模型接入层：解析用哪个模型、取鉴权信息、发起 completion 调用。
 *
 * 模型类型与鉴权细节都收在本模块，判定流程只依赖 verdict.ts 的窄接口。
 */
import { completeSimple } from "@earendil-works/pi-ai/compat";
import type { ExtensionContext } from "@earendil-works/pi-coding-agent";
import { resolveModelRequestAuth, type ModelRequestAuth } from "pi-model-request";
import { i18n } from "./i18n.ts";
import {
  buildJudgeSystemPrompt,
  buildJudgeUserPrompt,
  STOP_REASON_LENGTH,
  type JudgeInvoker,
  type JudgeResponse,
} from "./verdict.ts";
import type { AutoGoalConfig } from "./config.ts";

/**
 * 判定调用的思考强度。
 * 判定只是一次短分类，思考会先花掉输出预算；
 * Pi 会把该值收敛到模型支持的最低档，不支持关闭思考的模型也不会报错。
 */
const JUDGE_REASONING_LEVEL = "minimal";
/** 输出被截断时，用翻倍预算重试一次；仍失败则按错误上报。 */
const TRUNCATION_RETRY_MULTIPLIER = 2;

/** Pi 的模型对象类型；对外暴露以便调用方构造判定来源。 */
export type PiModel = Parameters<typeof completeSimple>[0];

/** Pi 把决策模型标成这个类型；它们不生成文本，只回答类型化问题。 */
const CLASSIFIER_MODEL_TYPE = "classifier";
/** 交给决策模型的两个问题：是否可以停下，以及适用哪条规则；答案按名字取回。 */
const DECISION_QUESTION = "decision";
const RULE_QUESTION = "rule";
/** 规则里「可以停下」的五条与「必须继续」的六条；键名对应 locales 里的 judgeRule* 文案。 */
const STOP_RULES = ["request_complete", "needs_user", "blocked", "question_answered", "waiting_background"];
const CONTINUE_RULES = [
  "partial",
  "plan_not_executed",
  "offered_to_continue",
  "unverified_claim",
  "unrequested_confirmation",
  "change_only_restated",
];

/** 鉴权结果；ok 为 false 时判定不可执行。与共享请求器的鉴权结果同形，避免两处定义漂移。 */
export type JudgeAuth = ModelRequestAuth;

/**
 * 模型解析所需的最小字段；JudgeModelSource 在它之上增加鉴权解析。
 * 泛型化后测试可以用简单值验证解析逻辑，无需构造真实模型对象。
 */
export interface JudgeModelResolution<T> {
  /** 配置里指定的模型标识；空字符串表示复用当前会话模型。 */
  configuredModel: string;
  /** 当前会话模型；未选择模型时为 undefined。 */
  sessionModel: T | undefined;
  /** 按 provider 与 modelId 查找模型。 */
  findModel: (provider: string, modelId: string) => T | undefined;
}

/** 决策模型收到的问题：在给定的选项里选一个。 */
export interface JudgeChoiceQuestion {
  type: "choice";
  instructions: string;
  criteria: Record<string, string>;
}

/**
 * modelRegistry 上与决策模型有关的两个方法。较新的 Pi 才提供，本包的类型检查
 * 钉在较旧的 Pi 类型上，所以按结构声明并保留「可能不存在」，运行时再确认。
 */
interface DecisionRegistry {
  findOfType?: (type: "classifier", provider: string, modelId: string) => unknown;
  classify?: (model: unknown, context: unknown, options?: unknown) => Promise<JudgeDecisionResult>;
}

/** 决策模型答案的最小结构；完整类型见 pi-ai 的 ClassifierResult。 */
export interface JudgeDecisionResult {
  provider: string;
  model: string;
  answers: Record<
    string,
    {
      type: string;
      choice?: string;
      confidence?: number;
      probabilities?: Record<string, number>;
      [field: string]: unknown;
    }
  >;
  stopReason: string;
  errorMessage?: string;
}

/**
 * 决策模型调用契约：默认接 Pi 的 ctx.modelRegistry.classify，
 * 它与 completion 一样使用已配置的 provider 并解析鉴权；测试可替换。
 */
export type JudgeClassifier = (
  model: PiModel,
  context: {
    state: Record<string, unknown>;
    questions: Record<string, JudgeChoiceQuestion>;
  },
  options: {
    apiKey?: string;
    headers?: NonNullable<Extract<JudgeAuth, { ok: true }>["headers"]>;
    signal?: AbortSignal;
  },
) => Promise<JudgeDecisionResult>;

/** 判定模型来源：决定用哪个模型、如何鉴权、如何调用决策模型；测试可以直接构造。 */
export interface JudgeModelSource extends JudgeModelResolution<PiModel> {
  /** 取指定模型的鉴权信息。 */
  resolveAuth: (model: PiModel) => Promise<JudgeAuth>;
  /** 调用决策模型；只在配置的模型是 classifier 类型时用到，可以为空。 */
  classify?: JudgeClassifier;
}

/** 底层 completion 契约；Pi 的 complete 与测试替身都实现它。 */
export type JudgeCompletion = (options: {
  /** 判定模型。 */
  model: PiModel;
  /** 系统提示词。 */
  systemPrompt: string;
  /** 用户提示词。 */
  userPrompt: string;
  /** 已解析出的鉴权信息。 */
  auth: Extract<JudgeAuth, { ok: true }>;
  /** 输出 token 上限。 */
  maxTokens: number;
  /** 中止信号。 */
  signal?: AbortSignal;
  /** 决策模型调用；模型是 classifier 类型时必须有。 */
  classify?: JudgeClassifier;
}) => Promise<JudgeResponse>;

/** 拼接响应里的文本块，忽略思考块与工具调用块。 */
function extractResponseText(content: ReadonlyArray<{ type: string; text?: string }>): string {
  return content
    .filter((part) => part.type === "text" && typeof part.text === "string")
    .map((part) => part.text ?? "")
    .join("\n");
}

/** 汇总响应内容块（如 thinking:174），用于空响应时说明模型到底返回了什么。 */
function summarizeParts(
  content: ReadonlyArray<{ type: string; text?: string; thinking?: string }>,
): string[] {
  return content.map((part) => `${part.type}:${(part.text ?? part.thinking ?? "").length}`);
}

/** 模型是否为 Pi 的决策模型（classifier 类型）。 */
export function isDecisionModel(model: PiModel): boolean {
  return (model as { type?: unknown }).type === CLASSIFIER_MODEL_TYPE;
}

function formatProbability(value: number | undefined): string {
  return typeof value === "number" && Number.isFinite(value) ? value.toFixed(2) : "n/a";
}

/** 交给决策模型挑选的规则清单，文案与提示词里的规则一一对应。 */
function ruleCriteria(): Record<string, string> {
  return {
    request_complete: i18n.t("judgeRuleRequestComplete"),
    needs_user: i18n.t("judgeRuleNeedsUser"),
    blocked: i18n.t("judgeRuleBlocked"),
    question_answered: i18n.t("judgeRuleQuestionAnswered"),
    waiting_background: i18n.t("judgeRuleWaitingBackground"),
    partial: i18n.t("judgeRulePartial"),
    plan_not_executed: i18n.t("judgeRulePlanNotExecuted"),
    offered_to_continue: i18n.t("judgeRuleOfferedToContinue"),
    unverified_claim: i18n.t("judgeRuleUnverifiedClaim"),
    unrequested_confirmation: i18n.t("judgeRuleUnrequestedConfirmation"),
    change_only_restated: i18n.t("judgeRuleChangeOnlyRestated"),
  };
}

/**
 * 用决策模型判定：把同一套判定规则作为状态交给它，
 * 让它在 stop / continue 两个选项里选一个，并指出适用哪条规则，
 * 再把两个答案换算成判定流程期望的 JSON 文本。
 * 决策模型不写理由：催促里的「还差什么」取自它选中的规则；
 * 所选规则与结论矛盾时不用它，改用概括的说法，避免催促带着错误的原因。
 */
async function decisionJudgeCompletion(options: {
  model: PiModel;
  systemPrompt: string;
  userPrompt: string;
  auth: Extract<JudgeAuth, { ok: true }>;
  classify: JudgeClassifier;
  signal?: AbortSignal;
}): Promise<JudgeResponse> {
  const { model, systemPrompt, userPrompt, auth, classify, signal } = options;
  const result = await classify(
    model,
    {
      state: { rules: systemPrompt, turn: userPrompt },
      questions: {
        [DECISION_QUESTION]: {
          type: "choice",
          instructions: i18n.t("judgeDecisionInstructions"),
          criteria: {
            stop: i18n.t("judgeCriterionStop"),
            continue: i18n.t("judgeCriterionContinue"),
          },
        },
        [RULE_QUESTION]: {
          type: "choice",
          instructions: i18n.t("judgeRuleInstructions"),
          criteria: ruleCriteria(),
        },
      },
    },
    { apiKey: auth.apiKey, headers: auth.headers, signal },
  );
  const answer = result.answers[DECISION_QUESTION];
  if (result.stopReason !== "stop" || answer?.type !== "choice" || typeof answer.choice !== "string") {
    return {
      text: "",
      stopReason: result.stopReason,
      errorMessage:
        result.errorMessage ??
        i18n.t("judgeDecisionNoChoice", { answer: answer?.type ?? "none" }),
      partTypes: [`decision:${answer?.type ?? "none"}`],
    };
  }
  const decision = answer.choice;
  const probabilities = answer.probabilities ?? {};
  const confidence = Number.isFinite(answer.confidence)
    ? (answer.confidence as number)
    : (probabilities[decision] ?? 0);

  const ruleAnswer = result.answers[RULE_QUESTION];
  const ruleKey =
    ruleAnswer?.type === "choice" && typeof ruleAnswer.choice === "string" ? ruleAnswer.choice : undefined;
  const rules = ruleCriteria();
  const consistent =
    ruleKey !== undefined &&
    rules[ruleKey] !== undefined &&
    (decision === "continue" ? CONTINUE_RULES.includes(ruleKey) : STOP_RULES.includes(ruleKey));
  const rule = consistent
    ? rules[ruleKey]
    : decision === "continue"
      ? i18n.t("judgeDecisionRuleUnknown")
      : i18n.t("judgeDecisionStopGeneric");

  const detail = {
    rule,
    stop: formatProbability(probabilities.stop),
    continue: formatProbability(probabilities.continue),
  };
  const reason =
    decision === "continue"
      ? i18n.t("judgeDecisionContinueReason", detail)
      : i18n.t("judgeDecisionStopReason", detail);
  return {
    text: JSON.stringify({ decision, confidence, reason }),
    stopReason: "stop",
    partTypes: ["decision:choice", `rule:${ruleKey ?? "none"}`],
  };
}

/**
 * 解析要使用的判定模型：配置了 model 就按 provider/modelId 查找，否则用当前会话模型。
 * 泛型形式便于用简单值验证解析逻辑，无需构造真实模型对象。
 */
export function resolveJudgeModel<T>(source: JudgeModelResolution<T>): T | undefined {
  const configured = source.configuredModel.trim();
  if (!configured) return source.sessionModel;
  const separator = configured.indexOf("/");
  return source.findModel(configured.slice(0, separator), configured.slice(separator + 1));
}

/**
 * 把 Pi 的扩展上下文包装成判定模型来源。
 * 鉴权与 provider 会话头统一走共享请求器，判定流程只拿到已备好的请求头；
 * 决策模型走 Pi 的 modelRegistry，与 completion 一样能用已配置的 provider。
 */
export function createJudgeModelSource(
  ctx: Pick<ExtensionContext, "model" | "modelRegistry" | "sessionManager">,
  config: AutoGoalConfig,
): JudgeModelSource {
  const registry = ctx.modelRegistry as unknown as DecisionRegistry;
  return {
    configuredModel: config.model,
    sessionModel: ctx.model,
    // find() 只查对话模型；决策模型要用 findOfType 才能找到。
    findModel: (provider: string, modelId: string) =>
      ctx.modelRegistry.find(provider, modelId) ??
      (typeof registry.findOfType === "function"
        ? (registry.findOfType("classifier", provider, modelId) as PiModel | undefined)
        : undefined),
    resolveAuth: (model: PiModel) => resolveModelRequestAuth(ctx, model),
    classify: (model, context, options) => {
      // 旧版 Pi 的 modelRegistry 没有 classify：明确报错，不静默换模型。
      if (typeof registry.classify !== "function") {
        const reference = `${(model as { provider?: string }).provider ?? ""}/${(model as { id?: string }).id ?? ""}`;
        return Promise.reject(new Error(i18n.t("judgeClassifierUnavailable", { model: reference })));
      }
      return registry.classify(model, context, options);
    },
  };
}

/**
 * 默认 completion：用 Pi 的 completeSimple 发一次判定请求；
 * 模型是决策模型时改走 classify。
 * 这里固定使用 Pi 的调用；需要替换实现时注入自己的 JudgeCompletion。
 */
export const piJudgeCompletion: JudgeCompletion = async ({
  model,
  systemPrompt,
  userPrompt,
  auth,
  maxTokens,
  signal,
  classify,
}) => {
  // 鉴权解析出 baseUrl 时按 Pi 核心的方式覆盖模型地址。
  const target = auth.baseUrl ? { ...model, baseUrl: auth.baseUrl } : model;
  if (isDecisionModel(target)) {
    if (!classify) {
      throw new Error(
        i18n.t("judgeClassifierUnavailable", {
          model: `${(model as { provider?: string }).provider ?? ""}/${(model as { id?: string }).id ?? ""}`,
        }),
      );
    }
    return decisionJudgeCompletion({
      model: target,
      systemPrompt,
      userPrompt,
      auth,
      classify,
      signal,
    });
  }
  const response = await completeSimple(
    target,
    {
      systemPrompt,
      messages: [{
        role: "user",
        content: [{ type: "text", text: userPrompt }],
        timestamp: Date.now(),
      }],
    },
    {
      apiKey: auth.apiKey,
      headers: auth.headers,
      env: auth.env,
      maxTokens,
      reasoning: JUDGE_REASONING_LEVEL,
      signal,
    },
  );
  return {
    text: extractResponseText(response.content),
    stopReason: response.stopReason,
    errorMessage: response.errorMessage,
    partTypes: summarizeParts(response.content),
  };
};

/**
 * 计算本次判定使用的输出预算：取配置值与模型输出上限的较小值。
 * 配置默认值本身不预留额度，只在真的需要时（重试）才翻倍并重新收敛到模型上限。
 */
export function resolveJudgeMaxTokens(model: PiModel, configured: number): number {
  return Math.max(1, Math.min(model.maxTokens, configured));
}

/** 判断响应是否为「被输出预算截断且没有任何文本」这种可重试的失败。 */
function isTruncatedWithoutText(response: JudgeResponse): boolean {
  return response.stopReason === STOP_REASON_LENGTH && response.text.trim() === "";
}

/**
 * 用判定模型来源与 completion 组装判定调用。
 * 模型不存在或鉴权失败时直接抛错，由上层报告给用户，不静默跳过；
 * 输出被截断且没有文本时，加大预算重试一次，避免一次偶发的截断就丢掉判定。
 * 决策模型没有输出预算，也不走这条重试。
 */
export function createJudgeModelInvoker(options: {
  /** 模型来源。 */
  source: JudgeModelSource;
  /** 可替换的底层 completion。 */
  completion?: JudgeCompletion;
  /** 判定输出 token 上限；默认取模型上限。 */
  maxTokens?: number;
}): JudgeInvoker {
  const { source, completion = piJudgeCompletion } = options;
  return async ({ snapshot, signal }) => {
    const model = resolveJudgeModel(source);
    if (model === undefined) {
      throw new Error(i18n.t("judgeModelMissing", {
        model: source.configuredModel || i18n.t("judgeModelCurrentFallback"),
      }));
    }
    const auth = await source.resolveAuth(model);
    if (auth.ok === false) throw new Error(i18n.t("judgeAuthFailed", { error: auth.error }));

    const request = {
      model,
      systemPrompt: buildJudgeSystemPrompt(),
      userPrompt: buildJudgeUserPrompt(snapshot),
      auth,
      signal,
      classify: source.classify,
    };
    const decisionModel = isDecisionModel(model);
    const maxTokens = decisionModel ? 0 : resolveJudgeMaxTokens(model, options.maxTokens ?? model.maxTokens);
    const response = await completion({ ...request, maxTokens });
    if (decisionModel || !isTruncatedWithoutText(response)) return response;

    const retryMaxTokens = resolveJudgeMaxTokens(model, maxTokens * TRUNCATION_RETRY_MULTIPLIER);
    if (retryMaxTokens <= maxTokens) return response;
    return completion({ ...request, maxTokens: retryMaxTokens });
  };
}
