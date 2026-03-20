import { existsSync } from "node:fs";
import * as path from "node:path";
import OpenAI from "openai";
import * as dotenv from "dotenv";
import { consola } from "consola";

type ModelMode = "groq" | "grow" | "ollama";

type LlmMessage = {
	role: "system" | "user" | "assistant";
	content: string;
};

type OllamaChatResponse = {
	message?: { role: "assistant"; content: string };
};

const logger = consola.withTag("office-llm");

const loadEnvironment = (): void => {
	const cwd = process.cwd();
	const envPaths = [path.resolve(cwd, ".env"), path.resolve(cwd, "..", ".env")];

	for (const envPath of envPaths) {
		dotenv.config({ path: envPath, override: false });
	}
};

loadEnvironment();

const GROQ_TOKEN = (process.env.GROQ_API_KEY ?? "").trim();
const OLLAMA_HOST = process.env.OLLAMA_HOST ?? "http://localhost:11434";
const OLLAMA_MODEL = process.env.OLLAMA_MODEL ?? "qwen3:8b";
const GROQ_BASE_URL = "https://api.groq.com/openai/v1";
const GROQ_MODELS = [
	"openai/gpt-oss-120b",
	"llama-3.3-70b-versatile",
	"qwen/qwen3-32b",
	"meta-llama/llama-4-scout-17b-16e-instruct",
] as const;

const resolveModelMode = (): ModelMode => {
	const mode = (process.env.MODEL_MODE ?? "ollama").trim().toLowerCase();
	if (mode === "groq" || mode === "grow" || mode === "ollama") {
		return mode;
	}
	return "ollama";
};

const MODEL_MODE: ModelMode = resolveModelMode();

const parseErrorMessage = (error: unknown): string => {
	if (error instanceof Error) {
		return error.message;
	}
	return String(error);
};

const isModelUnavailableError = (error: unknown): boolean => {
	const message = parseErrorMessage(error).toLowerCase();
	return (
		message.includes("model")
		&& (message.includes("not found") || message.includes("unavailable") || message.includes("decommissioned"))
	);
};

const assertModelConfig = (): void => {
	if ((MODEL_MODE === "groq" || MODEL_MODE === "grow") && !GROQ_TOKEN) {
		throw new Error("GROQ_API_KEY is not set in process.env for groq/grow mode");
	}
};

const toGroqInput = (messages: LlmMessage[]): string => {
	const chunks = messages.map(({ role, content }) => `${role.toUpperCase()}:\n${content}`);
	return chunks.join("\n\n");
};

const groqClient = new OpenAI({
	apiKey: GROQ_TOKEN,
	baseURL: GROQ_BASE_URL,
});

const fetchOllama = async (messages: LlmMessage[]): Promise<string> => {
	const response = await fetch(`${OLLAMA_HOST}/api/chat`, {
		method: "POST",
		headers: { "content-type": "application/json" },
		body: JSON.stringify({
			model: OLLAMA_MODEL,
			stream: false,
			messages,
		}),
	});

	if (!response.ok) {
		const body = await response.text();
		throw new Error(`Ollama request failed (${response.status}): ${body}`);
	}

	const data = (await response.json()) as OllamaChatResponse;
	return data.message?.content?.trim() ?? "";
};

const fetchGroq = async (messages: LlmMessage[]): Promise<string> => {
	if (!GROQ_TOKEN) {
		throw new Error("GROQ_TOKEN or GROQ_API_KEY is not set for groq/grow mode");
	}

	const input = toGroqInput(messages);
	const errorsByModel: string[] = [];

	for (const model of GROQ_MODELS) {
		try {
			const response = await groqClient.responses.create({
				model,
				input,
			});
			const outputText = response.output_text?.trim() ?? "";
			if (outputText) {
				return outputText;
			}
			errorsByModel.push(`${model}: empty response`);
		} catch (error) {
			const message = parseErrorMessage(error);
			errorsByModel.push(`${model}: ${message}`);
			if (!isModelUnavailableError(error)) {
				throw new Error(`Groq request failed on model ${model}: ${message}`);
			}
		}
	}

	throw new Error(`No available models in Groq: ${errorsByModel.join(" | ")}`);
};

const fetchModelResponse = async (messages: LlmMessage[]): Promise<string> => {
	if (MODEL_MODE === "ollama") {
		return fetchOllama(messages);
	}
	return fetchGroq(messages);
};

export interface AgentReplyInput {
	agentName: string;
	systemPrompt: string;
	mcpAccess: string[];
	peers: string[];
	from: string;
	message: string;
	memories: string[];
}

export const checkLlmConnection = async (): Promise<void> => {
	assertModelConfig();

	const testMessages: LlmMessage[] = [
		{
			role: "system",
			content: "You are a connectivity health-check assistant. Reply with a short OK.",
		},
		{
			role: "user",
			content: "Health check",
		},
	];

	const response = await fetchModelResponse(testMessages);
	if (!response.trim()) {
		throw new Error("LLM health check returned an empty response");
	}
};

const resolveProjectRoot = (): string => {
	const cwd = process.cwd();
	if (existsSync(path.join(cwd, "office.config.json"))) {
		return cwd;
	}
	if (existsSync(path.join(cwd, "..", "office.config.json"))) {
		return path.resolve(cwd, "..");
	}
	return cwd;
};

const PROJECT_ROOT = resolveProjectRoot();

export const generateAgentReply = async (input: AgentReplyInput): Promise<string> => {
	assertModelConfig();

	const mainTargetHint = `Project root: ${PROJECT_ROOT}`;
	const system = [
		input.systemPrompt,
		`You are agent: ${input.agentName}`,
		`MCP access: ${input.mcpAccess.join(", ") || "none"}`,
		`Known office peers: ${input.peers.join(", ") || "none"}`,
		mainTargetHint,
		"You must be proactive, never passive.",
		"Operate in autonomous mode: do not ask for permission, confirmation, or prioritization.",
		"The leader decides and assigns next tasks with available context.",
		"If information is incomplete, make explicit assumptions and continue execution.",
		"Do not answer with generic status phrases like 'everything is good' or 'ready to help'.",
		"Always provide concrete next steps and who should do them.",
		"Response format is mandatory:",
		"1) STATUS: one-sentence current status",
		"2) FINDINGS: 1-3 concrete findings or assumptions",
		"3) NEXT_ACTIONS: at least 2 explicit actions you will execute now",
		"4) DELEGATIONS: optional, include exact peer names and what they should do",
		"5) REQUESTS: must be 'none' unless a hard technical blocker exists",
		"Keep answers concise and action-oriented.",
	].join("\n");

	const user = [
		`From: ${input.from}`,
		`Message: ${input.message}`,
		`Relevant memories: ${input.memories.length ? input.memories.join(" | ") : "none"}`,
	].join("\n");

	const messages: LlmMessage[] = [
		{ role: "system", content: system },
		{ role: "user", content: user },
	];

	const response = await fetchModelResponse(messages);
	if (!response.trim()) {
		logger.warn("LLM returned empty response, using minimal fallback");
		return `Agent ${input.agentName}: I don't have a useful response right now.`;
	}
	return response.trim();
};
