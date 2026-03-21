import { generateAgentReply } from "../llm";
import path from "node:path";

interface AgentProfile {
	name: string;
	systemPrompt: string;
	mcpAccess: string[];
	peers: string[];
}

interface MemoryRecord {
	text: string;
	metadata?: Record<string, unknown>;
}

interface WorkerMessage {
	type: string;
	[key: string]: unknown;
}

interface InternetSearchResult {
	title: string;
	url: string;
	snippet: string;
}

interface InternetSearchResponse {
	query: string;
	fetchedAt: string;
	results: InternetSearchResult[];
}

const pendingMemory = new Map<
	string,
	{ resolve: (value: unknown) => void; reject: (error: Error) => void }
>();
const pendingWorkspace = new Map<
	string,
	{ resolve: (value: unknown) => void; reject: (error: Error) => void }
>();

let profile: AgentProfile = {
	name: "unknown",
	systemPrompt: "",
	mcpAccess: [],
	peers: [],
};

const currentStandupReports = new Map<string, string>();

const TRACE_ENABLED = (process.env.AGENT_TRACE ?? "1") !== "0";
let llmTraceEnabled = (process.env.AGENT_LLM_TRACE ?? "0") === "1";

function nextId(prefix: string): string {
	return `${prefix}-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function send(message: Record<string, unknown>): void {
	if (typeof process.send === "function") {
		process.send(message);
	}
}

function trace(event: string, details?: Record<string, unknown>): void {
	if (!TRACE_ENABLED) {
		return;
	}
	send({ type: "trace", event, details });
}

function llmTrace(event: string, details?: Record<string, unknown>): void {
	if (!llmTraceEnabled) {
		return;
	}
	send({ type: "llm-trace", event, details });
}

function requestMemory(op: string, payload: Record<string, unknown>): Promise<unknown> {
	const requestId = nextId("mem");
	return new Promise((resolve, reject) => {
		pendingMemory.set(requestId, { resolve, reject });
		send({ type: "memory", requestId, op, payload });
		setTimeout(() => {
			if (!pendingMemory.has(requestId)) {
				return;
			}
			pendingMemory.delete(requestId);
			reject(new Error(`Memory timeout for ${requestId}`));
		}, 10000);
	});
}

function requestWorkspace(op: string, payload: Record<string, unknown>): Promise<unknown> {
	const requestId = nextId("work");
	return new Promise((resolve, reject) => {
		pendingWorkspace.set(requestId, { resolve, reject });
		send({ type: "workspace", requestId, op, payload });
		setTimeout(() => {
			if (!pendingWorkspace.has(requestId)) {
				return;
			}
			pendingWorkspace.delete(requestId);
			reject(new Error(`Workspace timeout for ${requestId}`));
		}, 20000);
	});
}

function hasInternetAccess(): boolean {
	return profile.mcpAccess.includes("internet");
}

function normalizeSearchQuery(text: string): string {
	const cleaned = text
		.replace(/^standup-request:\s*/i, "")
		.replace(/^standup-report:\s*/i, "")
		.replace(/^continue-task:\s*/i, "")
		.replace(/^delegate:\s*/i, "")
		.trim();
	if (!cleaned) {
		return "gobierno de Colombia 2026 verificación de noticias falsas y verdaderas";
	}
	return cleaned.slice(0, 240);
}

function formatInternetContext(data: InternetSearchResponse): string {
	if (!data.results.length) {
		return `WEB_SEARCH(query=${data.query}, fetchedAt=${data.fetchedAt}): no results.`;
	}

	const lines = data.results.map((item, index) => {
		const snippet = item.snippet ? ` | ${item.snippet}` : "";
		return `${index + 1}. ${item.title} | ${item.url}${snippet}`;
	});

	return [
		`WEB_SEARCH(query=${data.query}, fetchedAt=${data.fetchedAt}):`,
		...lines,
	].join("\n");
}

async function gatherInternetContext(message: string): Promise<string> {
	if (!hasInternetAccess()) {
		return "";
	}

	const query = normalizeSearchQuery(message);
	llmTrace("internet.search.request", { query });
	const result = (await requestWorkspace("web-search", {
		query,
		limit: 5,
	}).catch((error) => {
		llmTrace("internet.search.error", { query, error: String(error) });
		return null;
	})) as InternetSearchResponse | null;

	if (!result) {
		return "";
	}

	llmTrace("internet.search.response", { query: result.query, count: result.results.length });
	return formatInternetContext(result);
}

function buildReply(from: string, text: string, recalled: MemoryRecord[]): string {
	const memoryNote = recalled.length
		? `Recent memories: ${recalled
				.slice(0, 2)
				.map((item) => item.text)
				.join(" | ")}`
		: "No previous memories found.";

	return [
		`Agent ${profile.name} responding to ${from}.`,
		`System prompt focus: ${profile.systemPrompt.slice(0, 120)}`,
		`MCP access: ${profile.mcpAccess.join(", ") || "none"}.`,
		`Known peers: ${profile.peers.join(", ") || "none"}.`,
		`Message received: ${text}`,
		memoryNote,
	].join("\n");
}

function isPassiveReply(reply: string): boolean {
	const normalized = reply.toLowerCase();
	const passivePatterns = [
		"everything's good",
		"everything is good",
		"ready to help",
		"need anything",
		"let me know",
		"i can help",
	];
	if (reply.trim().length < 180) {
		return true;
	}
	return passivePatterns.some((pattern) => normalized.includes(pattern));
}

function buildProactiveRewrite(from: string, message: string, recalled: MemoryRecord[]): string {
	const memorySummary = recalled.length
		? recalled
				.slice(0, 2)
				.map((item) => item.text)
				.join(" | ")
		: "No relevant memories yet.";

	const delegation = profile.peers.length
		? profile.peers.map((peer) => `- ${peer}: gather one verifiable source and report back now.`).join("\n")
		: "- none";

	return [
		"STATUS: Active. I am processing the current request and moving investigation forward.",
		`FINDINGS: (1) Latest user request: ${message}. (2) Memory context: ${memorySummary}.`,
		"NEXT_ACTIONS:",
		"- Validate at least two independent sources related to the current request.",
		"- Produce a short evidence summary with links, claims, and verification status.",
		"DELEGATIONS:",
		delegation,
		"REQUESTS: none",
	].join("\n");
}

function isLeaderAgent(): boolean {
	return profile.name.toLowerCase().includes("redactor");
}

function enforceAutonomousResponse(reply: string): string {
	const lines = reply.split("\n");
	let hasRequestsLine = false;
	const updated = lines.map((line) => {
		if (/^\s*requests\s*:/i.test(line)) {
			hasRequestsLine = true;
			return "REQUESTS: none";
		}
		return line;
	});
	if (!hasRequestsLine) {
		updated.push("REQUESTS: none");
	}
	return updated.join("\n");
}

function stripStandupPrefix(text: string): string {
	if (!text.startsWith("standup-report:")) {
		return text;
	}
	return text.replace("standup-report:", "").trim();
}

type WorkspaceAction =
	| { kind: "create-file"; relativePath: string; content: string }
	| { kind: "clone-repo"; repoUrl: string; targetFolder?: string };

function sanitizeWorkspaceRelativePath(rawPath: string): string {
	let cleaned = String(rawPath ?? "").trim();
	if (!cleaned) {
		return "artifact.md";
	}

	cleaned = cleaned.replace(/\\/g, "/");

	if (path.isAbsolute(cleaned) || /^[a-zA-Z]:\//.test(cleaned)) {
		const base = path.posix.basename(cleaned);
		return base || "artifact.md";
	}

	let normalized = path.posix.normalize(cleaned);
	while (normalized.startsWith("../")) {
		normalized = normalized.slice(3);
	}
	normalized = normalized.replace(/^\.\//, "");
	if (!normalized || normalized === "." || normalized === "..") {
		return "artifact.md";
	}
	return normalized;
}

function parseWorkspaceActions(reply: string): { cleanedReply: string; actions: WorkspaceAction[] } {
	const actions: WorkspaceAction[] = [];
	let cleaned = reply;

	const createFileRegex = /<<CREATE_FILE\s+path="([^"]+)">>([\s\S]*?)<<\/CREATE_FILE>>/g;
	cleaned = cleaned.replace(createFileRegex, (_match, filePath: string, content: string) => {
		actions.push({
			kind: "create-file",
			relativePath: sanitizeWorkspaceRelativePath(String(filePath).trim()),
			content: String(content).replace(/^\n/, "").replace(/\n$/, ""),
		});
		return "";
	});

	const createFileSingleTagRegex = /<<CREATE_FILE\s+path="([^"]+)"(?:\s+content="([\s\S]*?)")?\s*>>?/g;
	cleaned = cleaned.replace(createFileSingleTagRegex, (_match, filePath: string, content?: string) => {
		actions.push({
			kind: "create-file",
			relativePath: sanitizeWorkspaceRelativePath(String(filePath).trim()),
			content: String(content ?? ""),
		});
		return "";
	});

	const cloneRepoRegex = /<<CLONE_REPO\s+url="([^"]+)"(?:\s+target="([^"]*)")?\s*>><<\/CLONE_REPO>>/g;
	cleaned = cleaned.replace(cloneRepoRegex, (_match, repoUrl: string, targetFolder?: string) => {
		actions.push({
			kind: "clone-repo",
			repoUrl: String(repoUrl).trim(),
			targetFolder: String(targetFolder ?? "").trim() || undefined,
		});
		return "";
	});

	cleaned = cleaned.replace(/\n{3,}/g, "\n\n").trim();
	return { cleanedReply: cleaned, actions };
}

async function executeWorkspaceActions(actions: WorkspaceAction[]): Promise<string[]> {
	const results: string[] = [];
	for (const action of actions) {
		if (action.kind === "create-file") {
			const result = await requestWorkspace("create-file", {
				relativePath: action.relativePath,
				content: action.content,
			}).catch((error) => ({ error: String(error) }));
			const failed = Boolean((result as { error?: string }).error);
			trace("workspace.create-file", { relativePath: action.relativePath, ok: !failed });
			results.push(
				failed
					? `- create-file ${action.relativePath}: failed (${(result as { error: string }).error})`
					: `- create-file ${action.relativePath}: ok`,
			);
			continue;
		}

		const result = await requestWorkspace("clone-repo", {
			repoUrl: action.repoUrl,
			targetFolder: action.targetFolder ?? "",
		}).catch((error) => ({ error: String(error) }));
		const failed = Boolean((result as { error?: string }).error);
		trace("workspace.clone-repo", {
			repoUrl: action.repoUrl,
			targetFolder: action.targetFolder ?? "",
			ok: !failed,
		});
		results.push(
			failed
				? `- clone-repo ${action.repoUrl}: failed (${(result as { error: string }).error})`
				: `- clone-repo ${action.repoUrl}: ok`,
		);
	}
	return results;
}

async function handleChat(msg: WorkerMessage): Promise<void> {
	const from = String(msg.from ?? "unknown");
	const text = String(msg.text ?? "");
	trace("chat.received", { from, preview: text.slice(0, 160) });

	await requestMemory("add", {
		text: `[INCOMING] from=${from} text=${text}`,
		metadata: { direction: "incoming", from },
	}).catch(() => undefined);

	const recalled = (await requestMemory("query", {
		query: text,
		limit: 3,
	}).catch(() => [])) as MemoryRecord[];
	llmTrace("memory.recalled", { count: recalled.length });

	if (text.startsWith("standup-report:")) {
		trace("standup.report_received", { from });
		await requestMemory("add", {
			text: `[STANDUP_REPORT] from=${from} text=${text}`,
			metadata: { direction: "standup_report", from },
		}).catch(() => undefined);

		if (isLeaderAgent()) {
			currentStandupReports.set(from, stripStandupPrefix(text));

			const standupContext = Array.from(currentStandupReports.entries())
				.map(([agent, report]) => `- ${agent}: ${report}`)
				.join("\n");

			let leaderReply = "STATUS: Ongoing standup. FINDINGS: Reports collected. NEXT_ACTIONS: continue investigation and gather stronger evidence. REQUESTS: none";
			try {
				const internetContext = await gatherInternetContext(standupContext);
				const leaderMessage = internetContext
					? `Standup report received from ${from}. Consolidated reports:\n${standupContext}\n\nUse the following web findings and cite direct URLs in FINDINGS:\n${internetContext}`
					: `Standup report received from ${from}. Consolidated reports:\n${standupContext}`;
				leaderReply = await generateAgentReply({
					agentName: profile.name,
					systemPrompt: `${profile.systemPrompt}\nYou are the standup leader. Consolidate team progress and produce artifacts when needed.`,
					mcpAccess: profile.mcpAccess,
					peers: profile.peers,
					from,
					message: leaderMessage,
					memories: [],
				});
			} catch (error) {
				llmTrace("llm.error", { error: String(error) });
			}

			leaderReply = enforceAutonomousResponse(leaderReply);
			const { cleanedReply, actions } = parseWorkspaceActions(leaderReply);
			const actionResults = await executeWorkspaceActions(actions);
			const replyWithActionStatus = actionResults.length
				? `${cleanedReply}\n\nARTIFACT_ACTIONS:\n${actionResults.join("\n")}`
				: cleanedReply;

			send({
				type: "chat",
				from: profile.name,
				to: "SYSTEM",
				text: replyWithActionStatus,
			});

			const nextTask = `continue-task: keep investigating current claims, gather at least 2 fresh sources, and send a new standup-report with updates and contradictions found.`;
			send({
				type: "chat",
				from: profile.name,
				to: from,
				text: nextTask,
			});
			trace("standup.next_task_assigned", { to: from });
		}
		return;
	}

	if (from === "SYSTEM" && text.toLowerCase().includes("standup time")) {
		trace("standup.start", { peers: profile.peers.length });
		if (isLeaderAgent()) {
			currentStandupReports.clear();
		}
		for (const peer of profile.peers) {
			send({
				type: "chat",
				from: profile.name,
				to: peer,
				text: "standup-request: send your latest findings, evidence links, and next actions now.",
			});
		}
	}

	if (text.startsWith("standup-request:")) {
		trace("standup.request_received", { from });
		let standupReply = "STATUS: Active. FINDINGS: No validated findings yet. NEXT_ACTIONS: gather two sources and summarize contradictions. REQUESTS: none.";
		try {
			const internetContext = await gatherInternetContext(text);
			const messageWithContext = internetContext
				? `${text}\n\nUse the following web findings and cite direct URLs in FINDINGS:\n${internetContext}`
				: text;
			standupReply = await generateAgentReply({
				agentName: profile.name,
				systemPrompt: `${profile.systemPrompt}\nYou are responding to a standup request. Be concise and evidence-focused.`,
				mcpAccess: profile.mcpAccess,
				peers: profile.peers,
				from,
				message: messageWithContext,
				memories: recalled.map((item) => item.text),
			});
		} catch (error) {
			trace("standup.request_error", { error: String(error) });
		}

		standupReply = enforceAutonomousResponse(standupReply);

		send({
			type: "chat",
			from: profile.name,
			to: from,
			text: `standup-report: ${standupReply}`,
		});
		trace("standup.report_sent", { to: from, length: standupReply.length });
		return;
	}

	if (text.startsWith("workspace:create-file ")) {
		const raw = text.replace("workspace:create-file ", "").trim();
		const [relativePath, ...contentParts] = raw.split(" ");
		const content = contentParts.join(" ");
		const result = await requestWorkspace("create-file", {
			relativePath,
			content,
		}).catch((error) => ({ error: String(error) }));
		trace("workspace.create-file", { relativePath, ok: !(result as { error?: string }).error });
		send({
			type: "chat",
			from: profile.name,
			to: from,
			text: `Workspace result: ${JSON.stringify(result)}`,
		});
		return;
	}

	if (text.startsWith("workspace:clone-repo ")) {
		const raw = text.replace("workspace:clone-repo ", "").trim();
		const [repoUrl, targetFolder] = raw.split(" ");
		const result = await requestWorkspace("clone-repo", {
			repoUrl,
			targetFolder,
		}).catch((error) => ({ error: String(error) }));
		trace("workspace.clone-repo", { repoUrl, targetFolder, ok: !(result as { error?: string }).error });
		send({
			type: "chat",
			from: profile.name,
			to: from,
			text: `Workspace clone result: ${JSON.stringify(result)}`,
		});
		return;
	}

	let reply = buildReply(from, text, recalled);
	const wantsNaturalReply = from === "USER";
	try {
		const internetContext = await gatherInternetContext(text);
		const messageWithContext = internetContext
			? `${text}\n\nUse the following web findings and cite direct URLs in FINDINGS:\n${internetContext}`
			: text;
		llmTrace("llm.request", { from });
		reply = await generateAgentReply({
			agentName: profile.name,
			systemPrompt: profile.systemPrompt,
			mcpAccess: profile.mcpAccess,
			peers: profile.peers,
			from,
			message: messageWithContext,
			memories: recalled.map((item) => item.text),
			responseStyle: wantsNaturalReply ? "natural" : "structured",
		});
		llmTrace("llm.response", { length: reply.length });
		if (!wantsNaturalReply && isPassiveReply(reply)) {
			reply = buildProactiveRewrite(from, text, recalled);
			llmTrace("llm.rewritten_proactive", { reason: "passive_reply_detected" });
		}
		if (!wantsNaturalReply) {
			reply = enforceAutonomousResponse(reply);
		}
	} catch (error) {
		llmTrace("llm.error", { error: String(error) });
		reply = `${reply}\nLLM fallback reason: ${String(error)}`;
	}

	const { cleanedReply, actions } = parseWorkspaceActions(reply);
	const actionResults = await executeWorkspaceActions(actions);
	reply = actionResults.length
		? `${cleanedReply}\n\nARTIFACT_ACTIONS:\n${actionResults.join("\n")}`
		: cleanedReply;

	await requestMemory("add", {
		text: `[OUTGOING] to=${from} text=${reply}`,
		metadata: { direction: "outgoing", to: from },
	}).catch(() => undefined);

	send({ type: "chat", from: profile.name, to: from, text: reply });
	trace("chat.sent", { to: from, length: reply.length });

	if (text.startsWith("delegate:")) {
		const payload = text.replace("delegate:", "").trim();
		const space = payload.indexOf(" ");
		if (space > 0) {
			const peer = payload.slice(0, space).trim();
			const delegatedText = payload.slice(space + 1).trim();
			send({
				type: "chat",
				from: profile.name,
				to: peer,
				text: `[Delegated by ${from}] ${delegatedText}`,
			});
		}
	}
}

process.on("message", async (msg: WorkerMessage | null) => {
	if (!msg || typeof msg !== "object") {
		return;
	}

	if (msg.type === "init") {
		profile = msg.profile as AgentProfile;
		trace("agent.initialized", {
			name: profile.name,
			mcpAccess: profile.mcpAccess,
			peers: profile.peers,
		});
		send({
			type: "ready",
			name: profile.name,
			summary: {
				mcpAccess: profile.mcpAccess,
				peers: profile.peers,
			},
		});
		return;
	}

	if (msg.type === "chat") {
		await handleChat(msg);
		return;
	}

	if (msg.type === "config") {
		if (typeof msg.llmTraceEnabled === "boolean") {
			llmTraceEnabled = msg.llmTraceEnabled;
			trace("config.updated", { llmTraceEnabled });
		}
		return;
	}

	if (msg.type === "memory-result") {
		const requestId = String(msg.requestId ?? "");
		const pending = pendingMemory.get(requestId);
		if (!pending) {
			return;
		}
		pendingMemory.delete(requestId);
		if (msg.error) {
			pending.reject(new Error(String(msg.error)));
			return;
		}
		pending.resolve(msg.data);
		return;
	}

	if (msg.type === "workspace-result") {
		const requestId = String(msg.requestId ?? "");
		const pending = pendingWorkspace.get(requestId);
		if (!pending) {
			return;
		}
		pendingWorkspace.delete(requestId);
		if (msg.error) {
			pending.reject(new Error(String(msg.error)));
			return;
		}
		pending.resolve(msg.data);
	}
});
