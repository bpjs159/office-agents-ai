import { generateAgentReply } from "../llm";

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
			standupReply = await generateAgentReply({
				agentName: profile.name,
				systemPrompt: `${profile.systemPrompt}\nYou are responding to a standup request. Be concise and evidence-focused.`,
				mcpAccess: profile.mcpAccess,
				peers: profile.peers,
				from,
				message: text,
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
	try {
		llmTrace("llm.request", { from });
		reply = await generateAgentReply({
			agentName: profile.name,
			systemPrompt: profile.systemPrompt,
			mcpAccess: profile.mcpAccess,
			peers: profile.peers,
			from,
			message: text,
			memories: recalled.map((item) => item.text),
		});
		llmTrace("llm.response", { length: reply.length });
		if (isPassiveReply(reply)) {
			reply = buildProactiveRewrite(from, text, recalled);
			llmTrace("llm.rewritten_proactive", { reason: "passive_reply_detected" });
		}
		reply = enforceAutonomousResponse(reply);
	} catch (error) {
		llmTrace("llm.error", { error: String(error) });
		reply = `${reply}\nLLM fallback reason: ${String(error)}`;
	}

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
