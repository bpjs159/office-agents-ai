import { fork, spawn } from "node:child_process";
import fsSync from "node:fs";
import fs from "node:fs/promises";
import net from "node:net";
import path from "node:path";
import readline from "node:readline";
import pc from "picocolors";
import { checkLlmConnection } from "./llm";

interface MemoryConfig {
	chroma?: {
	};
}

interface AgentConfig {
	name: string;
	systemPrompt: string;
	mcpAccess: MCPServerName[];
	memory?: MemoryConfig;
	workspace?: {
		root?: string;
	};
}

type MCPServerName = "terminal" | "files" | "internet" | "wait";

interface MCPServerConfig {
	enabled: boolean;
	description?: string;
}

interface MemoryItem {
	id: string;
	text: string;
	metadata?: Record<string, unknown>;
	createdAt: string;
}

interface MemoryStore {
	add(agentName: string, text: string, metadata?: Record<string, unknown>): Promise<string>;
	query(agentName: string, query: string, limit: number): Promise<MemoryItem[]>;
}

interface WebSearchResult {
	title: string;
	url: string;
	snippet: string;
}

const CHROMA_URL = process.env.CHROMA_URL ?? "http://localhost:8000";

function createChromaClientOptions(urlValue: string): { host: string; port: number; ssl: boolean } {
	const parsed = new URL(urlValue);
	const ssl = parsed.protocol === "https:";
	const host = parsed.hostname;
	const port = Number(parsed.port || (ssl ? "443" : "80"));
	if (!host || !Number.isFinite(port)) {
		throw new Error(`Invalid CHROMA_URL: ${urlValue}`);
	}
	return { host, port, ssl };
}

const chromaEmbeddingFunction = {
	generate: async (texts: string[]): Promise<number[][]> => texts.map((text) => embedText(text)),
};

class ChromaMemoryStore implements MemoryStore {
	private client: any;

	constructor(private readonly url: string) {
	}

	private async getClient() {
		if (this.client) {
			return this.client;
		}
		const chroma = await import("chromadb");
		this.client = new chroma.ChromaClient(createChromaClientOptions(this.url));
		return this.client;
	}

	private collectionName(agentName: string): string {
		return agentName.replace(/[^a-zA-Z0-9_-]/g, "_");
	}

	async add(agentName: string, text: string, metadata?: Record<string, unknown>): Promise<string> {
		const client = await this.getClient();
		const collection = await client.getOrCreateCollection({
			name: this.collectionName(agentName),
			embeddingFunction: chromaEmbeddingFunction,
		});
		const id = `${Date.now()}-${Math.random().toString(16).slice(2)}`;
		await collection.add({
			ids: [id],
			documents: [text],
			metadatas: [metadata ?? {}],
			embeddings: [embedText(text)],
		});
		return id;
	}

	async query(agentName: string, query: string, limit: number): Promise<MemoryItem[]> {
		const client = await this.getClient();
		const collection = await client.getOrCreateCollection({
			name: this.collectionName(agentName),
			embeddingFunction: chromaEmbeddingFunction,
		});
		const result = await collection.query({
			queryEmbeddings: [embedText(query)],
			nResults: limit,
			include: ["documents", "metadatas"],
		});

		const ids = result.ids?.[0] ?? [];
		const docs = result.documents?.[0] ?? [];
		const metas = result.metadatas?.[0] ?? [];

		return ids.map((id: string, index: number) => ({
			id,
			text: docs[index] ?? "",
			metadata: metas[index] ?? {},
			createdAt: new Date().toISOString(),
		}));
	}
}

function embedText(text: string): number[] {
	const vector = new Array(16).fill(0);
	for (let index = 0; index < text.length; index += 1) {
		const bucket = index % vector.length;
		vector[bucket] += text.charCodeAt(index) / 255;
	}
	const norm = Math.sqrt(vector.reduce((sum, value) => sum + value * value, 0)) || 1;
	return vector.map((value) => value / norm);
}

function formatTimestamp(date = new Date()): string {
	const year = date.getFullYear();
	const month = String(date.getMonth() + 1).padStart(2, "0");
	const day = String(date.getDate()).padStart(2, "0");
	const hours = String(date.getHours()).padStart(2, "0");
	const minutes = String(date.getMinutes()).padStart(2, "0");
	const seconds = String(date.getSeconds()).padStart(2, "0");
	return `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
}

type ColorFn = (text: string) => string;

const AGENT_COLOR_PALETTE: ColorFn[] = [
	pc.cyan,
	pc.magenta,
	pc.blue,
	pc.green,
	pc.yellow,
];

function hashString(input: string): number {
	let hash = 0;
	for (let index = 0; index < input.length; index += 1) {
		hash = (hash << 5) - hash + input.charCodeAt(index);
		hash |= 0;
	}
	return Math.abs(hash);
}

function colorByAgent(agentName: string): ColorFn {
	const index = hashString(agentName) % AGENT_COLOR_PALETTE.length;
	return AGENT_COLOR_PALETTE[index];
}

function formatAgentName(agentName: string): string {
	return colorByAgent(agentName)(agentName);
}

function isPathInside(basePath: string, targetPath: string): boolean {
	const relative = path.relative(path.resolve(basePath), path.resolve(targetPath));
	return relative === "" || (!relative.startsWith("..") && !path.isAbsolute(relative));
}

function logLine(icon: string, label: string, message: string): void {
	const time = pc.dim(`[${formatTimestamp()}]`);
	const scope = pc.bold(pc.white(`${icon} ${label}`));
	console.log(`${time} ${scope} ${message}`);
}

interface AgentRuntime {
	config: AgentConfig;
	process: ReturnType<typeof fork>;
	workspaceRoot: string;
	memory: MemoryStore;
}

interface OfficeConfig {
	standupIntervalMinutes: number;
	standupLeader: string;
	mainTarget: string;
	mcpServers: Record<MCPServerName, MCPServerConfig>;
}

interface PendingMessageWait {
	requestId: string;
	runtime: AgentRuntime;
	createdAt: string;
}

type CliOutcome = "continue" | "disconnect" | "shutdown";

class OfficeOrchestrator {
	private readonly workspaceRoot: string;
	private readonly agentsDir: string;
	private readonly workerPath: string;
	private readonly officeConfigPath: string;
	private readonly cliSocketPath: string;
	private readonly runtimes = new Map<string, AgentRuntime>();
	private readonly recentAgentActivity = new Map<string, string[]>();
	private readonly attachedCliSockets = new Set<net.Socket>();
	private readonly pendingMessageWaits = new Map<string, PendingMessageWait[]>();
	private cliServer?: net.Server;
	private standupTimer?: NodeJS.Timeout;
	private isShuttingDown = false;
	private traceEnabled = (process.env.AGENT_TRACE ?? "1") !== "0";
	private llmTraceEnabled = (process.env.AGENT_LLM_TRACE ?? "0") === "1";
	private officeConfig: OfficeConfig = {
		standupIntervalMinutes: 60,
		standupLeader: "",
		mainTarget: "",
		mcpServers: {
			terminal: { enabled: true, description: "Terminal manipulation MCP" },
			files: { enabled: true, description: "File system MCP" },
			internet: { enabled: true, description: "Internet access MCP" },
			wait: { enabled: true, description: "Wait/scheduling MCP" },
		},
	};

	constructor(workspaceRoot: string) {
		this.workspaceRoot = workspaceRoot;
		this.agentsDir = path.join(workspaceRoot, "agents");
		this.officeConfigPath = path.join(workspaceRoot, "office.config.json");
		this.cliSocketPath = path.join(workspaceRoot, ".office-cli.sock");
		const tsWorker = path.join(this.agentsDir, "agent-worker.ts");
		const jsWorker = path.join(this.agentsDir, "agent-worker.js");
		this.workerPath = fsSync.existsSync(tsWorker) ? tsWorker : jsWorker;
	}

	async boot(): Promise<void> {
		await this.runStartupChecks();
		this.officeConfig = await this.loadOfficeConfig();
		const configs = await this.loadAgentConfigs();
		if (!configs.length) {
			throw new Error("No agent json configs found in ./agents");
		}

		for (const config of configs) {
			const runtime = await this.spawnAgent(config, configs);
			this.runtimes.set(config.name, runtime);
		}

		if (!this.runtimes.has(this.officeConfig.standupLeader)) {
			throw new Error(
				`standupLeader '${this.officeConfig.standupLeader}' is not a valid agent in office.config.json`,
			);
		}

		console.log(`Office started with agents: ${Array.from(this.runtimes.keys()).join(", ")}`);
		logLine(
			"🏢",
			"office",
			`Started with agents: ${Array.from(this.runtimes.keys()).map((name) => formatAgentName(name)).join(", ")}`,
		);
		logLine(
			"🎯",
			"target",
			`${this.officeConfig.mainTarget} | Standup every ${this.officeConfig.standupIntervalMinutes} minute(s) led by ${formatAgentName(this.officeConfig.standupLeader)}`,
		);
		this.triggerStandup();
		this.startStandupLoop();
		await this.startCliServer();
		this.startCli();
	}

	private appendAgentActivity(agentName: string, event: string): void {
		if (!agentName) {
			return;
		}
		const history = this.recentAgentActivity.get(agentName) ?? [];
		history.push(`[${formatTimestamp()}] ${event}`);
		if (history.length > 200) {
			history.splice(0, history.length - 200);
		}
		this.recentAgentActivity.set(agentName, history);
	}

	private registerWaitUntilMessage(agentName: string, runtime: AgentRuntime, requestId: string): void {
		const items = this.pendingMessageWaits.get(agentName) ?? [];
		items.push({
			requestId,
			runtime,
			createdAt: new Date().toISOString(),
		});
		this.pendingMessageWaits.set(agentName, items);
	}

	private resolveWaitUntilMessage(agentName: string, wakeSource: "USER" | "AGENT", from: string): void {
		const waiters = this.pendingMessageWaits.get(agentName) ?? [];
		if (!waiters.length) {
			return;
		}
		this.pendingMessageWaits.delete(agentName);
		for (const waiter of waiters) {
			waiter.runtime.process.send({
				type: "workspace-result",
				requestId: waiter.requestId,
				data: {
					mode: "until-message",
					status: "woken",
					wakeSource,
					from,
					waitStartedAt: waiter.createdAt,
					wokeAt: new Date().toISOString(),
				},
			});
		}
	}

	private getCliHelpText(): string {
		return [
			"Commands:",
			"  help",
			"  agents",
			"  agent <name>",
			"  activity <agent> [limit]",
			"  mcp",
			"  trace <on|off>",
			"  llm-trace <on|off>",
			"  standup-now",
			"  chat <fromAgent> <toAgent> <message>",
			"  ask <agent> <message>      # message from USER to agent",
			"  memory <agent> <query>",
			"  exit",
		].join("\n");
	}

	private async executeCliCommand(
		line: string,
		writer: (text: string) => void,
		remoteSession: boolean,
	): Promise<CliOutcome> {
		const trimmed = line.trim();
		if (!trimmed) {
			return "continue";
		}

		const [command, ...args] = trimmed.split(" ");

		if (command === "help") {
			writer(this.getCliHelpText());
			return "continue";
		}

		if (command === "exit") {
			return remoteSession ? "disconnect" : "shutdown";
		}

		if (command === "agents") {
			for (const runtime of this.runtimes.values()) {
				writer(
					`- ${runtime.config.name} | mcp=${runtime.config.mcpAccess.join(",") || "none"} | workspace=${runtime.workspaceRoot}`,
				);
			}
			return "continue";
		}

		if (command === "agent") {
			const agentName = String(args[0] ?? "").trim();
			const runtime = this.runtimes.get(agentName);
			if (!runtime) {
				writer("Usage: agent <name>");
				return "continue";
			}
			writer(`name=${runtime.config.name}`);
			writer(`workspace=${runtime.workspaceRoot}`);
			writer(`mcp=${runtime.config.mcpAccess.join(",") || "none"}`);
			writer(`peers=${Array.from(this.runtimes.keys()).filter((name) => name !== runtime.config.name).join(",") || "none"}`);
			return "continue";
		}

		if (command === "activity") {
			const [agentName, limitRaw] = args;
			const normalizedAgent = String(agentName ?? "").trim();
			if (!normalizedAgent) {
				writer("Usage: activity <agent> [limit]");
				return "continue";
			}
			const limit = Math.max(1, Math.min(Number(limitRaw ?? 10) || 10, 100));
			const history = this.recentAgentActivity.get(normalizedAgent) ?? [];
			if (!history.length) {
				writer(`No activity for agent '${normalizedAgent}'`);
				return "continue";
			}
			for (const item of history.slice(-limit)) {
				writer(item);
			}
			return "continue";
		}

		if (command === "mcp") {
			writer("MCP servers:");
			for (const serverName of ["terminal", "files", "internet", "wait"] as MCPServerName[]) {
				const config = this.officeConfig.mcpServers[serverName];
				writer(`- ${serverName}: ${config.enabled ? "enabled" : "disabled"}${config.description ? ` (${config.description})` : ""}`);
			}
			return "continue";
		}

		if (command === "trace") {
			const mode = String(args[0] ?? "").toLowerCase();
			if (mode === "on") {
				this.traceEnabled = true;
				writer("Agent traces enabled");
				return "continue";
			}
			if (mode === "off") {
				this.traceEnabled = false;
				writer("Agent traces disabled");
				return "continue";
			}
			writer(`Trace is currently ${this.traceEnabled ? "on" : "off"}. Usage: trace <on|off>`);
			return "continue";
		}

		if (command === "llm-trace") {
			const mode = String(args[0] ?? "").toLowerCase();
			if (mode === "on") {
				this.llmTraceEnabled = true;
				for (const runtime of this.runtimes.values()) {
					runtime.process.send({ type: "config", llmTraceEnabled: true });
				}
				writer("LLM traces enabled");
				return "continue";
			}
			if (mode === "off") {
				this.llmTraceEnabled = false;
				for (const runtime of this.runtimes.values()) {
					runtime.process.send({ type: "config", llmTraceEnabled: false });
				}
				writer("LLM traces disabled");
				return "continue";
			}
			writer(`LLM trace is currently ${this.llmTraceEnabled ? "on" : "off"}. Usage: llm-trace <on|off>`);
			return "continue";
		}

		if (command === "standup-now") {
			this.triggerStandup();
			writer("Standup triggered manually");
			return "continue";
		}

		if (command === "chat") {
			const [from, to, ...messageParts] = args;
			const runtime = this.runtimes.get(from ?? "");
			if (!runtime || !to || messageParts.length === 0) {
				writer("Usage: chat <fromAgent> <toAgent> <message>");
				return "continue";
			}
			runtime.process.send({ type: "chat", from: "SYSTEM", text: `delegate:${to} ${messageParts.join(" ")}` });
			writer(`Delegated from ${from} to ${to}`);
			return "continue";
		}

		if (command === "ask") {
			const [agentName, ...messageParts] = args;
			const runtime = this.runtimes.get(agentName ?? "");
			if (!runtime || messageParts.length === 0) {
				writer("Usage: ask <agent> <message>");
				return "continue";
			}
			runtime.process.send({ type: "chat", from: "USER", text: messageParts.join(" ") });
			this.resolveWaitUntilMessage(String(agentName), "USER", "USER");
			writer(`Question sent to ${agentName}`);
			return "continue";
		}

		if (command === "memory") {
			const [agentName, ...queryParts] = args;
			const runtime = this.runtimes.get(agentName ?? "");
			if (!runtime || queryParts.length === 0) {
				writer("Usage: memory <agent> <query>");
				return "continue";
			}
			const result = await runtime.memory.query(agentName!, queryParts.join(" "), 5);
			writer(JSON.stringify(result, null, 2));
			return "continue";
		}

		writer(this.getCliHelpText());
		return "continue";
	}

	private async startCliServer(): Promise<void> {
		await fs.unlink(this.cliSocketPath).catch((error: NodeJS.ErrnoException) => {
			if (error?.code !== "ENOENT") {
				throw error;
			}
		});

		this.cliServer = net.createServer((socket) => {
			this.attachedCliSockets.add(socket);
			socket.setEncoding("utf8");
			socket.write("Connected to office CLI. Use 'help' for commands.\n__OFFICE_END__\n");
			let buffer = "";
			let chain = Promise.resolve();

			socket.on("close", () => {
				this.attachedCliSockets.delete(socket);
			});

			socket.on("error", () => {
				this.attachedCliSockets.delete(socket);
			});

			socket.on("data", (chunk) => {
				buffer += chunk;
				let newlineIndex = buffer.indexOf("\n");
				while (newlineIndex >= 0) {
					const line = buffer.slice(0, newlineIndex).replace(/\r$/, "");
					buffer = buffer.slice(newlineIndex + 1);
					chain = chain.then(async () => {
						const output: string[] = [];
						const outcome = await this.executeCliCommand(line, (text) => output.push(text), true);
						if (output.length) {
							socket.write(`${output.join("\n")}\n`);
						}
						socket.write("__OFFICE_END__\n");
						if (outcome === "disconnect") {
							socket.end();
						}
					});
					newlineIndex = buffer.indexOf("\n");
				}
			});
		});

		await new Promise<void>((resolve, reject) => {
			this.cliServer?.once("error", reject);
			this.cliServer?.listen(this.cliSocketPath, () => {
				this.cliServer?.off("error", reject);
				resolve();
			});
		});

		logLine("🧩", "cli", `Attach CLI socket ready at ${this.cliSocketPath}`);
	}

	private broadcastToAttachedCli(message: string): void {
		if (!this.attachedCliSockets.size || !message.trim()) {
			return;
		}
		for (const socket of this.attachedCliSockets) {
			if (socket.destroyed) {
				this.attachedCliSockets.delete(socket);
				continue;
			}
			socket.write(`${message}\n__OFFICE_END__\n`);
		}
	}

	private async stopCliServer(): Promise<void> {
		for (const socket of this.attachedCliSockets) {
			socket.end();
		}
		this.attachedCliSockets.clear();

		await new Promise<void>((resolve) => {
			if (!this.cliServer) {
				resolve();
				return;
			}
			this.cliServer.close(() => resolve());
			this.cliServer = undefined;
		});
		await fs.unlink(this.cliSocketPath).catch((error: NodeJS.ErrnoException) => {
			if (error?.code !== "ENOENT") {
				console.error(`Failed to remove CLI socket ${this.cliSocketPath}:`, error);
			}
		});
	}

	private async shutdown(): Promise<void> {
		if (this.isShuttingDown) {
			return;
		}
		this.isShuttingDown = true;
		if (this.standupTimer) {
			clearInterval(this.standupTimer);
		}
		for (const runtime of this.runtimes.values()) {
			runtime.process.kill();
		}
		await this.stopCliServer();
		process.exit(0);
	}

	private async runStartupChecks(): Promise<void> {
		await this.checkChromaConnection();
		await checkLlmConnection();
	}

	private async checkChromaConnection(): Promise<void> {
		const chroma = await import("chromadb");
		const client = new chroma.ChromaClient(createChromaClientOptions(CHROMA_URL));
		await client.heartbeat();
	}

	private async loadOfficeConfig(): Promise<OfficeConfig> {
		const raw = await fs.readFile(this.officeConfigPath, "utf8");
		const parsed = JSON.parse(raw) as Partial<OfficeConfig>;

		const standupIntervalMinutes = Number(parsed.standupIntervalMinutes);
		const standupLeader = String(parsed.standupLeader ?? "").trim();
		const mainTarget = String(parsed.mainTarget ?? "").trim();
		const mcpServers = parsed.mcpServers;

		if (!Number.isFinite(standupIntervalMinutes) || standupIntervalMinutes <= 0) {
			throw new Error("office.config.json requires standupIntervalMinutes > 0");
		}
		if (!standupLeader) {
			throw new Error("office.config.json requires standupLeader");
		}
		if (!mainTarget) {
			throw new Error("office.config.json requires mainTarget");
		}
		if (!mcpServers) {
			throw new Error("office.config.json requires mcpServers");
		}

		const validMcp: Record<MCPServerName, MCPServerConfig> = {
			terminal: { enabled: Boolean(mcpServers.terminal?.enabled), description: mcpServers.terminal?.description },
			files: { enabled: Boolean(mcpServers.files?.enabled), description: mcpServers.files?.description },
			internet: { enabled: Boolean(mcpServers.internet?.enabled), description: mcpServers.internet?.description },
			wait: {
				enabled: mcpServers.wait?.enabled == null ? true : Boolean(mcpServers.wait?.enabled),
				description: mcpServers.wait?.description,
			},
		};

		return {
			standupIntervalMinutes,
			standupLeader,
			mainTarget,
			mcpServers: validMcp,
		};
	}

	private async loadAgentConfigs(): Promise<AgentConfig[]> {
		const entries = await fs.readdir(this.agentsDir, { withFileTypes: true });
		const jsonFiles = entries
			.filter((entry) => entry.isFile() && entry.name.endsWith(".json"))
			.map((entry) => path.join(this.agentsDir, entry.name));

		const configs: AgentConfig[] = [];
		const allowedServers: MCPServerName[] = ["terminal", "files", "internet", "wait"];
		for (const filePath of jsonFiles) {
			const raw = await fs.readFile(filePath, "utf8");
			const parsed = JSON.parse(raw) as AgentConfig;
			if (!parsed.name || !parsed.systemPrompt) {
				throw new Error(`Invalid agent config in ${filePath}`);
			}
			parsed.mcpAccess = (parsed.mcpAccess ?? []).map((item) => String(item).trim().toLowerCase() as MCPServerName);
			for (const serverName of parsed.mcpAccess) {
				if (!allowedServers.includes(serverName)) {
					throw new Error(
						`Invalid MCP server '${serverName}' in ${filePath}. Allowed: terminal, files, internet`,
					);
				}
				if (!this.officeConfig.mcpServers[serverName].enabled) {
					throw new Error(
						`Agent '${parsed.name}' uses disabled MCP server '${serverName}' in office.config.json`,
					);
				}
			}
			configs.push(parsed);
		}
		return configs;
	}

	private hasMcpAccess(agentName: string, server: MCPServerName): boolean {
		const runtime = this.runtimes.get(agentName);
		if (!runtime) {
			return false;
		}
		if (!this.officeConfig.mcpServers[server].enabled) {
			return false;
		}
		return runtime.config.mcpAccess.includes(server);
	}

	private async createMemoryStore(config: AgentConfig): Promise<MemoryStore> {
		return new ChromaMemoryStore(CHROMA_URL);
	}

	private async ensureWorkspace(config: AgentConfig): Promise<string> {
		const configured = config.workspace?.root;
		const root = configured
			? path.resolve(this.workspaceRoot, configured)
			: path.join(this.workspaceRoot, "agent-workspaces", config.name);
		if (!isPathInside(this.workspaceRoot, root)) {
			throw new Error(
				`Invalid workspace.root for agent '${config.name}'. It must be inside the office workspace.`,
			);
		}
		await fs.mkdir(root, { recursive: true });
		return root;
	}

	private async spawnAgent(config: AgentConfig, allConfigs: AgentConfig[]): Promise<AgentRuntime> {
		const memory = await this.createMemoryStore(config);
		const workspaceRoot = await this.ensureWorkspace(config);
		const child = fork(this.workerPath, [], {
			stdio: ["pipe", "pipe", "pipe", "ipc"],
			env: {
				...process.env,
				AGENT_NAME: config.name,
				AGENT_TRACE: this.traceEnabled ? "1" : "0",
				AGENT_LLM_TRACE: this.llmTraceEnabled ? "1" : "0",
			},
		});

		child.stdout?.on("data", (chunk) => {
			process.stdout.write(`[${config.name}:stdout] ${chunk}`);
		});

		child.stderr?.on("data", (chunk) => {
			process.stderr.write(`[${config.name}:stderr] ${chunk}`);
		});

		child.on("message", async (message) => {
			await this.handleWorkerMessage(config.name, message as Record<string, unknown>);
		});

		child.on("exit", (code, signal) => {
			console.log(`Agent ${config.name} exited (code=${code} signal=${signal ?? "none"})`);
			this.pendingMessageWaits.delete(config.name);
			this.runtimes.delete(config.name);
		});

		child.send({
			type: "init",
			profile: {
				name: config.name,
				systemPrompt: config.systemPrompt,
				mcpAccess: config.mcpAccess,
				peers: allConfigs.filter((item) => item.name !== config.name).map((item) => item.name),
			},
		});

		return {
			config,
			process: child,
			workspaceRoot,
			memory,
		};
	}

	private async handleWorkerMessage(
		fromAgent: string,
		message: Record<string, unknown>,
	): Promise<void> {
		const runtime = this.runtimes.get(fromAgent);
		if (!runtime || !message.type) {
			return;
		}

		if (message.type === "ready") {
			logLine("✅", "ready", `${formatAgentName(fromAgent)} is online`);
			this.appendAgentActivity(fromAgent, "ready: online");
			return;
		}

		if (message.type === "trace") {
			if (!this.traceEnabled) {
				return;
			}
			const event = String(message.event ?? "event");
			const details = message.details ? ` ${JSON.stringify(message.details)}` : "";
			logLine("🔎", `trace:${formatAgentName(fromAgent)}`, `${pc.cyan(event)}${pc.dim(details)}`);
			this.appendAgentActivity(fromAgent, `trace:${event}${details}`);
			return;
		}

		if (message.type === "llm-trace") {
			if (!this.llmTraceEnabled) {
				return;
			}
			const event = String(message.event ?? "event");
			const details = message.details ? ` ${JSON.stringify(message.details)}` : "";
			logLine("🧠", `llm:${formatAgentName(fromAgent)}`, `${pc.magenta(event)}${pc.dim(details)}`);
			this.appendAgentActivity(fromAgent, `llm:${event}${details}`);
			return;
		}

		if (message.type === "chat") {
			const to = String(message.to ?? "");
			const text = String(message.text ?? "");
			const preview = text.split("\n")[0];
			this.appendAgentActivity(fromAgent, `chat->${to || "unknown"}: ${preview}`);
			if (to === "USER") {
				logLine("🤖", `agent:${formatAgentName(fromAgent)}`, text);
				this.broadcastToAttachedCli(`[${formatTimestamp()}] agent:${fromAgent} ${text}`);
				return;
			}
			if (to === "SYSTEM") {
				logLine("🛰️", `system:${formatAgentName(fromAgent)}`, text);
				return;
			}

			const toRuntime = this.runtimes.get(to);
			if (!toRuntime) {
				console.warn(`Chat target not found: ${to}`);
				return;
			}
			toRuntime.process.send({ type: "chat", from: fromAgent, text });
			this.resolveWaitUntilMessage(to, "AGENT", fromAgent);
			this.appendAgentActivity(to, `chat<-${fromAgent}: ${preview}`);
			logLine("💬", "chat", `${formatAgentName(fromAgent)} → ${formatAgentName(to)}: ${preview}`);
			return;
		}

		if (message.type === "memory") {
			const requestId = String(message.requestId ?? "");
			const op = String(message.op ?? "");
			const payload = (message.payload ?? {}) as Record<string, unknown>;
			try {
				if (op === "add") {
					const id = await runtime.memory.add(
						fromAgent,
						String(payload.text ?? ""),
						(payload.metadata as Record<string, unknown>) ?? {},
					);
					runtime.process.send({ type: "memory-result", requestId, data: { id } });
					return;
				}
				if (op === "query") {
					const items = await runtime.memory.query(
						fromAgent,
						String(payload.query ?? ""),
						Number(payload.limit ?? 3),
					);
					runtime.process.send({ type: "memory-result", requestId, data: items });
					return;
				}
				runtime.process.send({ type: "memory-result", requestId, error: `Unknown memory op ${op}` });
			} catch (error) {
				runtime.process.send({
					type: "memory-result",
					requestId,
					error: `Memory operation failed: ${String(error)}`,
				});
			}
			return;
		}

		if (message.type === "workspace") {
			const requestId = String(message.requestId ?? "");
			const op = String(message.op ?? "");
			const payload = (message.payload ?? {}) as Record<string, unknown>;
			try {
				if (op === "wait") {
					if (!this.hasMcpAccess(fromAgent, "wait")) {
						runtime.process.send({
							type: "workspace-result",
							requestId,
							error: "MCP access denied: wait",
						});
						return;
					}

					const mode = String(payload.mode ?? "").trim().toLowerCase();
					const untilMessage = payload.untilMessage === true || mode === "until-message" || mode === "message";

					if (untilMessage) {
						this.registerWaitUntilMessage(fromAgent, runtime, requestId);
						return;
					}

					const seconds = Number(payload.seconds ?? payload.durationSeconds ?? 0);
					if (!Number.isFinite(seconds) || seconds <= 0) {
						runtime.process.send({
							type: "workspace-result",
							requestId,
							error: "wait requires seconds > 0 or mode=until-message",
						});
						return;
					}

					const waitMs = Math.round(seconds * 1000);
					await new Promise<void>((resolve) => {
						setTimeout(resolve, waitMs);
					});

					runtime.process.send({
						type: "workspace-result",
						requestId,
						data: {
							mode: "seconds",
							status: "elapsed",
							waitedSeconds: seconds,
							waitedMs: waitMs,
						},
					});
					return;
				}

				if (op === "web-search") {
					if (!this.hasMcpAccess(fromAgent, "internet")) {
						runtime.process.send({
							type: "workspace-result",
							requestId,
							error: "MCP access denied: internet",
						});
						return;
					}
					const query = String(payload.query ?? "").trim();
					const limit = Number(payload.limit ?? 5);
					const result = await this.searchInternet(query, limit);
					runtime.process.send({ type: "workspace-result", requestId, data: result });
					return;
				}

				if (op === "create-file") {
					if (!this.hasMcpAccess(fromAgent, "files")) {
						runtime.process.send({
							type: "workspace-result",
							requestId,
							error: "MCP access denied: files",
						});
						return;
					}
					const result = await this.createAgentFile(
						runtime.workspaceRoot,
						String(payload.relativePath ?? ""),
						String(payload.content ?? ""),
					);
					runtime.process.send({ type: "workspace-result", requestId, data: result });
					return;
				}
				if (op === "clone-repo") {
					if (!this.hasMcpAccess(fromAgent, "terminal") || !this.hasMcpAccess(fromAgent, "files")) {
						runtime.process.send({
							type: "workspace-result",
							requestId,
							error: "MCP access denied: terminal and files are required for clone-repo",
						});
						return;
					}
					const result = await this.cloneRepoToWorkspace(
						runtime.workspaceRoot,
						String(payload.repoUrl ?? ""),
						String(payload.targetFolder ?? ""),
					);
					runtime.process.send({ type: "workspace-result", requestId, data: result });
					return;
				}
				if (op === "git-commit") {
					if (!this.hasMcpAccess(fromAgent, "terminal") || !this.hasMcpAccess(fromAgent, "files")) {
						runtime.process.send({
							type: "workspace-result",
							requestId,
							error: "MCP access denied: terminal and files are required for git-commit",
						});
						return;
					}
					const result = await this.gitCommitInWorkspace(
						runtime.workspaceRoot,
						String(payload.repoPath ?? "."),
						String(payload.message ?? ""),
						payload.addAll !== false,
					);
					runtime.process.send({ type: "workspace-result", requestId, data: result });
					return;
				}
				if (op === "git-push") {
					if (!this.hasMcpAccess(fromAgent, "terminal") || !this.hasMcpAccess(fromAgent, "files")) {
						runtime.process.send({
							type: "workspace-result",
							requestId,
							error: "MCP access denied: terminal and files are required for git-push",
						});
						return;
					}
					const result = await this.gitPushInWorkspace(
						runtime.workspaceRoot,
						String(payload.repoPath ?? "."),
						String(payload.remote ?? "origin"),
						String(payload.branch ?? ""),
					);
					runtime.process.send({ type: "workspace-result", requestId, data: result });
					return;
				}
				if (op === "pr-create") {
					if (!this.hasMcpAccess(fromAgent, "terminal") || !this.hasMcpAccess(fromAgent, "files")) {
						runtime.process.send({
							type: "workspace-result",
							requestId,
							error: "MCP access denied: terminal and files are required for pr-create",
						});
						return;
					}
					const result = await this.createPullRequestInWorkspace(
						runtime.workspaceRoot,
						String(payload.repoPath ?? "."),
						String(payload.title ?? ""),
						String(payload.body ?? ""),
						String(payload.base ?? ""),
						String(payload.head ?? ""),
						payload.draft === true,
					);
					runtime.process.send({ type: "workspace-result", requestId, data: result });
					return;
				}
				if (op === "pr-approve") {
					if (!this.hasMcpAccess(fromAgent, "terminal") || !this.hasMcpAccess(fromAgent, "files")) {
						runtime.process.send({
							type: "workspace-result",
							requestId,
							error: "MCP access denied: terminal and files are required for pr-approve",
						});
						return;
					}
					const result = await this.approvePullRequestInWorkspace(
						runtime.workspaceRoot,
						String(payload.repoPath ?? "."),
						String(payload.prNumber ?? ""),
						String(payload.prUrl ?? ""),
						String(payload.body ?? ""),
					);
					runtime.process.send({ type: "workspace-result", requestId, data: result });
					return;
				}
				runtime.process.send({ type: "workspace-result", requestId, error: `Unknown workspace op ${op}` });
			} catch (error) {
				runtime.process.send({
					type: "workspace-result",
					requestId,
					error: `Workspace operation failed: ${String(error)}`,
				});
			}
		}
	}

	private decodeHtmlEntities(text: string): string {
		return text
			.replace(/&amp;/g, "&")
			.replace(/&quot;/g, '"')
			.replace(/&#39;|&apos;/g, "'")
			.replace(/&lt;/g, "<")
			.replace(/&gt;/g, ">")
			.replace(/&nbsp;/g, " ")
			.replace(/&#(\d+);/g, (_match, code) => String.fromCharCode(Number(code)));
	}

	private stripHtml(text: string): string {
		return this.decodeHtmlEntities(text.replace(/<[^>]+>/g, " ")).replace(/\s+/g, " ").trim();
	}

	private unwrapDuckDuckGoUrl(rawUrl: string): string {
		try {
			const parsed = new URL(rawUrl, "https://duckduckgo.com");
			if (parsed.hostname.includes("duckduckgo.com") && parsed.pathname === "/l/") {
				const target = parsed.searchParams.get("uddg");
				if (target) {
					return decodeURIComponent(target);
				}
			}
			return parsed.toString();
		} catch {
			return rawUrl;
		}
	}

	private async searchDuckDuckGo(query: string, limit: number): Promise<WebSearchResult[]> {
		const url = `https://duckduckgo.com/html/?q=${encodeURIComponent(query)}`;
		const response = await fetch(url, {
			headers: {
				"user-agent": "Mozilla/5.0 (compatible; office-agents-ai/1.0)",
			},
		});
		if (!response.ok) {
			throw new Error(`DuckDuckGo request failed (${response.status})`);
		}

		const html = await response.text();
		const resultRegex = /<a[^>]*class="result__a"[^>]*href="([^"]+)"[^>]*>([\s\S]*?)<\/a>/g;
		const snippetRegex = /<a[^>]*class="result__snippet"[^>]*>([\s\S]*?)<\/a>|<td[^>]*class="result-snippet"[^>]*>([\s\S]*?)<\/td>/g;

		const snippets: string[] = [];
		let snippetMatch: RegExpExecArray | null = snippetRegex.exec(html);
		while (snippetMatch) {
			snippets.push(this.stripHtml(snippetMatch[1] ?? snippetMatch[2] ?? ""));
			snippetMatch = snippetRegex.exec(html);
		}

		const results: WebSearchResult[] = [];
		let linkMatch: RegExpExecArray | null = resultRegex.exec(html);
		while (linkMatch && results.length < limit) {
			const rawLink = String(linkMatch[1] ?? "").trim();
			const title = this.stripHtml(String(linkMatch[2] ?? "").trim());
			if (!rawLink || !title) {
				linkMatch = resultRegex.exec(html);
				continue;
			}

			const urlValue = this.unwrapDuckDuckGoUrl(rawLink);
			if (!/^https?:\/\//i.test(urlValue)) {
				linkMatch = resultRegex.exec(html);
				continue;
			}

			results.push({
				title,
				url: urlValue,
				snippet: snippets[results.length] ?? "",
			});
			linkMatch = resultRegex.exec(html);
		}

		return results;
	}

	private async searchWikipedia(query: string, limit: number): Promise<WebSearchResult[]> {
		const url = `https://es.wikipedia.org/w/api.php?action=query&list=search&srsearch=${encodeURIComponent(
			query,
		)}&utf8=1&format=json&srlimit=${Math.max(1, Math.min(limit, 10))}`;
		const response = await fetch(url);
		if (!response.ok) {
			throw new Error(`Wikipedia request failed (${response.status})`);
		}
		const data = (await response.json()) as {
			query?: { search?: Array<{ title: string; snippet: string; pageid: number }> };
		};

		const items = data.query?.search ?? [];
		return items.slice(0, limit).map((item) => ({
			title: item.title,
			url: `https://es.wikipedia.org/?curid=${item.pageid}`,
			snippet: this.stripHtml(item.snippet),
		}));
	}

	private async searchInternet(
		query: string,
		limit: number,
	): Promise<{ query: string; fetchedAt: string; results: WebSearchResult[] }> {
		if (!query.trim()) {
			throw new Error("query is required for web-search");
		}
		const safeLimit = Math.max(1, Math.min(Number.isFinite(limit) ? limit : 5, 10));

		let results: WebSearchResult[] = [];
		try {
			results = await this.searchDuckDuckGo(query, safeLimit);
		} catch {
			results = [];
		}

		if (!results.length) {
			results = await this.searchWikipedia(query, safeLimit).catch(() => []);
		}

		return {
			query,
			fetchedAt: new Date().toISOString(),
			results,
		};
	}

	private async createAgentFile(
		workspaceRoot: string,
		relativePath: string,
		content: string,
	): Promise<{ path: string }> {
		if (!relativePath) {
			throw new Error("relativePath is required");
		}
		const target = path.isAbsolute(relativePath)
			? path.resolve(relativePath)
			: path.resolve(workspaceRoot, relativePath);
		if (!isPathInside(workspaceRoot, target)) {
			throw new Error("Invalid path outside agent workspace");
		}
		await fs.mkdir(path.dirname(target), { recursive: true });
		await fs.writeFile(target, content, "utf8");
		return { path: target };
	}

	private async cloneRepoToWorkspace(
		workspaceRoot: string,
		repoUrl: string,
		targetFolder = "",
	): Promise<{ path: string; repoUrl: string }> {
		if (!repoUrl) {
			throw new Error("repoUrl is required");
		}
		const destination = targetFolder
			? path.resolve(workspaceRoot, targetFolder)
			: path.resolve(workspaceRoot, path.basename(repoUrl, ".git"));

		if (!isPathInside(workspaceRoot, destination)) {
			throw new Error("Invalid destination outside agent workspace");
		}

		await fs.mkdir(path.dirname(destination), { recursive: true });

		await new Promise<void>((resolve, reject) => {
			const child = spawn("git", ["clone", repoUrl, destination], { stdio: "inherit" });
			child.on("exit", (code) => {
				if (code === 0) {
					resolve();
					return;
				}
				reject(new Error(`git clone failed with code ${code}`));
			});
			child.on("error", reject);
		});

		return { path: destination, repoUrl };
	}

	private resolveRepoPath(workspaceRoot: string, repoPath: string): string {
		const resolved = repoPath
			? path.resolve(workspaceRoot, repoPath)
			: workspaceRoot;
		if (!isPathInside(workspaceRoot, resolved)) {
			throw new Error("Invalid repoPath outside agent workspace");
		}
		return resolved;
	}

	private async runCommand(
		command: string,
		args: string[],
		cwd: string,
	): Promise<{ stdout: string; stderr: string }> {
		return await new Promise<{ stdout: string; stderr: string }>((resolve, reject) => {
			const child = spawn(command, args, { cwd, stdio: ["ignore", "pipe", "pipe"] });
			let stdout = "";
			let stderr = "";

			child.stdout?.on("data", (chunk) => {
				stdout += String(chunk);
			});
			child.stderr?.on("data", (chunk) => {
				stderr += String(chunk);
			});

			child.on("exit", (code) => {
				if (code === 0) {
					resolve({ stdout: stdout.trim(), stderr: stderr.trim() });
					return;
				}
				reject(new Error(`${command} ${args.join(" ")} failed with code ${code}. ${stderr || stdout}`));
			});
			child.on("error", reject);
		});
	}

	private async gitCommitInWorkspace(
		workspaceRoot: string,
		repoPath: string,
		message: string,
		addAll: boolean,
	): Promise<{ repoPath: string; message: string; committed: boolean; output?: string }> {
		if (!message.trim()) {
			throw new Error("git-commit requires message");
		}
		const repoRoot = this.resolveRepoPath(workspaceRoot, repoPath);
		if (addAll) {
			await this.runCommand("git", ["add", "-A"], repoRoot);
		}
		const result = await this.runCommand("git", ["commit", "-m", message], repoRoot);
		return {
			repoPath: repoRoot,
			message,
			committed: true,
			output: result.stdout || result.stderr || undefined,
		};
	}

	private async gitPushInWorkspace(
		workspaceRoot: string,
		repoPath: string,
		remote: string,
		branch: string,
	): Promise<{ repoPath: string; remote: string; branch?: string; pushed: boolean; output?: string }> {
		const repoRoot = this.resolveRepoPath(workspaceRoot, repoPath);
		const args = branch.trim() ? ["push", remote || "origin", branch] : ["push", remote || "origin"];
		const result = await this.runCommand("git", args, repoRoot);
		return {
			repoPath: repoRoot,
			remote: remote || "origin",
			branch: branch || undefined,
			pushed: true,
			output: result.stdout || result.stderr || undefined,
		};
	}

	private async createPullRequestInWorkspace(
		workspaceRoot: string,
		repoPath: string,
		title: string,
		body: string,
		base: string,
		head: string,
		draft: boolean,
	): Promise<{ repoPath: string; created: boolean; prUrl?: string; output?: string }> {
		if (!title.trim()) {
			throw new Error("pr-create requires title");
		}
		const repoRoot = this.resolveRepoPath(workspaceRoot, repoPath);
		const args = ["pr", "create", "--title", title, "--body", body || ""]; 
		if (base.trim()) {
			args.push("--base", base);
		}
		if (head.trim()) {
			args.push("--head", head);
		}
		if (draft) {
			args.push("--draft");
		}
		const result = await this.runCommand("gh", args, repoRoot);
		return {
			repoPath: repoRoot,
			created: true,
			prUrl: result.stdout.split("\n").find((line) => /^https?:\/\//i.test(line.trim())),
			output: result.stdout || result.stderr || undefined,
		};
	}

	private async approvePullRequestInWorkspace(
		workspaceRoot: string,
		repoPath: string,
		prNumber: string,
		prUrl: string,
		body: string,
	): Promise<{ repoPath: string; approved: boolean; target: string; output?: string }> {
		const repoRoot = this.resolveRepoPath(workspaceRoot, repoPath);
		const target = prUrl.trim() || prNumber.trim();
		if (!target) {
			throw new Error("pr-approve requires prNumber or prUrl");
		}
		const args = ["pr", "review", target, "--approve"];
		if (body.trim()) {
			args.push("--body", body);
		}
		const result = await this.runCommand("gh", args, repoRoot);
		return {
			repoPath: repoRoot,
			approved: true,
			target,
			output: result.stdout || result.stderr || undefined,
		};
	}

	private startStandupLoop(): void {
		const intervalMs = this.officeConfig.standupIntervalMinutes * 60 * 1000;
		this.standupTimer = setInterval(() => {
			this.triggerStandup();
		}, intervalMs);
	}

	private triggerStandup(): void {
		const leaderRuntime = this.runtimes.get(this.officeConfig.standupLeader);
		if (!leaderRuntime) {
			logLine("⚠️", "standup", `Leader unavailable: ${formatAgentName(this.officeConfig.standupLeader)}`);
			return;
		}

		const participantList = Array.from(this.runtimes.keys()).join(", ");
		leaderRuntime.process.send({
			type: "chat",
			from: "SYSTEM",
			text: `Standup time. Lead this standup for agents: ${participantList}. Main target: ${this.officeConfig.mainTarget}. Collect updates and send delegations as needed.`,
		});
		logLine("📣", "standup", `Triggered for leader ${formatAgentName(this.officeConfig.standupLeader)}`);
	}

	private startCli(): void {
		const cli = readline.createInterface({
			input: process.stdin,
			output: process.stdout,
			prompt: "office> ",
		});

		console.log(this.getCliHelpText());
		cli.prompt();

		cli.on("line", async (line) => {
			const outcome = await this.executeCliCommand(line, (text) => console.log(text), false);
			if (outcome === "shutdown") {
				cli.close();
				return;
			}
			cli.prompt();
		});

		cli.on("close", async () => {
			await this.shutdown();
		});
	}
}

async function runAttachCliClient(workspaceRoot: string): Promise<void> {
	const socketPath = path.join(workspaceRoot, ".office-cli.sock");
	const socket = net.createConnection(socketPath);
	const cli = readline.createInterface({
		input: process.stdin,
		output: process.stdout,
		prompt: "office(remote)> ",
	});

	let buffer = "";

	socket.setEncoding("utf8");
	socket.on("connect", () => {
		cli.prompt();
	});

	socket.on("data", (chunk) => {
		buffer += chunk;
		let newlineIndex = buffer.indexOf("\n");
		while (newlineIndex >= 0) {
			const line = buffer.slice(0, newlineIndex).replace(/\r$/, "");
			buffer = buffer.slice(newlineIndex + 1);
			if (line === "__OFFICE_END__") {
				cli.prompt();
			} else if (line.trim()) {
				console.log(line);
			}
			newlineIndex = buffer.indexOf("\n");
		}
	});

	socket.on("error", (error: NodeJS.ErrnoException) => {
		if (error.code === "ENOENT") {
			console.error(`Office CLI socket not found at ${socketPath}. Start office first.`);
		} else {
			console.error("Failed to connect to office CLI socket:", error);
		}
		process.exit(1);
	});

	socket.on("close", () => {
		console.log("Detached from office CLI");
		process.exit(0);
	});

	cli.on("line", (line) => {
		socket.write(`${line}\n`);
	});

	cli.on("close", () => {
		socket.end();
	});
}

async function main() {
	const workspaceRoot = __dirname;
	if (process.argv.includes("--attach-cli")) {
		await runAttachCliClient(workspaceRoot);
		return;
	}
	const office = new OfficeOrchestrator(workspaceRoot);
	await office.boot();
}

main().catch((error) => {
	console.error("Failed to start office orchestrator:", error);
	process.exit(1);
});

