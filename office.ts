import { fork, spawn } from "node:child_process";
import fsSync from "node:fs";
import fs from "node:fs/promises";
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

type MCPServerName = "terminal" | "files" | "internet";

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

class OfficeOrchestrator {
	private readonly workspaceRoot: string;
	private readonly agentsDir: string;
	private readonly workerPath: string;
	private readonly officeConfigPath: string;
	private readonly runtimes = new Map<string, AgentRuntime>();
	private standupTimer?: NodeJS.Timeout;
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
		},
	};

	constructor(workspaceRoot: string) {
		this.workspaceRoot = workspaceRoot;
		this.agentsDir = path.join(workspaceRoot, "agents");
		this.officeConfigPath = path.join(workspaceRoot, "office.config.json");
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
		this.startCli();
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
		const allowedServers: MCPServerName[] = ["terminal", "files", "internet"];
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
			return;
		}

		if (message.type === "trace") {
			if (!this.traceEnabled) {
				return;
			}
			const event = String(message.event ?? "event");
			const details = message.details ? ` ${JSON.stringify(message.details)}` : "";
			logLine("🔎", `trace:${formatAgentName(fromAgent)}`, `${pc.cyan(event)}${pc.dim(details)}`);
			return;
		}

		if (message.type === "llm-trace") {
			if (!this.llmTraceEnabled) {
				return;
			}
			const event = String(message.event ?? "event");
			const details = message.details ? ` ${JSON.stringify(message.details)}` : "";
			logLine("🧠", `llm:${formatAgentName(fromAgent)}`, `${pc.magenta(event)}${pc.dim(details)}`);
			return;
		}

		if (message.type === "chat") {
			const to = String(message.to ?? "");
			const text = String(message.text ?? "");
			if (to === "USER") {
				logLine("🤖", `agent:${formatAgentName(fromAgent)}`, text);
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
			logLine("💬", "chat", `${formatAgentName(fromAgent)} → ${formatAgentName(to)}: ${text.split("\n")[0]}`);
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
		const target = path.resolve(workspaceRoot, relativePath);
		if (!target.startsWith(path.resolve(workspaceRoot))) {
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

		if (!destination.startsWith(path.resolve(workspaceRoot))) {
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

		const help = [
			"Commands:",
			"  agents",
			"  mcp",
			"  trace <on|off>",
			"  llm-trace <on|off>",
			"  standup-now",
			"  chat <fromAgent> <toAgent> <message>",
			"  ask <agent> <message>      # message from USER to agent",
			"  memory <agent> <query>",
			"  exit",
		].join("\n");

		console.log(help);
		cli.prompt();

		cli.on("line", async (line) => {
			const trimmed = line.trim();
			if (!trimmed) {
				cli.prompt();
				return;
			}

			const [command, ...args] = trimmed.split(" ");
			if (command === "exit") {
				cli.close();
				return;
			}

			if (command === "agents") {
				for (const runtime of this.runtimes.values()) {
					console.log(
						`- ${runtime.config.name} | mcp=${runtime.config.mcpAccess.join(",") || "none"
						} | workspace=${runtime.workspaceRoot}`,
					);
				}
				cli.prompt();
				return;
			}

			if (command === "mcp") {
				console.log("MCP servers:");
				for (const serverName of ["terminal", "files", "internet"] as MCPServerName[]) {
					const config = this.officeConfig.mcpServers[serverName];
					console.log(
						`- ${serverName}: ${config.enabled ? "enabled" : "disabled"}${config.description ? ` (${config.description})` : ""}`,
					);
				}
				cli.prompt();
				return;
			}

			if (command === "trace") {
				const mode = String(args[0] ?? "").toLowerCase();
				if (mode === "on") {
					this.traceEnabled = true;
					console.log("Agent traces enabled");
					cli.prompt();
					return;
				}
				if (mode === "off") {
					this.traceEnabled = false;
					console.log("Agent traces disabled");
					cli.prompt();
					return;
				}
				console.log(`Trace is currently ${this.traceEnabled ? "on" : "off"}. Usage: trace <on|off>`);
				cli.prompt();
				return;
			}

			if (command === "llm-trace") {
				const mode = String(args[0] ?? "").toLowerCase();
				if (mode === "on") {
					this.llmTraceEnabled = true;
					for (const runtime of this.runtimes.values()) {
						runtime.process.send({ type: "config", llmTraceEnabled: true });
					}
					console.log("LLM traces enabled");
					cli.prompt();
					return;
				}
				if (mode === "off") {
					this.llmTraceEnabled = false;
					for (const runtime of this.runtimes.values()) {
						runtime.process.send({ type: "config", llmTraceEnabled: false });
					}
					console.log("LLM traces disabled");
					cli.prompt();
					return;
				}
				console.log(`LLM trace is currently ${this.llmTraceEnabled ? "on" : "off"}. Usage: llm-trace <on|off>`);
				cli.prompt();
				return;
			}

			if (command === "standup-now") {
				this.triggerStandup();
				console.log("Standup triggered manually");
				cli.prompt();
				return;
			}

			if (command === "chat") {
				const [from, to, ...messageParts] = args;
				const runtime = this.runtimes.get(from ?? "");
				if (!runtime || !to || messageParts.length === 0) {
					console.log("Usage: chat <fromAgent> <toAgent> <message>");
					cli.prompt();
					return;
				}
				runtime.process.send({ type: "chat", from: "SYSTEM", text: `delegate:${to} ${messageParts.join(" ")}` });
				cli.prompt();
				return;
			}

			if (command === "ask") {
				const [agentName, ...messageParts] = args;
				const runtime = this.runtimes.get(agentName ?? "");
				if (!runtime || messageParts.length === 0) {
					console.log("Usage: ask <agent> <message>");
					cli.prompt();
					return;
				}
				runtime.process.send({ type: "chat", from: "USER", text: messageParts.join(" ") });
				cli.prompt();
				return;
			}

			if (command === "memory") {
				const [agentName, ...queryParts] = args;
				const runtime = this.runtimes.get(agentName ?? "");
				if (!runtime || queryParts.length === 0) {
					console.log("Usage: memory <agent> <query>");
					cli.prompt();
					return;
				}
				const result = await runtime.memory.query(agentName!, queryParts.join(" "), 5);
				console.log(result);
				cli.prompt();
				return;
			}

			console.log(help);
			cli.prompt();
		});

		cli.on("close", () => {
			if (this.standupTimer) {
				clearInterval(this.standupTimer);
			}
			for (const runtime of this.runtimes.values()) {
				runtime.process.kill();
			}
			process.exit(0);
		});
	}
}

async function main() {
	const workspaceRoot = __dirname;
	const office = new OfficeOrchestrator(workspaceRoot);
	await office.boot();
}

main().catch((error) => {
	console.error("Failed to start office orchestrator:", error);
	process.exit(1);
});

