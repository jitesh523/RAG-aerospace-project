export class RAGClient {
  constructor({ baseUrl, apiKey, timeoutMs = 10000 }) {
    this.baseUrl = baseUrl.replace(/\/$/, "");
    this.apiKey = apiKey;
    this.timeoutMs = timeoutMs;
  }
  async ask(query) {
    const controller = new AbortController();
    const to = setTimeout(() => controller.abort(), this.timeoutMs);
    const headers = { "Content-Type": "application/json" };
    if (this.apiKey) headers["x-api-key"] = this.apiKey;
    const res = await fetch(`${this.baseUrl}/ask`, {
      method: "POST",
      headers,
      body: JSON.stringify({ query }),
      signal: controller.signal,
    });
    clearTimeout(to);
    if (!res.ok) {
      const txt = await res.text();
      throw new Error(`HTTP ${res.status}: ${txt}`);
    }
    return res.json();
  }
}
