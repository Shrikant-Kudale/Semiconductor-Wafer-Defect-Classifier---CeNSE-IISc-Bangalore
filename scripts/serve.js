// Zero-dependency local server: static files from /public + the /api functions.
// Usage: npm start   →  http://localhost:3000
const http = require("http"), fs = require("fs"), path = require("path");
const PUB = path.join(__dirname, "..", "public"), PORT = process.env.PORT || 3000;
const MIME = { ".html": "text/html", ".js": "text/javascript", ".css": "text/css", ".json": "application/json", ".svg": "image/svg+xml", ".webmanifest": "application/manifest+json" };
http.createServer((req, res) => {
  const u = new URL(req.url, "http://x");
  if (u.pathname.startsWith("/api/")) {
    const name = u.pathname.slice(5).replace(/[^a-z]/g, "");
    let fn; try { fn = require(path.join(__dirname, "..", "api", name + ".js")); } catch { res.writeHead(404); return res.end("{}"); }
    let data = ""; req.on("data", (c) => (data += c));
    req.on("end", () => {
      req.body = data ? (() => { try { return JSON.parse(data); } catch { return data; } })() : undefined;
      res.status = (c) => { res.statusCode = c; return res; };
      res.json = (o) => { res.setHeader("Content-Type", "application/json"); res.end(JSON.stringify(o)); };
      fn(req, res);
    });
    return;
  }
  let p = path.normalize(path.join(PUB, u.pathname === "/" ? "index.html" : u.pathname));
  if (!p.startsWith(PUB) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) { res.writeHead(404); return res.end("Not found"); }
  res.writeHead(200, { "Content-Type": MIME[path.extname(p)] || "application/octet-stream" });
  fs.createReadStream(p).pipe(res);
}).listen(PORT, () => console.log(`WaferFab Intelligence → http://localhost:${PORT}`));
