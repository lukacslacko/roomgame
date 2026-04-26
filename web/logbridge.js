// Mirror console.log / .warn / .error to /log on the server. Invaluable for
// debugging on Android Chrome where remote DevTools is fiddly. Failures to
// post are silently dropped so we don't infinite-loop on transport errors.

const LEVELS = ["log", "info", "warn", "error", "debug"];

function fire(level, args) {
  let body;
  try {
    body = `[${level}] ` + args.map((a) => {
      if (a instanceof Error) return `${a.name}: ${a.message}\n${a.stack || ""}`;
      if (typeof a === "object") {
        try { return JSON.stringify(a); } catch { return String(a); }
      }
      return String(a);
    }).join(" ");
  } catch {
    body = `[${level}] (unstringifiable)`;
  }
  fetch("/log", { method: "POST", headers: { "Content-Type": "text/plain" }, body, keepalive: true })
    .catch(() => {});
}

export function hookConsoleToServer() {
  for (const lvl of LEVELS) {
    const orig = console[lvl] && console[lvl].bind(console);
    if (!orig) continue;
    console[lvl] = (...args) => {
      orig(...args);
      fire(lvl, args);
    };
  }
  window.addEventListener("error", (e) => fire("uncaught", [e.message, e.filename + ":" + e.lineno]));
  window.addEventListener("unhandledrejection", (e) => fire("unhandled", [String(e.reason)]));
}
