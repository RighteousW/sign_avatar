import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import { flushSync } from "react-dom";
import { registerLibraryTool } from "./model-context";
import signs from "../../shared/signs.json";
import "./style.css";

type Page =
  | "Translate"
  | "Sign library"
  | "Dataset"
  | "Experiments"
  | "Settings";
type Sign = (typeof signs)[number];
const pages: Page[] = [
  "Translate",
  "Sign library",
  "Dataset",
  "Experiments",
  "Settings",
];
const glyphs = ["↔", "▦", "▤", "◴", "⚙"];
function App() {
  const [page, setPage] = useState<Page>("Translate");
  const [health, setHealth] = useState<"checking" | "online" | "offline">(
    "checking",
  );
  const [direction, setDirection] = useState("recognise");
  const [query, setQuery] = useState("");
  const [category, setCategory] = useState("all");
  const [selected, setSelected] = useState<Sign | null>(null);
  const [text, setText] = useState("");
  const [tokens, setTokens] = useState<string[] | null>(null);
  const [theme, setTheme] = useState(() => {
    try {
      return localStorage.getItem("sign-avatar-theme") || "light";
    } catch {
      return "light";
    }
  });
  useEffect(
    () =>
      registerLibraryTool((query, category) =>
        flushSync(() => {
          setPage("Sign library");
          setQuery(query);
          setCategory(category);
          setSelected(null);
        }),
      ),
    [],
  );
  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    try {
      localStorage.setItem("sign-avatar-theme", theme);
    } catch {}
  }, [theme]);
  useEffect(() => {
    const control = new AbortController();
    const timer = setTimeout(() => control.abort(), 4000);
    fetch("/api/health", { signal: control.signal })
      .then((r) => {
        if (!r.ok) throw Error();
        return r.json();
      })
      .then((d) =>
        setHealth(d.service === "sign-avatar" ? "online" : "offline"),
      )
      .catch(() => setHealth("offline"))
      .finally(() => clearTimeout(timer));
    return () => {
      control.abort();
      clearTimeout(timer);
    };
  }, []);
  const filtered = signs.filter(
    (s) =>
      (category === "all" || category === s.category) &&
      (s.label.toLowerCase().includes(query.toLowerCase()) ||
        s.id.includes(query.toLowerCase())),
  );
  return (
    <div className="app">
      <aside className="sidebar">
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            setPage("Translate");
          }}
        >
          <span className="brandmark">sa</span>
          <span>
            Sign Avatar<small>NSL WORKSPACE</small>
          </span>
        </a>
        <nav aria-label="Main navigation">
          {pages.map((p, i) => (
            <button
              key={p}
              aria-current={page === p ? "page" : undefined}
              className={page === p ? "active" : ""}
              onClick={() => {
                setPage(p);
              }}
            >
              <span aria-hidden>{glyphs[i]}</span>
              {p}
            </button>
          ))}
        </nav>
        <div className="sidebar-foot">
          <span className="pill">Research prototype</span>
          <p>Namibian Sign Language</p>
          <small>Recovered 49-class baseline</small>
        </div>
      </aside>
      <div className="body">
        <header className="topbar">
          <span>{page}</span>
          <div>
            <span className={"connection " + health}>
              {health === "online"
                ? "Backend connected"
                : health === "checking"
                  ? "Checking connection"
                  : "Backend offline"}
            </span>
            <button
              className="icon-button"
              aria-label={
                theme === "light" ? "Use dark theme" : "Use light theme"
              }
              onClick={() => setTheme(theme === "light" ? "dark" : "light")}
            >
              {theme === "light" ? "☾" : "☀"}
            </button>
          </div>
        </header>
        <main>
          <div className="heading">
            <div>
              <p className="eyebrow">SIGN AVATAR / NSL</p>
              <h1>
                {page === "Translate" ? "Make room for understanding." : page}
              </h1>
              <p className="muted">
                {page === "Translate"
                  ? "Capture a sign or explore the written-to-sign flow."
                  : page === "Sign library"
                    ? "The vocabulary behind the recovered recognition models."
                    : page === "Dataset"
                      ? "Keep recordings, derived samples and review decisions traceable."
                      : page === "Experiments"
                        ? "Separate historical evidence from new validation."
                        : "Preferences for this browser and the local workspace."}
              </p>
            </div>
            {page === "Sign library" && (
              <span className="count">49 labels</span>
            )}
          </div>
          {page === "Translate" && (
            <>
              <div
                className="tabs"
                role="tablist"
                aria-label="Translation direction"
              >
                <button
                  role="tab"
                  aria-selected={direction === "recognise"}
                  onClick={() => setDirection("recognise")}
                >
                  Sign to text <span>→</span>
                </button>
                <button
                  role="tab"
                  aria-selected={direction === "visualise"}
                  onClick={() => setDirection("visualise")}
                >
                  Text to sign <span>→</span>
                </button>
              </div>
              {direction === "recognise" ? (
                <div className="workspace">
                  <Capture />
                  <section className="panel result">
                    <div className="panel-title">
                      <h2>Recognition</h2>
                      <span className="pill">Not connected</span>
                    </div>
                    <div className="result-empty">
                      <span className="result-glyph" aria-hidden>
                        “
                      </span>
                      <h3>A sign starts here.</h3>
                      <p>
                        Record one isolated sign. The prediction and
                        alternatives will appear here once recognition is
                        connected.
                      </p>
                    </div>
                    <div className="result-footer">
                      <p className="eyebrow">CURRENT BASELINE</p>
                      <strong>49 classes · hands only</strong>
                      <p className="muted">26 letters · 10 digits · 13 words</p>
                      <button
                        className="text-button"
                        onClick={() => setPage("Sign library")}
                      >
                        Explore supported labels →
                      </button>
                    </div>
                  </section>
                </div>
              ) : (
                <div className="workspace">
                  <section className="panel">
                    <div className="panel-title">
                      <h2>Written input</h2>
                      <span className="pill">Vocabulary lookup</span>
                    </div>
                    <label htmlFor="text-input">
                      English words or a single letter
                    </label>
                    <textarea
                      id="text-input"
                      value={text}
                      onChange={(e) => {
                        setText(e.target.value);
                        setTokens(null);
                      }}
                      placeholder="Try: book, again, after"
                      maxLength={1000}
                    />
                    <p className="muted">
                      This checks exact vocabulary matches. Sentence translation
                      and sign playback are not connected yet.
                    </p>
                    <button
                      className="primary"
                      disabled={!text.trim()}
                      onClick={() =>
                        setTokens(
                          text
                            .trim()
                            .toLowerCase()
                            .split(/[\s,.;!?]+/)
                            .filter(Boolean),
                        )
                      }
                    >
                      Check vocabulary
                    </button>
                  </section>
                  <section className="panel">
                    <div className="panel-title">
                      <h2>Sign sequence</h2>
                      <span className="pill">Playback unavailable</span>
                    </div>
                    {tokens ? (
                      <>
                        <div className="token-list">
                          {tokens.map((t, i) => {
                            const s = signs.find(
                              (s) =>
                                s.id === t ||
                                (s.category === "letter" &&
                                  s.label.toLowerCase() === t),
                            );
                            return (
                              <span
                                key={i}
                                className={"token " + (!s ? "unsupported" : "")}
                              >
                                {s?.label || t}
                                {!s && <small>Not in vocabulary</small>}
                              </span>
                            );
                          })}
                        </div>
                        <p className="muted">
                          Order and repetitions are preserved. A match does not
                          confirm NSL grammar or meaning; “I” here is an
                          alphabet lookup, not a pronoun translation.
                        </p>
                      </>
                    ) : (
                      <div className="result-empty">
                        <h3>From words to movement.</h3>
                        <p>
                          Check the vocabulary to see which labels are
                          available. Reviewed playback assets will be added in a
                          later stage.
                        </p>
                      </div>
                    )}
                  </section>
                </div>
              )}
              <section className="strip">
                <div>
                  <p className="eyebrow">A DEFINED STARTING POINT</p>
                  <h3>One sign at a time.</h3>
                  <p>
                    Use an isolated sign from the library, with both hands in
                    frame.
                  </p>
                </div>
                <button
                  className="secondary"
                  onClick={() => setPage("Experiments")}
                >
                  View baseline evidence →
                </button>
              </section>
            </>
          )}
          {page === "Sign library" && (
            <>
              <div className="toolbar">
                <label className="search">
                  <span aria-hidden>⌕</span>
                  <input
                    aria-label="Search signs"
                    placeholder="Search the vocabulary…"
                    maxLength={100}
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                  />
                </label>
                <select
                  aria-label="Label category"
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                >
                  <option value="all">All labels</option>
                  <option value="letter">Letters · 26</option>
                  <option value="digit">Digits · 10</option>
                  <option value="word">Words · 13</option>
                </select>
                <span className="muted">{filtered.length} results</span>
              </div>
              <div className="library-layout">
                <div className="sign-grid">
                  {filtered.map((s) => (
                    <button
                      className={
                        "sign-card " + (selected?.id === s.id ? "selected" : "")
                      }
                      key={s.id}
                      onClick={() => setSelected(s)}
                    >
                      <span className="eyebrow">{s.category}</span>
                      <strong>{s.label}</strong>
                      <span>View details ↗</span>
                    </button>
                  ))}
                  {!filtered.length && <p>No labels match your search.</p>}
                </div>
                <section className="panel sign-detail" aria-live="polite">
                  {selected ? (
                    <>
                      <span className="pill">{selected.category}</span>
                      <h2>{selected.label}</h2>
                      <dl>
                        <dt>Language</dt>
                        <dd>Namibian Sign Language</dd>
                        <dt>Model label</dt>
                        <dd>{selected.id}</dd>
                        <dt>Language review</dt>
                        <dd>Not recorded</dd>
                        <dt>Example playback</dt>
                        <dd>Asset not connected</dd>
                      </dl>
                      <p className="muted">
                        This is a recovered model label. Its meaning and sign
                        form still need a documented linguistic review.
                      </p>
                      <button
                        className="secondary"
                        onClick={() => {
                          setPage("Translate");
                          setDirection("recognise");
                        }}
                      >
                        Open capture workspace
                      </button>
                    </>
                  ) : (
                    <>
                      <h2>Explore a label</h2>
                      <p className="muted">
                        Select a card to inspect its model identity and review
                        status.
                      </p>
                    </>
                  )}
                </section>
              </div>
            </>
          )}
          {page === "Dataset" && (
            <>
              <div className="metrics">
                <Metric value="4,900" label="Recovered landmark samples" />
                <Metric value="2,450" label="Original-named samples" />
                <Metric value="2,450" label="Flipped derivatives" />
              </div>
              <div className="notice">
                Historical inventory, confirmed during recovery. These files are
                on your machine; they have not been imported into this browser
                application.
              </div>
              <div className="workspace">
                <section className="panel">
                  <h2>From recording to reviewed sample</h2>
                  <ol className="steps">
                    <li>
                      <strong>Record or import</strong>
                      <p>Associate a sign, signer and recording session.</p>
                    </li>
                    <li>
                      <strong>Review quality</strong>
                      <p>Check visibility, boundaries and label correctness.</p>
                    </li>
                    <li>
                      <strong>Prepare landmarks</strong>
                      <p>
                        Keep the source recording and its derived samples
                        linked.
                      </p>
                    </li>
                    <li>
                      <strong>Assign a split</strong>
                      <p>
                        Keep related samples together; reserve new sessions or
                        signers.
                      </p>
                    </li>
                  </ol>
                </section>
                <section className="panel">
                  <h2>Dataset connection</h2>
                  <p className="muted">
                    The import and review workflow is planned for the data
                    stage. Original videos were not recovered from Git history.
                  </p>
                  <button disabled className="primary">
                    Import dataset — not connected
                  </button>
                  <p className="muted">
                    You can capture and download a new video from Translate
                    today.
                  </p>
                  <button
                    className="text-button"
                    onClick={() => {
                      setPage("Translate");
                      setDirection("recognise");
                    }}
                  >
                    Open camera capture →
                  </button>
                </section>
              </div>
            </>
          )}
          {page === "Experiments" && (
            <>
              <div className="notice">
                Historical checkpoint values supplied during recovery. These are
                model-selection validation scores, not new test results or
                unseen-signer performance.
              </div>
              <section className="panel">
                <div className="panel-title">
                  <h2>Recovered checkpoints</h2>
                  <span className="pill">49 classes</span>
                </div>
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        <th>Model</th>
                        <th>Training</th>
                        <th>Validation</th>
                        <th>Load check</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[
                        ["No skip", "97.06%", "98.10%"],
                        ["Skip 1", "97.64%", "96.87%"],
                        ["Skip 2", "97.32%", "98.50%"],
                      ].map((r) => (
                        <tr key={r[0]}>
                          {r.map((x) => (
                            <td key={x}>{x}</td>
                          ))}
                          <td>Passed locally</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </section>
              <div className="workspace below">
                <section className="panel">
                  <p className="eyebrow">RECOVERY CHECK</p>
                  <h2>48 / 49 selected samples</h2>
                  <p>
                    The no-skip model matched one selected example from 48
                    classes. Alphabet U was predicted as R.
                  </p>
                  <p className="muted">
                    These examples may have been used in training. This confirms
                    recovered assets work together, not independent test
                    accuracy.
                  </p>
                </section>
                <section className="panel">
                  <h2>Next validation run</h2>
                  <p className="muted">
                    Fresh recordings, an explicit split manifest and per-class
                    errors will establish a new baseline. The historical
                    filename order is unknown, so its exact split cannot be
                    assumed reproducible.
                  </p>
                  <button disabled className="primary">
                    Run evaluation — not connected
                  </button>
                </section>
              </div>
            </>
          )}
          {page === "Settings" && (
            <section className="panel settings">
              <h2>Appearance</h2>
              <label htmlFor="theme">Theme</label>
              <select
                id="theme"
                value={theme}
                onChange={(e) => setTheme(e.target.value)}
              >
                <option value="light">Light</option>
                <option value="dark">Dark</option>
              </select>
              <p className="muted">Saved on this browser only.</p>
              <hr />
              <h2>Connection</h2>
              <p>
                {health === "online"
                  ? "The local API is reachable."
                  : "The local API is not currently reachable."}
              </p>
              <p className="muted">
                Recognition, dataset imports and training are not implemented in
                this first interface stage.
              </p>
              <hr />
              <h2>Camera and microphone</h2>
              <p className="muted">
                The browser asks for camera permission when you open the camera.
                Video capture currently records without audio. Camera access
                requires localhost or HTTPS.
              </p>
              <button
                className="secondary"
                onClick={() => {
                  setPage("Translate");
                  setDirection("recognise");
                }}
              >
                Open capture settings
              </button>
            </section>
          )}
          <footer>
            Sign Avatar <span>Namibian Sign Language · Research prototype</span>
          </footer>
        </main>
      </div>
    </div>
  );
}
function Metric({ value, label }: { value: string; label: string }) {
  return (
    <div className="panel metric">
      <strong>{value}</strong>
      <span>{label}</span>
    </div>
  );
}
function Capture() {
  const video = useRef<HTMLVideoElement>(null);
  const stream = useRef<MediaStream | null>(null);
  const recorder = useRef<MediaRecorder | null>(null);
  const timeout = useRef<ReturnType<typeof setTimeout> | null>(null);
  const mounted = useRef(true);
  const urlRef = useRef("");
  const [state, setState] = useState<"off" | "opening" | "ready" | "recording">(
    "off",
  );
  const [error, setError] = useState("");
  const [url, setUrl] = useState("");
  const [filename, setFilename] = useState("sign.webm");
  function release() {
    if (timeout.current) clearTimeout(timeout.current);
    if (recorder.current?.state === "recording") recorder.current.stop();
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
  }
  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      release();
      if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    };
  }, []);
  function showBlob(blob: Blob, name: string) {
    if (urlRef.current) URL.revokeObjectURL(urlRef.current);
    urlRef.current = URL.createObjectURL(blob);
    setUrl(urlRef.current);
    setFilename(name);
  }
  async function open() {
    setError("");
    setState("opening");
    try {
      if (!navigator.mediaDevices?.getUserMedia)
        throw Error(
          "Camera access needs localhost or HTTPS and a supported browser.",
        );
      const s = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      if (!mounted.current) {
        s.getTracks().forEach((t) => t.stop());
        return;
      }
      stream.current = s;
      if (video.current) {
        video.current.srcObject = s;
        await video.current.play();
      }
      setState("ready");
    } catch (e) {
      release();
      if (mounted.current) {
        setState("off");
        setError(
          e instanceof Error
            ? e.message
            : "Could not open the camera. Check browser permissions.",
        );
      }
    }
  }
  function start() {
    if (!stream.current) return;
    setError("");
    try {
      const type = ["video/webm;codecs=vp8", "video/webm", "video/mp4"].find(
        (t) => MediaRecorder.isTypeSupported(t),
      );
      const r = new MediaRecorder(
        stream.current,
        type ? { mimeType: type } : undefined,
      );
      recorder.current = r;
      const chunks: BlobPart[] = [];
      r.ondataavailable = (e) => {
        if (e.data.size) chunks.push(e.data);
      };
      r.onstop = () => {
        if (timeout.current) clearTimeout(timeout.current);
        if (!mounted.current) return;
        showBlob(
          new Blob(chunks, { type: r.mimeType }),
          `sign-${Date.now()}.${r.mimeType.includes("mp4") ? "mp4" : "webm"}`,
        );
        setState(stream.current ? "ready" : "off");
      };
      r.onerror = () => {
        if (mounted.current)
          setError("Recording failed. Stop the camera and try again.");
      };
      r.start();
      setState("recording");
      timeout.current = setTimeout(() => {
        if (r.state === "recording") r.stop();
      }, 15000);
    } catch (e) {
      setError(
        e instanceof Error ? e.message : "Video recording is unavailable.",
      );
    }
  }
  return (
    <section className="panel capture">
      <div className="panel-title">
        <h2>Capture a sign</h2>
        <span className="pill">
          {state === "recording"
            ? "Recording · max 15s"
            : state === "ready"
              ? "Camera ready"
              : "Video input"}
        </span>
      </div>
      <div className="camera">
        <video
          ref={video}
          muted
          playsInline
          className={state === "off" || state === "opening" ? "hidden" : ""}
        />
        {(state === "off" || state === "opening") && (
          <div className="camera-empty">
            <div className="viewfinder" aria-hidden>
              ＋
            </div>
            <h3>
              {state === "opening"
                ? "Opening your camera…"
                : "Your signing space"}
            </h3>
            <p>
              Keep both hands visible.
              <br />
              Perform one sign at a time.
            </p>
          </div>
        )}
        {state === "recording" && (
          <span className="recording-label">● Recording</span>
        )}
      </div>
      <div className="capture-controls">
        {state === "off" || state === "opening" ? (
          <button
            className="primary"
            disabled={state === "opening"}
            onClick={open}
          >
            Open camera
          </button>
        ) : (
          <>
            <button
              className="primary"
              onClick={
                state === "recording" ? () => recorder.current?.stop() : start
              }
            >
              {state === "recording" ? "Stop recording" : "Record sign"}
            </button>
            <button
              className="secondary"
              onClick={() => {
                release();
                setState("off");
              }}
            >
              Close camera
            </button>
          </>
        )}
        <label className="upload">
          Choose video
          <input
            type="file"
            accept="video/*"
            disabled={state === "recording" || state === "opening"}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (!f) return;
              if (!f.type.startsWith("video/")) {
                setError("Choose a supported video file.");
                return;
              }
              if (f.size > 100 * 1024 * 1024) {
                setError("Choose a video smaller than 100 MB.");
                return;
              }
              setError("");
              showBlob(f, f.name);
              e.target.value = "";
            }}
          />
        </label>
      </div>
      {error && (
        <p role="alert" className="error">
          {error}
        </p>
      )}
      <p className="muted small">
        Video stays in this browser. Nothing is uploaded or classified in this
        stage.
      </p>
      {url && (
        <div className="clip">
          <h3>Review your clip</h3>
          <video
            src={url}
            controls
            playsInline
            onError={() =>
              setError("This browser cannot play that video format.")
            }
          />
          <a className="text-button" href={url} download={filename}>
            Download recording ↓
          </a>
        </div>
      )}
    </section>
  );
}
createRoot(document.getElementById("root")!).render(<App />);
