import React, { useEffect, useMemo, useRef, useState } from "react";

const DEFAULT_STATE = {
  mode: "Gaussian",
  playing: false,
  playLabel: "Play",
  fps: 24,
  frameIndex: 0,
  frameCount: 0,
  currentFrameId: null,
  status: "Initializing player UI…",
  onlineGaussianEnabled: true,
  onlineScale: 1.0,
  gaussianPath: "",
  meshPath: "",
  onlineCheckpointPath: "",
  hasGaussianSequence: false,
  hasMeshSequence: false,
  hasOnlineCheckpoint: false,
  availableModeFrameCount: 0,
  theme: "light",
};

function usePlayerBridge() {
  const [bridge, setBridge] = useState(null);
  const [connected, setConnected] = useState(false);

  useEffect(() => {
    let mounted = true;

    function attachBridgeObject(obj) {
      if (!mounted || !obj) return;
      setBridge(obj);
      setConnected(true);
    }

    if (window.qt?.webChannelTransport && window.QWebChannel) {
      new window.QWebChannel(window.qt.webChannelTransport, (channel) => {
        attachBridgeObject(channel.objects.playerBridge);
      });
      return () => {
        mounted = false;
      };
    }

    const timer = window.setInterval(() => {
      if (window.qt?.webChannelTransport && window.QWebChannel) {
        window.clearInterval(timer);
        new window.QWebChannel(window.qt.webChannelTransport, (channel) => {
          attachBridgeObject(channel.objects.playerBridge);
        });
      }
    }, 250);

    return () => {
      mounted = false;
      window.clearInterval(timer);
    };
  }, []);

  return { bridge, connected };
}

function AccordionSection({ title, subtitle, defaultOpen = true, children }) {
  return (
    <details className="accordion" open={defaultOpen}>
      <summary>
        <div className="panel-header compact">
          <h2>{title}</h2>
          <span>{subtitle}</span>
        </div>
      </summary>
      <div className="accordion-content">{children}</div>
    </details>
  );
}

function App() {
  const { bridge, connected } = usePlayerBridge();
  const [state, setState] = useState(DEFAULT_STATE);
  const [busy, setBusy] = useState(false);
  const shellRef = useRef(null);

  useEffect(() => {
    if (!bridge) return;

    const hydrate = async () => {
      const raw = await bridge.getInitialState();
      setState((prev) => ({ ...prev, ...JSON.parse(raw) }));
    };

    hydrate();
    bridge.eventEmitted.connect((eventJson) => {
      const event = JSON.parse(eventJson);
      if (event?.type === "stateChanged") {
        setState((prev) => ({ ...prev, ...event.payload }));
      }
    });
  }, [bridge]);

  const dispatch = async (type, payload = {}) => {
    if (!bridge) return;
    setBusy(true);
    try {
      const raw = await bridge.dispatchCommand(JSON.stringify({ type, payload }));
      const result = JSON.parse(raw);
      if (!result.ok) {
        console.error(result.error);
      }
    } finally {
      setBusy(false);
    }
  };

  const currentFrameText = useMemo(() => {
    if (state.currentFrameId === null || state.currentFrameId === undefined) {
      return "—";
    }
    return String(state.currentFrameId);
  }, [state.currentFrameId]);

  useEffect(() => {
    const node = shellRef.current;
    if (!node) return;

    let timer = null;
    const onScroll = () => {
      node.classList.add("scroll-active");
      if (timer) window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        node.classList.remove("scroll-active");
      }, 700);
    };

    node.addEventListener("scroll", onScroll, { passive: true });
    return () => {
      node.removeEventListener("scroll", onScroll);
      if (timer) window.clearTimeout(timer);
    };
  }, []);

  return (
    <div ref={shellRef} className="app-shell" data-theme={state.theme || "dark"}>
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />
      <div className="ambient ambient-c" />

      <section className="hero-card compact-hero">
        <div className="hero-left">
          <div className="hero-title-row">
            <h1>Control Center</h1>
            <span className="hero-subtitle">React shell · Native Render</span>
          </div>
        </div>

        <div className="hero-actions">
          <div className="mini-segmented">
            {["Gaussian", "Mesh", "Split"].map((mode) => (
              <button
                key={mode}
                className={state.mode === mode ? "active" : ""}
                onClick={() => dispatch("setMode", { mode })}
                disabled={busy}
              >
                {mode}
              </button>
            ))}
          </div>

          <button
            className="theme-toggle"
            onClick={() =>
              dispatch("setTheme", {
                theme: state.theme === "dark" ? "light" : "dark",
              })
            }
            disabled={busy}
          >
            {state.theme === "dark" ? "☀ Day" : "🌙 Night"}
          </button>
        </div>

        <div className="badge-row compact-badges">
          <span className={`badge ${connected ? "ok" : "warn"}`}>
            {connected ? "Bridge Ready" : "Waiting Bridge"}
          </span>
          <span className={`badge ${state.playing ? "accent" : ""}`}>
            {state.playing ? "Playing" : "Paused"}
          </span>
          <span className="badge">{state.mode}</span>
        </div>
      </section>

      <AccordionSection title="Playback" subtitle={`${state.frameCount} frames`} defaultOpen>
        <section className="panel inner-panel">
          <div className="playback-row">
            <button
              className="icon-button"
              onClick={() => dispatch("stepFrame", { delta: -1 })}
              disabled={busy || state.frameCount < 1}
              title="Previous Frame"
            >
              ⟨
            </button>
            <button
              className="primary playback-button"
              onClick={() => dispatch("togglePlayback")}
              disabled={busy}
            >
              {state.playing ? "Pause" : "Play"}
            </button>
            <button
              className="icon-button"
              onClick={() => dispatch("stepFrame", { delta: 1 })}
              disabled={busy || state.frameCount < 1}
              title="Next Frame"
            >
              ⟩
            </button>
          </div>

          <div className="timeline-meta">
            <div>
              <label>Current Frame</label>
              <strong>{currentFrameText}</strong>
            </div>
            <div>
              <label>Frame Index</label>
              <strong>{state.frameIndex}</strong>
            </div>
            <div>
              <label>FPS</label>
              <input
                type="number"
                min="1"
                max="60"
                value={state.fps}
                onChange={(e) => dispatch("setFps", { fps: Number(e.target.value || 1) })}
                disabled={busy}
              />
            </div>
          </div>

          <input
            className="timeline"
            type="range"
            min="0"
            max={Math.max(0, state.frameCount - 1)}
            value={Math.min(state.frameIndex, Math.max(0, state.frameCount - 1))}
            onChange={(e) => dispatch("setFrameIndex", { index: Number(e.target.value) })}
            disabled={busy || state.frameCount < 1}
          />
        </section>
      </AccordionSection>

      <AccordionSection title="Sources" subtitle="Open local assets" defaultOpen>
        <section className="panel inner-panel">
          <div className="source-card">
            <label>Online Model</label>
            <div className="source-path">{state.onlineCheckpointPath || "Not selected"}</div>
            <button onClick={() => dispatch("chooseOnlineModelDir")} disabled={busy}>
              Open Online Model Dir
            </button>
          </div>

          <div className="source-card">
            <label>Gaussian Sequence</label>
            <div className="source-path">{state.gaussianPath || "Not selected"}</div>
            <button onClick={() => dispatch("chooseGaussianDir")} disabled={busy}>
              Open Gaussian Dir
            </button>
          </div>

          <div className="source-card">
            <label>Mesh Sequence</label>
            <div className="source-path">{state.meshPath || "Not selected"}</div>
            <button onClick={() => dispatch("chooseMeshDir")} disabled={busy}>
              Open Mesh Dir
            </button>
          </div>
        </section>
      </AccordionSection>

      <AccordionSection title="View Settings" subtitle="Frontend only" defaultOpen={false}>
        <section className="panel inner-panel">
          <label className="toggle-row">
            <span>Online Gaussian</span>
            <input
              type="checkbox"
              checked={!!state.onlineGaussianEnabled}
              onChange={(e) =>
                dispatch("setOnlineGaussianEnabled", { enabled: e.target.checked })
              }
              disabled={busy}
            />
          </label>

          <div className="slider-block">
            <div className="slider-header">
              <span>GS Scale</span>
              <strong>{Number(state.onlineScale || 1).toFixed(2)}x</strong>
            </div>
            <input
              type="range"
              min="0.25"
              max="1.0"
              step="0.05"
              value={state.onlineScale ?? 1}
              onChange={(e) => dispatch("setOnlineScale", { scale: Number(e.target.value) })}
              disabled={busy}
            />
          </div>

          <div className="badge-row">
            <span className={`badge ${state.hasOnlineCheckpoint ? "ok" : "warn"}`}>
              {state.hasOnlineCheckpoint ? "Online Ready" : "No Online Dir"}
            </span>
            <span className={`badge ${state.hasGaussianSequence ? "ok" : "warn"}`}>
              {state.hasGaussianSequence ? "Gaussian Loaded" : "No Gaussian Seq"}
            </span>
            <span className={`badge ${state.hasMeshSequence ? "ok" : "warn"}`}>
              {state.hasMeshSequence ? "Mesh Loaded" : "No Mesh Seq"}
            </span>
          </div>
        </section>
      </AccordionSection>

      <AccordionSection title="Status" subtitle="Runtime info" defaultOpen={false}>
        <section className="panel inner-panel status-panel">
          <div className="panel-header">
            <h2>Status</h2>
            <button onClick={() => dispatch("requestState")} disabled={busy}>
              Refresh
            </button>
          </div>
          <div className="status-text">{state.status}</div>
          <div className="status-meta">
            <div>
              <label>Visible Mode</label>
              <strong>{state.mode}</strong>
            </div>
            <div>
              <label>Mode Frames</label>
              <strong>{state.availableModeFrameCount}</strong>
            </div>
          </div>
        </section>
      </AccordionSection>
    </div>
  );
}

export default App;
