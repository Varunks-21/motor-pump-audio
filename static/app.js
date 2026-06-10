/* MotorPump live monitor client */
(function () {
  "use strict";

  const STATUS_COLORS = { green: "#2ecc71", yellow: "#f1c40f", red: "#ff4757" };
  const STATUS_LABELS = {
    green: "normal operation",
    yellow: "uncertain / background",
    red: "anomaly detected",
  };

  const $ = (id) => document.getElementById(id);

  let toastTimer = null;
  function showToast(msg) {
    const t = $("toast");
    if (!t) return;
    t.textContent = msg;
    t.classList.add("show");
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => t.classList.remove("show"), 5000);
  }

  function setStatus(color, cls, label) {
    $("status-card").style.setProperty("--status", color);
    $("status-class").textContent = cls;
    $("status-label").textContent = label;
  }

  /* ---- history search + date seeding (works WITHOUT the realtime socket) ---- */
  function setupHistory() {
    const fmtLocal = (d) => {
      const pad = (n) => String(n).padStart(2, "0");
      return (
        d.getFullYear() + "-" + pad(d.getMonth() + 1) + "-" + pad(d.getDate()) +
        "T" + pad(d.getHours()) + ":" + pad(d.getMinutes())
      );
    };
    const now = new Date();
    $("search-to").value = fmtLocal(now);
    $("search-from").value = fmtLocal(new Date(now.getTime() - 3600 * 1000));

    $("search-btn").addEventListener("click", () => {
      const url =
        "/api/search?start=" + encodeURIComponent($("search-from").value) +
        "&end=" + encodeURIComponent($("search-to").value);
      fetch(url)
        .then((r) => r.json())
        .then(renderHistory)
        .catch((e) => showToast("search failed: " + e));
    });
  }

  function renderHistory(d) {
    const body = $("history-body");
    $("search-summary").textContent =
      d.count + " predictions \u00b7 " + d.anomalies + " anomalies";
    if (!d.rows || !d.rows.length) {
      body.innerHTML =
        '<tr><td colspan="8" class="empty-row">No predictions in this range.</td></tr>';
      return;
    }
    body.innerHTML = d.rows
      .map((r) => {
        const tag = "tag tag-" + r.status;
        return (
          "<tr>" +
          "<td>" + ((r.ts.split(" ")[1]) || r.ts) + "</td>" +
          "<td>" + r.predicted_class + "</td>" +
          "<td>" + (r.confidence * 100).toFixed(0) + "%</td>" +
          '<td><span class="' + tag + '">' + r.status + "</span></td>" +
          "<td>" + (r.anomaly ? "\u26a0 yes" : "no") + "</td>" +
          "<td>" + Number(r.ae_score).toFixed(4) + "</td>" +
          "<td>" + r.start_time + "\u2013" + r.end_time + "</td>" +
          "<td>" + r.duration + "s</td>" +
          "</tr>"
        );
      })
      .join("");
  }

  /* ---- live detection (needs Socket.IO) ---- */
  function setupLive() {
    if (typeof io === "undefined") {
      setStatus("#ff4757", "NO CONNECTION", "Socket.IO client failed to load");
      showToast("Realtime client (Socket.IO) failed to load \u2014 live detection disabled. " +
                "Vendor static/socket.io.min.js (see setup notes). History search still works.");
      return;
    }

    const socket = io();
    let listening = false;
    const toggleBtn = $("toggle-btn");
    const micPill = $("mic-pill");
    const deviceSelect = $("device-select");

    socket.on("connect", () => console.log("[motorpump] socket connected", socket.id));
    socket.on("connect_error", (e) =>
      showToast("connection error: " + ((e && e.message) || e)));
    socket.on("disconnect", () => {
      micPill.textContent = "\u25cf MIC OFFLINE";
      micPill.className = "pill pill-off";
    });

    fetch("/api/devices")
      .then((r) => r.json())
      .then((d) => {
        (d.devices || []).forEach((dev) => {
          const opt = document.createElement("option");
          opt.value = dev.index;
          opt.textContent = dev.name;
          deviceSelect.appendChild(opt);
        });
      })
      .catch(() => {});

    toggleBtn.addEventListener("click", () => {
      if (listening) socket.emit("stop", {});
      else socket.emit("start", { device: deviceSelect.value });
    });

    socket.on("status", (s) => {
      listening = !!s.listening;
      toggleBtn.textContent = listening ? "STOP" : "START";
      toggleBtn.classList.toggle("is-stop", listening);
      micPill.textContent = listening ? "\u25cf MIC LIVE" : "\u25cf MIC OFFLINE";
      micPill.className = "pill " + (listening ? "pill-on" : "pill-off");
      if (!listening) {
        setStatus("#4b5563", "STANDBY", "awaiting input");
        $("alarm").classList.remove("show");
      }
    });

    socket.on("error", (e) => showToast((e && e.message) || "error"));

    socket.on("prediction", (p) => {
      setStatus(STATUS_COLORS[p.status] || "#4b5563",
                p.predicted_class, STATUS_LABELS[p.status] || "");
      $("conf-val").textContent = (p.confidence * 100).toFixed(0) + "%";
      $("conf-bar").style.width = Math.min(100, p.confidence * 100) + "%";
      $("ae-val").textContent = p.ae_score.toFixed(4) + " / " + p.ae_threshold.toFixed(4);
      const aeRatio = p.ae_threshold > 0 ? p.ae_score / (p.ae_threshold * 2) : 0;
      $("ae-bar").style.width = Math.min(100, aeRatio * 100) + "%";
      $("alarm").classList.toggle("show", !!p.anomaly);
      if (p.spectrogram) {
        const img = $("spectrogram");
        img.src = "data:image/png;base64," + p.spectrogram;
        img.style.display = "block";
        $("spec-empty").style.display = "none";
      }
      addRecent(p);
    });

    function addRecent(p) {
      const list = $("recent-list");
      const li = document.createElement("li");
      if (p.status === "red") li.classList.add("is-red");
      const color = STATUS_COLORS[p.status] || "#4b5563";
      li.innerHTML =
        '<span class="recent-time">' + p.end_time + "</span>" +
        '<span class="recent-class"><span class="dot" style="background:' + color +
        '"></span>' + p.predicted_class + "</span>" +
        '<span class="recent-conf">' + (p.confidence * 100).toFixed(0) + "%</span>";
      list.prepend(li);
      while (list.children.length > 30) list.removeChild(list.lastChild);
    }
  }

  function boot() {
    setupHistory();   // always works
    setupLive();      // needs Socket.IO
  }

  if (document.readyState !== "loading") boot();
  else document.addEventListener("DOMContentLoaded", boot);
})();