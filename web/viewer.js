/* The HTML generator inlines this file and the shared, independently tested modules. */
(function () {
  "use strict";
  const data = JSON.parse(document.getElementById("meeting-data").textContent);
  const $ = id => document.getElementById(id);
  const canvas = $("thread"), ctx = canvas.getContext("2d"), stage = canvas.parentElement;
  const M = ThreadMath, {Playback, BufferedPlayback} = ThreadPlayback;
  for (const meeting of data.meetings) {
    const binary = atob(meeting.embeddings), bytes = Uint8Array.from(binary, ch => ch.charCodeAt(0));
    const values = new Float32Array(bytes.buffer);
    meeting.vectors = meeting.sentences.map((_, i) => values.subarray(i * 384, (i + 1) * 384));
    const option = document.createElement("option");
    option.value = meeting.id; option.textContent = meeting.id; $("meeting").append(option);
  }
  let meetingIndex = 0, meeting = data.meetings[0], geometry;
  let player = new Playback(meeting.sentences.length), camera = [0, 0], lastTime = null;
  let width = 1, height = 1, pcaReady = false, lastSentence = -1;
  for (const budget of Object.keys(meeting.context?.profiles || {})) {
    const option = document.createElement("option");
    option.value = budget; option.textContent = `${budget} tokens`; $("lookahead").append(option);
  }
  $("lookahead").value = String(data.defaultFuture || 48);
  if (!meeting.context) $("view").value = "causal";
  const isFuture = () => $("view").value === "thread" && Boolean(meeting.context);
  const profile = () => meeting.context.profiles[$("lookahead").value];

  function rebuild(preserveProgress = false) {
    const progress = preserveProgress ? player.progress : 0, speed = player.speed;
    if (isFuture()) {
      geometry = M.futureTrajectory(meeting.coordinates, profile(), +$("turn").value);
      player = new BufferedPlayback(profile().readyAt);
    } else {
      geometry = M.trajectory(meeting.vectors, meeting.coordinates, +$("context").value, +$("turn").value);
      player = new Playback(meeting.sentences.length);
    }
    player.setSpeed(speed); player.seek(progress);
    camera = M.frame(geometry, player.progress).tip.slice(); lastSentence = -1;
    $("timeline").max = String(meeting.sentences.length);
    sync(); draw(0);
  }

  function sync() {
    const pca = $("view").value === "pca";
    $("play").textContent = player.playing ? "Pause" : player.finished ? "Replay" : "Play";
    $("play").setAttribute("aria-label", $("play").textContent);
    for (const id of ["play", "restart", "speed", "turn", "timeline"]) $(id).disabled = pca;
    $("context").disabled = $("view").value !== "causal";
    $("lookahead").disabled = !isFuture();
    $("causal-context").hidden = $("view").value !== "causal";
    $("lookahead-label").hidden = !isFuture();
    $("context-hint").textContent = isFuture()
      ? `Past: ${meeting.context.pastBudget} tokens. Lookahead buffers incoming sentences.`
      : "Changing context or turn resets the thread.";
    $("thread-label").firstChild.textContent = isFuture() ? "FUTURE-AWARE THREAD" : "ORIGINAL CAUSAL THREAD";
    $("status").textContent = pca ? "PCA comparison · Playback paused" : player.playing
      ? player.buffering ? `Buffering · ${player.received} / ${player.count} sentences received` : "Playing"
      : player.finished ? "Complete" : "Paused";
    const index = M.frame(geometry, player.progress).sentenceIndex;
    $("timeline").value = String(player.progress + 1);
    $("timeline").setAttribute("aria-valuetext", `Sentence ${index + 1} of ${meeting.sentences.length}`);
    $("position").textContent = `${index + 1} / ${meeting.sentences.length}`;
    const format = value => value === null ? "unavailable" : value.toFixed(2);
    $("alignment").textContent = isFuture()
      ? `Past ${format(geometry.alignment[index])} · Future ${format(geometry.futureAlignment[index])} · Context ${profile().pastTokens[index]} / ${profile().futureTokens[index]} tokens`
      : "";
    $("alignment").hidden = !isFuture();
    if (index !== lastSentence) {
      $("sentence-number").textContent = `SENTENCE ${index + 1}`;
      $("sentence").textContent = meeting.sentences[index];
      $("sentence").scrollTop = 0;
      lastSentence = index;
    }
  }

  function curveTo(curve) {
    ctx.bezierCurveTo(...curve.c1, ...curve.c2, ...curve.p1);
  }

  function draw(seconds) {
    if ($("view").value === "pca") return;
    const frame = M.frame(geometry, player.progress);
    camera = M.follow(camera, frame.tip, seconds);
    const ratio = window.devicePixelRatio || 1;
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    ctx.translate(width / 2, height / 2);
    ctx.scale(36, -36);
    ctx.translate(-camera[0], -camera[1]);
    ctx.beginPath(); ctx.moveTo(0, 0);
    for (let i = 0; i < frame.completed; i++) curveTo(geometry.curves[i]);
    if (frame.partial) curveTo(frame.partial);
    ctx.strokeStyle = "#4cdbca"; ctx.lineWidth = 1.8 / 36; ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.stroke();
    ctx.beginPath(); ctx.arc(0, 0, 2.5 / 36, 0, Math.PI * 2); ctx.fillStyle = "#80bdb6"; ctx.fill();
    ctx.beginPath(); ctx.arc(...frame.tip, 8 / 36, 0, Math.PI * 2); ctx.fillStyle = "#4cdbca22"; ctx.fill();
    ctx.beginPath(); ctx.arc(...frame.tip, 3.6 / 36, 0, Math.PI * 2); ctx.fillStyle = "#b7fff1"; ctx.fill();
  }

  function resize() {
    const bounds = stage.getBoundingClientRect();
    width = bounds.width; height = bounds.height;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = Math.round(width * ratio); canvas.height = Math.round(height * ratio);
    draw(0);
    if (pcaReady && $("view").value === "pca") Plotly.Plots.resize($("pca"));
  }

  async function showPca() {
    if (!pcaReady) {
      await Plotly.newPlot($("pca"), data.pca.data, data.pca.layout,
        {responsive: true, displaylogo: false, scrollZoom: true});
      pcaReady = true;
    }
    await Plotly.restyle($("pca"), {visible: data.pca.data.map((_, i) => Math.floor(i / 3) === meetingIndex)});
    await Plotly.relayout($("pca"), {"title.text": `Conversation Tracker · ${meeting.id}`, "xaxis.autorange": true, "yaxis.autorange": true});
    Plotly.Plots.resize($("pca"));
  }

  $("play").addEventListener("click", () => {
    if (player.playing) player.pause();
    else {
      if (player.finished) camera = [0, 0];
      player.play();
    }
    lastTime = null; sync(); draw(0);
  });
  $("restart").addEventListener("click", () => {
    player.restart(); camera = [0, 0]; lastTime = null; sync(); draw(0);
  });
  $("speed").addEventListener("input", () => {
    player.setSpeed(+$("speed").value); $("speed-value").textContent = `${player.speed}×`;
  });
  for (const id of ["context", "turn"]) $(id).addEventListener("input", () => {
    $(`${id}-value`).textContent = $(id).value + (id === "turn" ? "°" : "");
    rebuild(); lastTime = null;
  });
  $("lookahead").addEventListener("change", () => { rebuild(); lastTime = null; });
  $("timeline").addEventListener("input", () => {
    player.seek(+$("timeline").value - 1); camera = M.frame(geometry, player.progress).tip.slice();
    lastTime = null; sync(); draw(0);
  });
  $("meeting").addEventListener("change", () => {
    meetingIndex = data.meetings.findIndex(m => m.id === $("meeting").value);
    meeting = data.meetings[meetingIndex];
    rebuild();
    if ($("view").value !== "pca") player.play();
    else showPca().catch(showError);
    lastTime = null; sync();
  });
  $("view").addEventListener("change", () => {
    player.pause(); lastTime = null;
    const pca = $("view").value === "pca";
    canvas.hidden = pca; $("thread-label").hidden = pca; $("pca").hidden = !pca;
    if (pca) showPca().catch(showError);
    else rebuild(true);
    sync();
  });
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) { player.pause(); lastTime = null; sync(); }
  });
  function showError(error) {
    player.pause(); $("status").textContent = "Could not display the view";
    $("sentence").textContent = error.message; console.error(error);
  }

  rebuild(); player.play(); sync();
  new ResizeObserver(resize).observe(stage);
  function animate(now) {
    const seconds = lastTime === null ? 0 : (now - lastTime) / 1000;
    lastTime = now;
    player.tick(seconds); sync(); draw(seconds);
    requestAnimationFrame(animate);
  }
  requestAnimationFrame(animate);
})();
