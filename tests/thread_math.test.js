const {test} = require('node:test');
const assert = require('node:assert/strict');
const M = require('../web/thread_math.js');
const {Playback, BufferedPlayback} = require('../web/playback.js');
const close = (a, b, eps = 1e-10) => assert.ok(Math.abs(a - b) < eps, `${a} != ${b}`);
const vectorClose = (a, b) => a.forEach((x, i) => close(x, b[i]));
const embeddings = [[1, 0], [0, 1], [-1, 0], [0, -1], [1, 0]];
const coordinates = [[0, 0], [0, 1], [1, 2], [2, 1], [1, 0]];

test('alignment uses only available past context with exponential recency weights', () => {
  const actual = M.alignments([[1, 0], [0, 1], [1, 0]], 5);
  assert.equal(actual[0], null); close(actual[1], 0);
  close(actual[2], 0.8 / Math.hypot(0.8, 1));
  close(M.alignments([[1, 0], [0, 1], [1, 0]], 1)[2], 0);
});

test('identical and opposite embeddings, clipped divergence, and zero context', () => {
  close(M.alignments([[1, 0], [1, 0]], 5)[1], 1);
  close(M.alignments([[1, 0], [-1, 0]], 5)[1], -1);
  assert.equal(M.alignments([[0, 0], [1, 0]], 5)[1], null);
  // 0.8*[1,0] + [-0.8,0] cancels: undefined context must stay straight.
  const cancelled = M.trajectory([[1, 0], [-0.8, 0], [0, 1]], [[0, 0], [0, 1], [1, 1]]);
  assert.equal(cancelled.alignment[2], null);
  vectorClose(cancelled.headings[2], cancelled.headings[1]);
  const opposite = M.trajectory([[1, 0], [-1, 0]], [[0, 0], [0, 1]]);
  close(opposite.divergence[1], 1);
});

test('cross product chooses left/right and near-zero or collinear differences stay straight', () => {
  assert.equal(M.turnSign([1, 0], [0, 1]), 1);
  assert.equal(M.turnSign([1, 0], [0, -1]), -1);
  assert.equal(M.turnSign([1, 0], [1, 0]), 0);
  assert.equal(M.turnSign([1, 0], [-1, 0]), 0);
  assert.equal(M.turnSign([1, 0], [0, 1e-10]), 0);
  assert.equal(M.turnSign([1, 0], [1, 1e-8]), 0);
});

test('turn magnitude, unit endpoint steps, and zero max-turn follow the specified recurrence', () => {
  const geometry = M.trajectory(embeddings, coordinates, 5, 60);
  close(Math.atan2(...geometry.headings[1].slice().reverse()), Math.PI / 3);
  for (let i = 1; i < geometry.positions.length; i++) {
    close(Math.hypot(...geometry.positions[i].map((x, j) => x - geometry.positions[i - 1][j])), 1);
  }
  const straight = M.trajectory(embeddings, coordinates, 5, 0);
  straight.positions.forEach((point, i) => vectorClose(point, [i, 0]));
  const repeated = M.trajectory([[1, 0], [0, 1]], [[2, 2], [2, 2]]);
  vectorClose(repeated.positions[1], [1, 0]);
});

test('later sentences cannot move the prefix of the existing thread', () => {
  const prefix = M.trajectory(embeddings.slice(0, 3), coordinates.slice(0, 3));
  const full = M.trajectory(embeddings, coordinates);
  assert.deepEqual(full.positions.slice(0, 3), prefix.positions);
  assert.deepEqual(full.curves.slice(0, 2), prefix.curves);
});

test('Bezier endpoints, adjoining tangent continuity, and arc-length partial tip', () => {
  const geometry = M.trajectory(embeddings, coordinates);
  geometry.curves.forEach((curve, i) => {
    vectorClose(M.bezier(curve, 0), curve.p0);
    vectorClose(M.bezier(curve, 1), curve.p1);
    assert.equal(curve.lengths.length, 33);
    close(M.arcParameter(curve, 0), 0); close(M.arcParameter(curve, 1), 1);
    if (i > 0) {
      const previous = geometry.curves[i - 1];
      vectorClose(previous.p1.map((v, j) => v - previous.c2[j]), curve.c1.map((v, j) => v - curve.p0[j]));
    }
    for (const f of [0.01, 0.2, 0.5, 0.99]) {
      const t = M.arcParameter(curve, f), partial = M.partialCurve(curve, t);
      vectorClose(partial.p1, M.bezier(curve, t));
      vectorClose(M.frame(geometry, i + f).tip, partial.p1);
      for (const u of [0, 0.25, 0.75, 1]) vectorClose(M.bezier(partial, u), M.bezier(curve, u * t));
    }
  });
});

test('fractional scrubbing reveals the right sentence and completed curves', () => {
  const g = M.trajectory(embeddings, coordinates);
  assert.equal(M.frame(g, 0).sentenceIndex, 0);
  assert.equal(M.frame(g, 1).sentenceIndex, 1);
  const middle = M.frame(g, 1.5);
  assert.equal(middle.completed, 1); assert.equal(middle.sentenceIndex, 2);
  assert.ok(middle.partial);
  vectorClose(M.frame(g, 4).tip, g.positions[4]);
  assert.equal(M.frame(g, 4).partial, null);
});

test('camera follow is independent of frame rate', () => {
  const tip = [3, -2];
  let camera = [0, 0];
  for (let i = 0; i < 60; i++) camera = M.follow(camera, tip, 1 / 60);
  vectorClose(camera, M.follow([0, 0], tip, 1));
});

test('playback pause, resume, speed, seek, restart, completion, and replay', () => {
  const p = new Playback(5);
  p.tick(0.5); close(p.progress, 0.5);
  p.pause(); p.tick(10); close(p.progress, 0.5);
  p.play(); p.tick(0.25); close(p.progress, 0.75);
  p.setSpeed(2); p.tick(0.25); close(p.progress, 1.25);
  p.seek(2.75); assert.equal(p.playing, false); close(p.progress, 2.75);
  p.play(); p.tick(100); close(p.progress, 4); assert.equal(p.playing, false);
  p.play(); close(p.progress, 0); assert.equal(p.playing, true);
  p.seek(2); p.restart(); close(p.progress, 0); assert.equal(p.playing, true);
  p.tick(-1); p.tick(NaN); close(p.progress, 0);
  p.seek(-10); close(p.progress, 0); p.seek(10); close(p.progress, 4);
  p.setSpeed(0); close(p.speed, 0.25); p.setSpeed(10); close(p.speed, 4);
});

test('single sentence and invalid geometry parameters', () => {
  const g = M.trajectory([[1, 0]], [[1, 2]]);
  assert.equal(g.curves.length, 0); vectorClose(M.frame(g, 0).tip, [0, 0]);
  const p = new Playback(1); assert.equal(p.playing, false); p.play(); assert.equal(p.playing, false);
  assert.throws(() => M.trajectory([], []));
  assert.throws(() => M.trajectory(embeddings, coordinates, 0));
  assert.throws(() => M.trajectory(embeddings, coordinates, 5, 121));
});

test('future support distinguishes sustained departure from isolated deviation without score averaging', () => {
  const coords = [[0, 0], [0, 1], [1, 2], [2, 3]];
  const strong = M.futureTrajectory(coords, {pastAlignment:[null,0,0,0], futureAlignment:[1,1,1,null]});
  const isolated = M.futureTrajectory(coords, {pastAlignment:[null,0,0,0], futureAlignment:[0,0,0,null]});
  close(strong.rawDivergence[1], 1); close(isolated.rawDivergence[1], 0.25);
  close(strong.divergence[1], 0.5); close(strong.divergence[2], 0.75);
  close(strong.rawDivergence[3], 1); // End of input: use past alone.
  const ending = M.futureTrajectory(coords, {pastAlignment:[null,1,1,1], futureAlignment:[0,0,0,null]});
  ending.positions.forEach((point, i) => vectorClose(point, [i, 0]));
});

test('future-aware geometry is prefix stable after context finalization', () => {
  const p = {pastAlignment:[null,0.4,0.5,0.1,0.2], futureAlignment:[0.8,0.6,0.2,0.5,null]};
  const full = M.futureTrajectory(coordinates, p);
  const prefix = M.futureTrajectory(coordinates.slice(0,3), {
    pastAlignment:p.pastAlignment.slice(0,3), futureAlignment:p.futureAlignment.slice(0,3),
  });
  assert.deepEqual(full.curves.slice(0,2), prefix.curves);
});

test('buffered playback waits for required future utterances, freezes, scrubs, and flushes at EOF', () => {
  const p = new BufferedPlayback([2,3,4,4,4]);
  assert.equal(p.buffering,true); assert.equal(p.received,1);
  p.tick(2.5); close(p.progress,0); assert.equal(p.received,3);
  p.tick(0.75); close(p.progress,0.25); assert.equal(p.buffering,false);
  p.pause(); p.tick(10); close(p.clock,3.25); close(p.progress,0.25);
  p.play(); p.tick(1); close(p.progress,1.25);
  p.seek(2.5); close(p.progress,2.5); close(p.clock,5.5); assert.equal(p.playing,false);
  p.play(); p.tick(100); close(p.progress,4); assert.equal(p.playing,false); assert.equal(p.received,5);
  p.restart(); close(p.clock,0); close(p.progress,0); assert.equal(p.buffering,true);
});

test('larger lookahead delays drawing, with deterministic results across frame rates', () => {
  const short = new BufferedPlayback([1,2,3,4,4]);
  const long = new BufferedPlayback([4,4,4,4,4]);
  short.tick(3.5); long.tick(3.5);
  assert.ok(short.progress>long.progress);
  const replay = new BufferedPlayback([4,4,4,4,4]);
  for(let i=0;i<350;i++) replay.tick(0.01);
  close(long.progress,replay.progress);
  long.tick(2); for(let i=0;i<200;i++) replay.tick(0.01);
  close(long.progress,replay.progress,1e-8);
});
