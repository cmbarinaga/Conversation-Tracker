/* Shared browser/Node calculations. Passage inference is performed by Python. */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.ThreadMath = api;
})(typeof globalThis === "object" ? globalThis : this, function () {
  "use strict";
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const mix = (a, b, t) => [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t];

  function alignments(embeddings, k = 5) {
    const values = Array(embeddings.length).fill(null);
    for (let i = 1; i < embeddings.length; i++) {
      const context = new Float64Array(embeddings[i].length);
      for (let lag = 1; lag <= Math.min(k, i); lag++) {
        const weight = 0.8 ** (lag - 1), previous = embeddings[i - lag];
        for (let j = 0; j < context.length; j++) context[j] += weight * previous[j];
      }
      let norm2 = 0, dot = 0, sentenceNorm2 = 0;
      for (let j = 0; j < context.length; j++) {
        norm2 += context[j] ** 2;
        dot += context[j] * embeddings[i][j];
        sentenceNorm2 += embeddings[i][j] ** 2;
      }
      if (norm2 > 1e-16 && sentenceNorm2 > 1e-16) {
        values[i] = clamp(dot / Math.sqrt(norm2 * sentenceNorm2), -1, 1);
      }
    }
    return values;
  }

  function turnSign(heading, difference) {
    const norm = Math.hypot(...difference);
    if (norm < 1e-8) return 0;
    const cross = (heading[0] * difference[1] - heading[1] * difference[0]) / norm;
    return Math.abs(cross) < 1e-6 ? 0 : Math.sign(cross);
  }

  function bezier(curve, t) {
    const u = 1 - t;
    return [0, 1].map(j => u ** 3 * curve.p0[j] + 3 * u ** 2 * t * curve.c1[j]
      + 3 * u * t ** 2 * curve.c2[j] + t ** 3 * curve.p1[j]);
  }

  function partialCurve(curve, t) {
    const a = mix(curve.p0, curve.c1, t), b = mix(curve.c1, curve.c2, t);
    const c = mix(curve.c2, curve.p1, t), d = mix(a, b, t), e = mix(b, c, t);
    return {p0: curve.p0, c1: a, c2: d, p1: mix(d, e, t)};
  }

  function arcTable(curve) {
    const lengths = [0];
    let previous = curve.p0;
    for (let i = 1; i <= 32; i++) {
      const point = bezier(curve, i / 32);
      lengths.push(lengths[i - 1] + Math.hypot(point[0] - previous[0], point[1] - previous[1]));
      previous = point;
    }
    return lengths;
  }

  function arcParameter(curve, fraction) {
    fraction = clamp(fraction, 0, 1);
    const lengths = curve.lengths, target = fraction * lengths[32];
    for (let i = 1; i <= 32; i++) {
      if (lengths[i] >= target) {
        const span = lengths[i] - lengths[i - 1];
        return (i - 1 + (span > 0 ? (target - lengths[i - 1]) / span : 0)) / 32;
      }
    }
    return 1;
  }

  function trajectory(embeddings, coordinates, k = 5, maxTurn = 60) {
    if (embeddings.length !== coordinates.length || !embeddings.length) {
      throw new Error("Embedding and coordinate counts must match and be nonempty.");
    }
    if (!Number.isInteger(k) || k < 1 || k > 30 || !Number.isFinite(maxTurn) || maxTurn < 0 || maxTurn > 120) {
      throw new Error("Invalid context window or maximum turn.");
    }
    const alignment = alignments(embeddings, k);
    const divergence = alignment.map(a => a === null ? 0 : clamp(1 - a, 0, 1));
    return trace(coordinates, divergence, alignment, maxTurn);
  }

  function futureTrajectory(coordinates, profile, maxTurn = 60) {
    if (!coordinates.length || profile.pastAlignment.length !== coordinates.length ||
        profile.futureAlignment.length !== coordinates.length || !Number.isFinite(maxTurn) || maxTurn < 0 || maxTurn > 120) {
      throw new Error("Invalid passage alignment or turning parameters.");
    }
    // Departure and forward support stay separate; never average the scores.
    // Unsupported departures retain a small signal instead of erasing tangents.
    const rawDivergence = profile.pastAlignment.map((past, i) => {
      if (past === null) return 0;
      const future = profile.futureAlignment[i];
      const support = future === null ? 1 : 0.25 + 0.75 * clamp(future, 0, 1);
      return clamp(1 - past, 0, 1) * support;
    });
    const divergence = [0];
    for (let i = 1; i < rawDivergence.length; i++) divergence.push(0.5 * rawDivergence[i] + 0.5 * divergence[i - 1]);
    return {...trace(coordinates, divergence, profile.pastAlignment, maxTurn),
      futureAlignment: profile.futureAlignment, rawDivergence};
  }

  function trace(coordinates, divergence, alignment, maxTurn) {
    const positions = [[0, 0]], headings = [[1, 0]], curves = [];
    let angle = 0;
    for (let i = 1; i < coordinates.length; i++) {
      const previous = positions[i - 1], heading = headings[i - 1];
      const difference = [coordinates[i][0] - coordinates[i - 1][0], coordinates[i][1] - coordinates[i - 1][1]];
      angle += turnSign(heading, difference) * maxTurn * Math.PI / 180 * divergence[i];
      const nextHeading = [Math.cos(angle), Math.sin(angle)];
      const next = [previous[0] + nextHeading[0], previous[1] + nextHeading[1]];
      const curve = {
        p0: previous, p1: next,
        c1: [previous[0] + heading[0] / 3, previous[1] + heading[1] / 3],
        c2: [next[0] - nextHeading[0] / 3, next[1] - nextHeading[1] / 3],
      };
      curve.lengths = arcTable(curve);
      positions.push(next); headings.push(nextHeading); curves.push(curve);
    }
    return {alignment, divergence, positions, headings, curves};
  }

  function frame(geometry, progress) {
    progress = clamp(progress, 0, geometry.positions.length - 1);
    const completed = Math.floor(progress), fraction = progress - completed;
    const partial = fraction > 0 ? partialCurve(geometry.curves[completed], arcParameter(geometry.curves[completed], fraction)) : null;
    return {completed, partial, tip: partial ? partial.p1 : geometry.positions[completed],
      sentenceIndex: Math.max(0, Math.min(geometry.positions.length - 1, Math.ceil(progress - 1e-10)))};
  }

  function follow(camera, tip, seconds) {
    return mix(camera, tip, 1 - Math.exp(-Math.max(0, seconds) / 0.2));
  }

  return {alignments, turnSign, bezier, partialCurve, arcParameter, trajectory, futureTrajectory, frame, follow};
});
