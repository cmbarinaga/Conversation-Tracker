(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.ThreadPlayback = api;
})(typeof globalThis === "object" ? globalThis : this, function () {
  "use strict";
  class Playback {
    constructor(count) {
      this.count = count;
      this.progress = 0;
      this.speed = 1;
      this.playing = count > 1;
    }
    get finished() { return this.progress >= this.count - 1; }
    pause() { this.playing = false; }
    play() {
      if (this.finished) this.progress = 0;
      this.playing = this.count > 1;
    }
    restart() { this.progress = 0; this.playing = this.count > 1; }
    seek(progress) {
      if (!Number.isFinite(progress)) return;
      this.progress = Math.max(0, Math.min(this.count - 1, progress));
      this.pause();
    }
    setSpeed(speed) {
      if (Number.isFinite(speed)) this.speed = Math.max(0.25, Math.min(4, speed));
    }
    tick(seconds) {
      if (this.playing && Number.isFinite(seconds) && seconds > 0) {
        this.progress = Math.min(this.count - 1, this.progress + seconds * this.speed);
        if (this.finished) this.pause();
      }
    }
  }
  class BufferedPlayback extends Playback {
    constructor(readyAt) {
      super(readyAt.length);
      this.readyAt = readyAt.slice();
      this.starts = [0]; this.ends = [0]; this.clock = 0;
      for (let i = 1; i < this.count; i++) {
        if (!Number.isFinite(readyAt[i]) || readyAt[i] < i || readyAt[i] >= this.count) throw new Error("Invalid lookahead release.");
        this.starts.push(Math.max(this.ends[i - 1], readyAt[i]));
        this.ends.push(this.starts[i] + 1);
      }
    }
    get received() { return Math.min(this.count, Math.floor(this.clock) + 1); }
    get buffering() {
      const next = Math.floor(this.progress) + 1;
      return next < this.count && this.clock < this.starts[next];
    }
    seek(progress) {
      if (!Number.isFinite(progress)) return;
      super.seek(progress);
      const completed = Math.floor(this.progress), fraction = this.progress - completed;
      this.clock = fraction > 0 ? this.starts[completed + 1] + fraction : this.ends[completed];
    }
    restart() { this.clock = 0; super.restart(); }
    play() { if (this.finished) this.clock = 0; super.play(); }
    tick(seconds) {
      if (!this.playing || !Number.isFinite(seconds) || seconds <= 0) return;
      this.clock = Math.min(this.ends[this.count - 1], this.clock + seconds * this.speed);
      // Binary search the completed segments; pure clock evaluation makes replay
      // independent of animation frame rate and safe across large time steps.
      let low = 0, high = this.count - 1;
      while (low < high) {
        const middle = Math.ceil((low + high) / 2);
        if (this.ends[middle] <= this.clock) low = middle;
        else high = middle - 1;
      }
      this.progress = low;
      if (low + 1 < this.count && this.clock > this.starts[low + 1]) {
        this.progress += Math.min(1, this.clock - this.starts[low + 1]);
      }
      if (this.finished) this.pause();
    }
  }
  return {Playback, BufferedPlayback};
});
