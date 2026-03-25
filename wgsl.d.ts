// Module declaration for WGSL shader files imported as raw strings
declare module '*.wgsl' {
  const source: string;
  export default source;
}

// Vite ?raw import suffix
declare module '*.wgsl?raw' {
  const source: string;
  export default source;
}

// MediaStreamTrackProcessor (WebCodecs — not yet in all TS libs)
declare class MediaStreamTrackProcessor {
  constructor(options: { track: MediaStreamTrack });
  readonly readable: ReadableStream<VideoFrame>;
}
