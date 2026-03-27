/**
 * SMPLLoaderUI — drag-drop / browse UI for loading SMPL .pkl model files.
 *
 * Parses the Python pickle (protocol 2) format with numpy array reconstruction,
 * persists the parsed buffers in IndexedDB (key: smpl_model_v1), and surfaces
 * a simple callback-based API so the rest of the pipeline can consume the data.
 */

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

export interface SMPLModelData {
  /** [6890, 3] mean template vertices (row-major) */
  vertices: Float32Array;
  /** [13776, 3] triangle faces */
  faces: Uint32Array;
  /** [6890, 3, 10] shape blend shapes */
  shapedirs: Float32Array;
  /** [6890, 3, 207] pose blend shapes */
  posedirs: Float32Array;
  /** [24, 6890] joint regressor */
  J_regressor: Float32Array;
  /** [2, 24] kinematic tree – row 0 = parent indices */
  kintree_table: Int32Array;
  /** [6890, 24] skinning weights */
  weights: Float32Array;
}

export interface SMPLLoaderOptions {
  container: HTMLElement;
  onLoad?: (data: SMPLModelData) => void;
  onError?: (err: Error) => void;
}

// ---------------------------------------------------------------------------
// .smpl.bin fast binary parser
// Format: magic "SMPL" + uint32 version + uint32 numArrays + [arrays...]
// Each array: uint32 nameLen + utf8 name + uint8 dtype(0=f32,1=i32,2=u32)
//             + uint32 rank + uint32[rank] shape + raw data bytes
// ---------------------------------------------------------------------------

function parseSMPLBin(buf: ArrayBuffer): SMPLModelData {
  const view = new DataView(buf);
  let pos = 0;

  const magic = String.fromCharCode(
    view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3));
  if (magic !== 'SMPL') throw new Error('Not a .smpl.bin file');
  pos = 4;
  const version = view.getUint32(pos, true); pos += 4;
  if (version !== 1) throw new Error(`Unsupported .smpl.bin version ${version}`);
  const numArrays = view.getUint32(pos, true); pos += 4;

  const arrays: Record<string, Float32Array | Int32Array | Uint32Array> = {};
  const dec = new TextDecoder();

  for (let i = 0; i < numArrays; i++) {
    const nameLen = view.getUint32(pos, true); pos += 4;
    const name = dec.decode(new Uint8Array(buf, pos, nameLen)); pos += nameLen;
    const dtype = view.getUint8(pos); pos += 1;
    const rank = view.getUint32(pos, true); pos += 4;
    let size = 1;
    for (let r = 0; r < rank; r++) {
      size *= view.getUint32(pos, true); pos += 4;
    }
    const byteLen = size * (dtype === 1 ? 4 : dtype === 2 ? 4 : 4);
    // Use slice to get aligned buffer for typed array construction
    const slice = buf.slice(pos, pos + byteLen);
    arrays[name] = dtype === 1
      ? new Int32Array(slice)
      : dtype === 2
        ? new Uint32Array(slice)
        : new Float32Array(slice);
    pos += byteLen;
  }

  function req<T>(k: string): T {
    if (!(k in arrays)) throw new Error(`Missing SMPL array: ${k}`);
    return arrays[k] as unknown as T;
  }
  return {
    vertices:      req<Float32Array>('v_template'),
    faces:         req<Uint32Array>('f'),
    shapedirs:     req<Float32Array>('shapedirs'),
    posedirs:      req<Float32Array>('posedirs'),
    J_regressor:   req<Float32Array>('J_regressor'),
    kintree_table: req<Int32Array>('kintree_table'),
    weights:       req<Float32Array>('weights'),
  };
}

// ---------------------------------------------------------------------------
// IndexedDB persistence helpers
// ---------------------------------------------------------------------------

const IDB_DB_NAME = 'smpl-model-store';
const IDB_DB_VERSION = 1;
const IDB_STORE = 'models';
const IDB_KEY = 'smpl_model_v1';

function idbOpen(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(IDB_DB_NAME, IDB_DB_VERSION);
    req.onupgradeneeded = (e) => {
      const db = (e.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(IDB_STORE)) {
        db.createObjectStore(IDB_STORE, { keyPath: 'key' });
      }
    };
    req.onsuccess = (e) => resolve((e.target as IDBOpenDBRequest).result);
    req.onerror = (e) => reject((e.target as IDBOpenDBRequest).error);
  });
}

async function idbSave(data: SMPLModelData): Promise<void> {
  const db = await idbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readwrite');
    const store = tx.objectStore(IDB_STORE);
    const req = store.put({
      key: IDB_KEY,
      vertices: data.vertices,
      faces: data.faces,
      shapedirs: data.shapedirs,
      posedirs: data.posedirs,
      J_regressor: data.J_regressor,
      kintree_table: data.kintree_table,
      weights: data.weights,
    });
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

async function idbLoad(): Promise<SMPLModelData | null> {
  try {
    const db = await idbOpen();
    return new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readonly');
      const store = tx.objectStore(IDB_STORE);
      const req = store.get(IDB_KEY);
      req.onsuccess = () => {
        const row = req.result;
        if (!row) { resolve(null); return; }
        resolve({
          vertices: row.vertices,
          faces: row.faces,
          shapedirs: row.shapedirs,
          posedirs: row.posedirs,
          J_regressor: row.J_regressor,
          kintree_table: row.kintree_table,
          weights: row.weights,
        });
      };
      req.onerror = () => reject(req.error);
    });
  } catch {
    return null;
  }
}

async function idbClear(): Promise<void> {
  const db = await idbOpen();
  return new Promise((resolve, reject) => {
    const tx = db.transaction(IDB_STORE, 'readwrite');
    const req = tx.objectStore(IDB_STORE).delete(IDB_KEY);
    req.onsuccess = () => resolve();
    req.onerror = () => reject(req.error);
  });
}

// ---------------------------------------------------------------------------
// Minimal Python pickle (protocol 2) parser
// ---------------------------------------------------------------------------
// Handles the subset of opcodes produced by SMPL .pkl files:
//   PROTO, FRAME, MARK, GLOBAL, REDUCE, BUILD, DICT, SETITEMS, SETITEM,
//   TUPLE, TUPLE1, TUPLE2, TUPLE3, EMPTY_TUPLE, LIST, APPENDS, APPEND,
//   SHORT_BINSTRING, BINSTRING, BINUNICODE, BINUNICODE8,
//   BININT1, BININT2, BININT, LONG_BINPUT, BINPUT, LONG_BINGET, BINGET,
//   NEWFALSE, NEWTRUE, NONE, POP, STOP, MARK, EMPTY_DICT, EMPTY_LIST,
//   BINFLOAT, LONG1, NEWOBJ.
// Numpy arrays are reconstructed from their raw byte data inline.

type PickleObj = unknown;

interface NpyDtype {
  kind: 'f' | 'i' | 'u';  // float, signed int, unsigned int
  itemsize: number;        // bytes per element
  littleEndian: boolean;
}

function parseDtype(desc: string): NpyDtype {
  // e.g. '<f4', '>i4', '|u1', '=f8'
  const endian = desc[0] ?? '<';
  const kind = (desc[1] ?? 'f') as 'f' | 'i' | 'u';
  const itemsize = parseInt(desc[2] ?? '4', 10);
  return { kind, itemsize, littleEndian: endian !== '>' };
}

function reconstructNdarray(
  dtype: NpyDtype,
  shape: number[],
  rawBytes: Uint8Array,
): Float32Array | Int32Array | Uint32Array {
  const total = shape.reduce((a, b) => a * b, 1);
  const view = new DataView(rawBytes.buffer, rawBytes.byteOffset, rawBytes.byteLength);

  if (dtype.kind === 'f' && dtype.itemsize === 4) {
    const arr = new Float32Array(total);
    for (let i = 0; i < total; i++) arr[i] = view.getFloat32(i * 4, dtype.littleEndian);
    return arr;
  }
  if (dtype.kind === 'f' && dtype.itemsize === 8) {
    // Downcast float64 → float32
    const arr = new Float32Array(total);
    for (let i = 0; i < total; i++) arr[i] = view.getFloat64(i * 8, dtype.littleEndian);
    return arr;
  }
  if ((dtype.kind === 'i' || dtype.kind === 'u') && dtype.itemsize === 4) {
    const arr = dtype.kind === 'i' ? new Int32Array(total) : new Uint32Array(total);
    for (let i = 0; i < total; i++) {
      (arr as Int32Array)[i] = dtype.kind === 'i'
        ? view.getInt32(i * 4, dtype.littleEndian)
        : view.getUint32(i * 4, dtype.littleEndian);
    }
    return arr as Int32Array | Uint32Array;
  }
  if ((dtype.kind === 'i' || dtype.kind === 'u') && dtype.itemsize === 8) {
    // int64 / uint64 → int32 (joint index values fit)
    const arr = new Int32Array(total);
    for (let i = 0; i < total; i++) {
      const lo = view.getUint32(i * 8, dtype.littleEndian);
      arr[i] = lo | 0;
    }
    return arr;
  }
  if ((dtype.kind === 'i' || dtype.kind === 'u') && dtype.itemsize === 2) {
    const arr = new Int32Array(total);
    for (let i = 0; i < total; i++) {
      arr[i] = dtype.kind === 'i'
        ? view.getInt16(i * 2, dtype.littleEndian)
        : view.getUint16(i * 2, dtype.littleEndian);
    }
    return arr;
  }
  // Fallback: int/uint8 → Int32Array
  const arr = new Int32Array(total);
  for (let i = 0; i < total; i++) arr[i] = rawBytes[i] ?? 0;
  return arr;
}

class PickleParser {
  private buf: Uint8Array;
  private pos = 0;
  private stack: PickleObj[] = [];
  private memo: Map<number, PickleObj> = new Map();
  private marks: number[] = [];

  constructor(buf: Uint8Array) {
    this.buf = buf;
  }

  private read1(): number { return this.buf[this.pos++] ?? 0; }

  private readBytes(n: number): Uint8Array {
    const slice = this.buf.slice(this.pos, this.pos + n);
    this.pos += n;
    return slice;
  }

  private readLine(): string {
    let end = this.pos;
    while (end < this.buf.length && this.buf[end] !== 0x0a) end++;
    const line = new TextDecoder().decode(this.buf.slice(this.pos, end));
    this.pos = end + 1;
    return line;
  }

  private readUint16LE(): number {
    const v = (this.buf[this.pos] ?? 0) | ((this.buf[this.pos + 1] ?? 0) << 8);
    this.pos += 2;
    return v;
  }

  private readUint32LE(): number {
    const v = ((this.buf[this.pos] ?? 0)
      | ((this.buf[this.pos + 1] ?? 0) << 8)
      | ((this.buf[this.pos + 2] ?? 0) << 16)
      | ((this.buf[this.pos + 3] ?? 0) << 24)) >>> 0;
    this.pos += 4;
    return v;
  }

  private readInt32LE(): number {
    const v = (this.buf[this.pos] ?? 0)
      | ((this.buf[this.pos + 1] ?? 0) << 8)
      | ((this.buf[this.pos + 2] ?? 0) << 16)
      | ((this.buf[this.pos + 3] ?? 0) << 24);
    this.pos += 4;
    return v;
  }

  private readFloat64BE(): number {
    const view = new DataView(this.buf.buffer, this.buf.byteOffset + this.pos, 8);
    this.pos += 8;
    return view.getFloat64(0, false); // big-endian
  }

  private push(v: PickleObj) { this.stack.push(v); }
  private pop(): PickleObj { return this.stack.pop() ?? null; }
  private top(): PickleObj { return this.stack[this.stack.length - 1] ?? null; }

  private popMark(): PickleObj[] {
    const markIdx = this.marks.pop() ?? 0;
    return this.stack.splice(markIdx);
  }


  parse(): PickleObj {
    while (this.pos < this.buf.length) {
      const opcode = this.read1();

      switch (opcode) {
        case 0x80: { // PROTO
          this.read1(); // protocol version
          break;
        }
        case 0x28: { // MARK '('
          this.marks.push(this.stack.length);
          break;
        }
        case 0x2e: { // STOP '.'
          return this.pop();
        }
        case 0x30: { // POP '0'
          this.pop();
          break;
        }
        case 0x32: { // DUP '2'
          this.push(this.top());
          break;
        }
        case 0x46: { // NONE 'N'
          this.push(null);
          break;
        }
        case 0x54: { // NEWTRUE 'T' (actually opcode 0x88)
          this.push(true);
          break;
        }
        case 0x55: { // NEWFALSE 'U' etc — actually handled below
          // SHORT_BINSTRING
          const slen = this.read1();
          this.push(this.readBytes(slen));
          break;
        }
        case 0x58: { // BINUNICODE 'X'
          const ulen = this.readUint32LE();
          this.push(new TextDecoder().decode(this.readBytes(ulen)));
          break;
        }
        case 0x4b: { // BININT1 'K'
          this.push(this.read1());
          break;
        }
        case 0x4d: { // BININT2 'M'
          this.push(this.readUint16LE());
          break;
        }
        case 0x4a: { // BININT 'J'
          this.push(this.readInt32LE());
          break;
        }
        case 0x4c: { // LONG 'L'
          const longStr = this.readLine();
          this.push(parseInt(longStr.replace('L', ''), 10));
          break;
        }
        case 0x8a: { // LONG1
          const nBytes = this.read1();
          let val = 0;
          for (let i = 0; i < nBytes; i++) val |= (this.read1() << (8 * i));
          this.push(val);
          break;
        }
        case 0x47: { // BINFLOAT 'G'
          this.push(this.readFloat64BE());
          break;
        }
        case 0x53: { // STRING 'S'
          const strLine = this.readLine();
          // Strip surrounding quotes
          this.push(strLine.replace(/^['"]|['"]$/g, ''));
          break;
        }
        case 0x54: { // SHORT_BINSTRING (0x55 handled above, 0x54 = 'T' NEWTRUE)
          this.push(true);
          break;
        }
        case 0x88: { // NEWTRUE
          this.push(true);
          break;
        }
        case 0x89: { // NEWFALSE
          this.push(false);
          break;
        }
        case 0x43: { // SHORT_BINBYTES 'C'
          const blen = this.read1();
          this.push(this.readBytes(blen));
          break;
        }
        case 0x42: { // BINBYTES 'B'
          const bblen = this.readUint32LE();
          this.push(this.readBytes(bblen));
          break;
        }
        case 0x63: { // GLOBAL 'c'
          const modName = this.readLine();
          const attrName = this.readLine();
          this.push(`${modName}.${attrName}`);
          break;
        }
        case 0x52: { // REDUCE 'R'
          const args = this.pop() as PickleObj[];
          const func = this.pop() as string;
          // Handle numpy reconstruct → sentinel object
          if (typeof func === 'string' && (
            func.includes('_reconstruct') || func === 'numpy.ndarray'
          )) {
            this.push({ __type__: 'ndarray_pending', func, args });
          } else if (typeof func === 'string' && func.includes('dtype')) {
            // numpy.dtype constructor
            const dtypeStr = Array.isArray(args) ? (args[0] as string) : String(args);
            this.push({ __type__: 'dtype', desc: dtypeStr });
          } else {
            // Generic: push a placeholder tuple with name
            this.push({ __type__: 'call', func, args });
          }
          break;
        }
        case 0x62: { // BUILD 'b'
          const state = this.pop();
          const obj = this.top();
          if (obj !== null && typeof obj === 'object' && (obj as Record<string, unknown>).__type__ === 'ndarray_pending') {
            // State for ndarray is a tuple:
            //   (version, shape_tuple, dtype_obj, isF_order, raw_bytes[, ??])
            const stateArr = state as PickleObj[];
            if (Array.isArray(stateArr) && stateArr.length >= 5) {
              const shape = (stateArr[1] as number[]);
              const dtypeObj = stateArr[2];
              const rawBytes = stateArr[4] as Uint8Array;

              let dtypeDesc = '<f4';
              if (dtypeObj !== null && typeof dtypeObj === 'object') {
                const dtRec = dtypeObj as Record<string, unknown>;
                if (dtRec.__type__ === 'dtype') {
                  dtypeDesc = dtRec.desc as string;
                } else if (typeof dtRec.desc === 'string') {
                  dtypeDesc = dtRec.desc;
                } else if (typeof dtRec.str === 'string') {
                  dtypeDesc = dtRec.str;
                }
              } else if (typeof dtypeObj === 'string') {
                dtypeDesc = dtypeObj;
              }

              const dtype = parseDtype(dtypeDesc);
              const reconstructed = reconstructNdarray(dtype, shape, rawBytes);

              // Replace top of stack with the actual array
              this.stack[this.stack.length - 1] = reconstructed;
            }
          } else if (obj !== null && typeof obj === 'object') {
            // Merge state into obj (dict __setstate__ pattern)
            const objRec = obj as Record<string, unknown>;
            if (typeof state === 'object' && state !== null && !ArrayBuffer.isView(state)) {
              if (Array.isArray(state)) {
                // dtype state: (endian, subdtype, names, formats, offsets, itemsize, aligned)
                // stash on obj for later use
                if ((obj as Record<string, unknown>).__type__ === 'dtype') {
                  const dtArr = state as PickleObj[];
                  if (typeof dtArr[3] === 'string') {
                    (obj as Record<string, unknown>).str = dtArr[3];
                    (obj as Record<string, unknown>).desc = dtArr[3];
                  }
                }
              } else {
                Object.assign(objRec, state);
              }
            }
          }
          break;
        }
        case 0x7d: { // EMPTY_DICT '}'
          this.push({});
          break;
        }
        case 0x64: { // DICT 'd'
          const items = this.popMark();
          const dict: Record<string, PickleObj> = {};
          for (let i = 0; i < items.length - 1; i += 2) {
            dict[String(items[i])] = items[i + 1] ?? null;
          }
          this.push(dict);
          break;
        }
        case 0x75: { // SETITEMS 'u'
          const pairs = this.popMark();
          const target = this.top() as Record<string, PickleObj>;
          for (let i = 0; i < pairs.length - 1; i += 2) {
            target[String(pairs[i])] = pairs[i + 1] ?? null;
          }
          break;
        }
        case 0x73: { // SETITEM 's'
          const val = this.pop();
          const key = this.pop();
          const tgt = this.top() as Record<string, PickleObj>;
          if (tgt && typeof tgt === 'object') tgt[String(key)] = val;
          break;
        }
        case 0x5d: { // EMPTY_LIST ']'
          this.push([]);
          break;
        }
        case 0x6c: { // LIST 'l'
          this.push(this.popMark());
          break;
        }
        case 0x61: { // APPEND 'a'
          const item = this.pop();
          const lst = this.top() as PickleObj[];
          if (Array.isArray(lst)) lst.push(item);
          break;
        }
        case 0x65: { // APPENDS 'e'
          const appendItems = this.popMark();
          const tgtList = this.top() as PickleObj[];
          if (Array.isArray(tgtList)) tgtList.push(...appendItems);
          break;
        }
        case 0x74: { // TUPLE 't'
          this.push(this.popMark());
          break;
        }
        case 0x85: { // TUPLE1
          this.push([this.pop()]);
          break;
        }
        case 0x86: { // TUPLE2
          const b = this.pop(); const a = this.pop();
          this.push([a, b]);
          break;
        }
        case 0x87: { // TUPLE3
          const c = this.pop(); const bb = this.pop(); const aa = this.pop();
          this.push([aa, bb, c]);
          break;
        }
        case 0x29: { // EMPTY_TUPLE ')'
          this.push([]);
          break;
        }
        case 0x71: { // BINPUT 'q'
          const putIdx = this.read1();
          this.memo.set(putIdx, this.top());
          break;
        }
        case 0x72: { // LONG_BINPUT 'r'
          const putIdx4 = this.readUint32LE();
          this.memo.set(putIdx4, this.top());
          break;
        }
        case 0x68: { // BINGET 'h'
          const getIdx = this.read1();
          this.push(this.memo.get(getIdx) ?? null);
          break;
        }
        case 0x6a: { // LONG_BINGET 'j'
          const getIdx4 = this.readUint32LE();
          this.push(this.memo.get(getIdx4) ?? null);
          break;
        }
        case 0x81: { // NEWOBJ
          const args2 = this.pop() as PickleObj[];
          const cls = this.pop() as string;
          if (typeof cls === 'string' && cls.includes('ndarray')) {
            this.push({ __type__: 'ndarray_pending', func: cls, args: args2 });
          } else if (typeof cls === 'string' && cls.includes('dtype')) {
            const dtStr = Array.isArray(args2) ? String(args2[0]) : '';
            this.push({ __type__: 'dtype', desc: dtStr });
          } else {
            this.push({ __type__: 'newobj', cls, args: args2 });
          }
          break;
        }
        case 0x82: { // EXT1 — skip
          this.read1();
          break;
        }
        case 0x83: { // EXT2 — skip
          this.readUint16LE();
          break;
        }
        case 0x84: { // EXT4 — skip
          this.readUint32LE();
          break;
        }
        case 0x69: { // INST — skip module + class + mark args
          this.readLine(); // module
          this.readLine(); // class
          this.popMark();
          this.push({});
          break;
        }
        case 0x6f: { // OBJ
          const objArgs = this.popMark();
          const objCls = this.stack.pop();
          this.push({ __type__: 'obj', cls: objCls, args: objArgs });
          break;
        }
        case 0x50: { // PERSID 'P'
          this.readLine();
          this.push(null);
          break;
        }
        case 0x51: { // BINPERSID 'Q'
          this.pop();
          this.push(null);
          break;
        }
        case 0x7e: { // FRAME (pickle 4) '~'
          // 8-byte frame length — just skip
          this.pos += 8;
          break;
        }
        case 0x8b: { // BYTEARRAY8
          const baLen = Number(
            (BigInt(this.buf[this.pos] ?? 0)
            | (BigInt(this.buf[this.pos + 1] ?? 0) << 8n)
            | (BigInt(this.buf[this.pos + 2] ?? 0) << 16n)
            | (BigInt(this.buf[this.pos + 3] ?? 0) << 24n)
            | (BigInt(this.buf[this.pos + 4] ?? 0) << 32n)
            | (BigInt(this.buf[this.pos + 5] ?? 0) << 40n)
            | (BigInt(this.buf[this.pos + 6] ?? 0) << 48n)
            | (BigInt(this.buf[this.pos + 7] ?? 0) << 56n))
          );
          this.pos += 8;
          this.push(this.readBytes(baLen));
          break;
        }
        case 0x8c: { // SHORT_BINUNICODE
          const suLen = this.read1();
          this.push(new TextDecoder().decode(this.readBytes(suLen)));
          break;
        }
        case 0x8d: { // BINUNICODE8
          // 8-byte length
          const buLen = Number(
            (BigInt(this.buf[this.pos] ?? 0)
            | (BigInt(this.buf[this.pos + 1] ?? 0) << 8n)
            | (BigInt(this.buf[this.pos + 2] ?? 0) << 16n)
            | (BigInt(this.buf[this.pos + 3] ?? 0) << 24n))
          );
          this.pos += 8;
          this.push(new TextDecoder().decode(this.readBytes(buLen)));
          break;
        }
        case 0x8e: { // BINBYTES8
          const bb8Len = Number(
            (BigInt(this.buf[this.pos] ?? 0)
            | (BigInt(this.buf[this.pos + 1] ?? 0) << 8n)
            | (BigInt(this.buf[this.pos + 2] ?? 0) << 16n)
            | (BigInt(this.buf[this.pos + 3] ?? 0) << 24n))
          );
          this.pos += 8;
          this.push(this.readBytes(bb8Len));
          break;
        }
        default: {
          // Unknown opcode — attempt recovery by skipping
          console.warn(`[PickleParser] Unknown opcode 0x${opcode.toString(16)} at pos=${this.pos - 1}`);
          break;
        }
      }
    }
    return this.pop();
  }
}

// ---------------------------------------------------------------------------
// SMPL dict → SMPLModelData extraction
// ---------------------------------------------------------------------------

function extractSMPLData(raw: unknown): SMPLModelData {
  if (!raw || typeof raw !== 'object') {
    throw new Error('Parsed pickle is not an object');
  }
  const d = raw as Record<string, unknown>;

  function requireArray(key: string): Float32Array | Int32Array | Uint32Array {
    const v = d[key];
    if (v instanceof Float32Array || v instanceof Int32Array || v instanceof Uint32Array) return v;
    throw new Error(`SMPL key "${key}" not found or not a typed array (got ${typeof v})`);
  }

  function toFloat32(a: Float32Array | Int32Array | Uint32Array): Float32Array {
    if (a instanceof Float32Array) return a;
    return new Float32Array(a);
  }

  function toInt32(a: Float32Array | Int32Array | Uint32Array): Int32Array {
    if (a instanceof Int32Array) return a;
    return new Int32Array(a);
  }

  function toUint32(a: Float32Array | Int32Array | Uint32Array): Uint32Array {
    if (a instanceof Uint32Array) return a;
    return new Uint32Array(a);
  }

  // SMPL .pkl keys: v_template, f, shapedirs, posedirs, J_regressor, kintree_table, weights
  const vertices = toFloat32(requireArray('v_template'));
  const faces = toUint32(requireArray('f'));
  const shapedirs = toFloat32(requireArray('shapedirs'));
  const posedirs = toFloat32(requireArray('posedirs'));
  const J_regressor = toFloat32(requireArray('J_regressor'));
  const kintree_table = toInt32(requireArray('kintree_table'));
  const weights = toFloat32(requireArray('weights'));

  // Sanity checks
  if (vertices.length !== 6890 * 3) {
    throw new Error(`v_template: expected ${6890 * 3} floats, got ${vertices.length}`);
  }
  if (faces.length !== 13776 * 3) {
    throw new Error(`f: expected ${13776 * 3} ints, got ${faces.length}`);
  }

  return { vertices, faces, shapedirs, posedirs, J_regressor, kintree_table, weights };
}

// ---------------------------------------------------------------------------
// SMPLLoaderUI class
// ---------------------------------------------------------------------------

export class SMPLLoaderUI {
  private opts: SMPLLoaderOptions;
  private model: SMPLModelData | null = null;
  private aborted = false;

  constructor(opts: SMPLLoaderOptions) {
    this.opts = opts;
    this.init();
  }

  private async init() {
    // Show a loading state immediately so the container isn't blank
    this.renderLoading('Checking for SMPL model…');

    // Try to hydrate from IndexedDB first
    try {
      const cached = await idbLoad();
      if (cached && !this.aborted) {
        this.model = cached;
        this.renderLoaded(cached, true);
        this.opts.onLoad?.(cached);
        return;
      }
    } catch {
      // IndexedDB unavailable — silently ignore
    }

    // Try auto-loading from pre-converted binary (fast, no pkl parsing needed)
    const BIN_PATH = '/smpl/smpl-neutral.smpl.bin';
    try {
      const resp = await fetch(BIN_PATH, { method: 'HEAD' });
      if (!resp.ok) {
        this.renderEmpty(); return;
      }
      if (this.aborted) return;
      this.renderLoading('Loading SMPL model (41 MB)…');
      const buf = await (await fetch(BIN_PATH)).arrayBuffer();
      if (this.aborted) return;
      this.renderLoading('Parsing SMPL arrays…');
      const smpl = parseSMPLBin(buf);
      if (!this.aborted) {
        this.model = smpl;
        this.renderLoaded(smpl, false);
        this.opts.onLoad?.(smpl);
        await idbSave(smpl).catch(() => {});
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      console.error('[SMPLLoaderUI] auto-load error:', e);
      // Show error in UI with fallback to drag-drop
      if (!this.aborted) {
        this.renderLoadError(msg);
      }
    }
  }

  private renderLoading(msg: string) {
    const c = this.opts.container;
    c.innerHTML = `
      <div style="padding:24px;color:#8b949e;font-size:13px;display:flex;align-items:center;gap:12px;">
        <div style="width:120px;height:6px;background:#21262d;border-radius:3px;overflow:hidden;">
          <div style="height:100%;background:#58a6ff;border-radius:3px;animation:smpl-pulse 1.2s ease-in-out infinite;width:60%;"></div>
        </div>
        <span>${msg}</span>
      </div>
      <style>@keyframes smpl-pulse{0%,100%{transform:translateX(-100%)}50%{transform:translateX(200%)}}</style>`;
  }

  private renderLoadError(msg: string) {
    const c = this.opts.container;
    c.innerHTML = `
      <div style="padding:16px;color:#f85149;font-size:13px;background:#21262d;border-radius:8px;max-width:420px;">
        <strong>SMPL auto-load failed:</strong><br/>
        <code style="font-size:11px;word-break:break-all;">${msg}</code><br/>
        <button id="smpl-fallback-btn" style="margin-top:10px;padding:6px 14px;background:#1f6feb;border:none;border-radius:4px;color:white;cursor:pointer;font-size:12px;">
          Upload manually instead
        </button>
      </div>`;
    c.querySelector('#smpl-fallback-btn')?.addEventListener('click', () => this.renderEmpty());
  }

  // ── DOM helpers ───────────────────────────────────────────────────────────

  private renderEmpty() {
    const c = this.opts.container;
    c.innerHTML = '';
    c.style.cssText = 'display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;';

    const zone = document.createElement('div');
    zone.style.cssText = `
      border: 2px dashed #444;
      border-radius: 10px;
      padding: 32px 40px;
      text-align: center;
      cursor: pointer;
      background: #0d0d14;
      transition: border-color 0.2s, background 0.2s;
      user-select: none;
      min-width: 300px;
    `;
    zone.innerHTML = `
      <div style="font-size:2.5rem;margin-bottom:8px;opacity:0.6">&#128220;</div>
      <div style="font-size:0.9rem;font-weight:600;color:#c0d0e0;margin-bottom:4px;">Upload SMPL model (.pkl)</div>
      <div style="font-size:0.75rem;color:#607080;margin-bottom:16px;">
        Register free at <a href="https://smpl.is.tue.mpg.de" target="_blank" rel="noopener"
          style="color:#7a9fc2;text-decoration:none;">smpl.is.tue.mpg.de</a>
        &mdash; download SMPL_NEUTRAL.pkl
      </div>
      <div style="display:flex;gap:8px;justify-content:center;align-items:center;">
        <div id="smpl-drop-hint" style="font-size:0.8rem;color:#507090;">
          Drag &amp; drop here &mdash; or
        </div>
        <label style="
          display:inline-block;padding:6px 16px;
          background:#1a3a5a;color:#7ab8e0;border:1px solid #2a5a8a;
          border-radius:6px;font-size:0.8rem;cursor:pointer;
          transition:background 0.15s;
        ">
          Browse
          <input type="file" accept=".pkl" style="display:none" id="smpl-file-input" />
        </label>
      </div>
    `;

    const fileInput = zone.querySelector('#smpl-file-input') as HTMLInputElement;

    // Drag-and-drop
    zone.addEventListener('dragover', (e) => {
      e.preventDefault();
      zone.style.borderColor = '#4a9fc2';
      zone.style.background = '#0d1a24';
    });
    zone.addEventListener('dragleave', () => {
      zone.style.borderColor = '#444';
      zone.style.background = '#0d0d14';
    });
    zone.addEventListener('drop', (e) => {
      e.preventDefault();
      zone.style.borderColor = '#444';
      zone.style.background = '#0d0d14';
      const file = e.dataTransfer?.files[0];
      if (file) this.handleFile(file);
    });

    fileInput.addEventListener('change', () => {
      const file = fileInput.files?.[0];
      if (file) this.handleFile(file);
    });

    c.appendChild(zone);

    const hint = document.createElement('div');
    hint.style.cssText = 'font-size:0.7rem;color:#405060;text-align:center;max-width:360px;';
    hint.textContent = 'The model file is stored in your browser (IndexedDB) — it will not be uploaded anywhere.';
    c.appendChild(hint);
  }

  private renderParsing(fileName: string) {
    const c = this.opts.container;
    c.innerHTML = `
      <div style="text-align:center;padding:32px;font-family:system-ui,sans-serif;">
        <div style="font-size:0.9rem;color:#7ab8e0;margin-bottom:8px;">Parsing ${fileName}…</div>
        <div style="
          width:240px;height:6px;background:#1a2a3a;border-radius:3px;
          overflow:hidden;margin:0 auto;
        ">
          <div id="smpl-parse-bar" style="
            height:100%;width:0%;background:linear-gradient(90deg,#4a9fc2,#7ab8e0);
            border-radius:3px;transition:width 0.3s;
          "></div>
        </div>
        <div id="smpl-parse-status" style="font-size:0.75rem;color:#405060;margin-top:8px;"></div>
      </div>
    `;
  }

  private renderLoaded(data: SMPLModelData, fromCache = false) {
    const c = this.opts.container;
    const mb = (arr: ArrayBufferView) => (arr.byteLength / 1024 / 1024).toFixed(1);

    c.innerHTML = `
      <div style="
        background:#0d1a0d;border:1px solid #2a4a2a;border-radius:10px;
        padding:20px 24px;font-family:system-ui,sans-serif;min-width:280px;
      ">
        <div style="display:flex;align-items:center;gap:10px;margin-bottom:14px;">
          <span style="font-size:1.6rem;">&#10004;</span>
          <div>
            <div style="font-size:0.9rem;font-weight:600;color:#4ade80;">SMPL Model Loaded</div>
            <div style="font-size:0.72rem;color:#406040;">
              ${fromCache ? 'Restored from browser cache' : 'Parsed from file'}
            </div>
          </div>
        </div>
        <table style="border-collapse:collapse;width:100%;font-size:0.78rem;color:#90b090;">
          <tbody>
            <tr><td style="padding:2px 8px 2px 0;color:#607060;">Vertices</td>
                <td style="color:#a0d0a0;">${data.vertices.length / 3} &nbsp;(${mb(data.vertices)} MB)</td></tr>
            <tr><td style="padding:2px 8px 2px 0;color:#607060;">Faces</td>
                <td style="color:#a0d0a0;">${data.faces.length / 3}</td></tr>
            <tr><td style="padding:2px 8px 2px 0;color:#607060;">Shape dirs</td>
                <td style="color:#a0d0a0;">${data.shapedirs.length} floats &nbsp;(${mb(data.shapedirs)} MB)</td></tr>
            <tr><td style="padding:2px 8px 2px 0;color:#607060;">Pose dirs</td>
                <td style="color:#a0d0a0;">${data.posedirs.length} floats &nbsp;(${mb(data.posedirs)} MB)</td></tr>
            <tr><td style="padding:2px 8px 2px 0;color:#607060;">J_regressor</td>
                <td style="color:#a0d0a0;">${data.J_regressor.length} floats</td></tr>
            <tr><td style="padding:2px 8px 2px 0;color:#607060;">Weights</td>
                <td style="color:#a0d0a0;">${data.weights.length} floats</td></tr>
          </tbody>
        </table>
        <div style="margin-top:12px;display:flex;gap:8px;">
          <button id="smpl-clear-btn" style="
            padding:4px 12px;background:#1a0808;color:#f87171;
            border:1px solid #4a1010;border-radius:5px;font-size:0.75rem;cursor:pointer;
          ">Clear model</button>
        </div>
      </div>
    `;

    const clearBtn = c.querySelector('#smpl-clear-btn') as HTMLButtonElement;
    clearBtn.addEventListener('click', () => this.clearModel());
  }

  private renderError(msg: string) {
    const c = this.opts.container;
    c.innerHTML = `
      <div style="
        background:#1a0808;border:1px solid #4a1010;border-radius:10px;
        padding:20px 24px;font-family:system-ui,sans-serif;min-width:280px;
        text-align:center;
      ">
        <div style="font-size:1.4rem;margin-bottom:8px;">&#10060;</div>
        <div style="font-size:0.85rem;color:#f87171;margin-bottom:8px;">${msg}</div>
        <button id="smpl-retry-btn" style="
          padding:6px 16px;background:#2a1010;color:#f0a0a0;
          border:1px solid #6a2020;border-radius:5px;font-size:0.8rem;cursor:pointer;
        ">Try again</button>
      </div>
    `;
    c.querySelector('#smpl-retry-btn')?.addEventListener('click', () => this.renderEmpty());
  }

  // ── File handling ─────────────────────────────────────────────────────────

  private async handleFile(file: File) {
    this.renderParsing(file.name);

    const bar = this.opts.container.querySelector('#smpl-parse-bar') as HTMLElement | null;
    const status = this.opts.container.querySelector('#smpl-parse-status') as HTMLElement | null;

    const setProgress = (pct: number, msg: string) => {
      if (bar) bar.style.width = `${pct}%`;
      if (status) status.textContent = msg;
    };

    try {
      setProgress(10, 'Reading file…');
      const arrayBuffer = await file.arrayBuffer();

      setProgress(30, 'Parsing pickle…');
      // Yield to let the UI update
      await new Promise<void>((r) => setTimeout(r, 0));

      const bytes = new Uint8Array(arrayBuffer);
      const parser = new PickleParser(bytes);
      const raw = parser.parse();

      setProgress(70, 'Extracting SMPL arrays…');
      await new Promise<void>((r) => setTimeout(r, 0));

      const data = extractSMPLData(raw);
      this.model = data;

      setProgress(90, 'Saving to IndexedDB…');
      await new Promise<void>((r) => setTimeout(r, 0));

      try {
        await idbSave(data);
      } catch {
        // Cache failure is non-fatal
        console.warn('[SMPLLoaderUI] IndexedDB save failed — model will be lost on reload');
      }

      setProgress(100, 'Done');
      await new Promise<void>((r) => setTimeout(r, 200));

      this.renderLoaded(data, false);
      this.opts.onLoad?.(data);
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err));
      console.error('[SMPLLoaderUI] Parse error:', error);
      this.renderError(error.message);
      this.opts.onError?.(error);
    }
  }

  // ── Public API ────────────────────────────────────────────────────────────

  getModel(): SMPLModelData | null {
    return this.model;
  }

  async clearModel(): Promise<void> {
    this.model = null;
    try {
      await idbClear();
    } catch {
      // Best effort
    }
    if (!this.aborted) this.renderEmpty();
  }

  dispose(): void {
    this.aborted = true;
    this.model = null;
    this.opts.container.innerHTML = '';
  }
}
