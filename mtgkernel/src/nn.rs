//! The policy/value network's forward pass on CPU (single thread, no NN crates), identical in function to
//! `py/model.py`'s `DzkNet` (the parity test checks it on real decisions).
//!
//! Architecture (DeepSets; every activation ReLU; E/H/S read from the file, defaults 32/128/256):
//! - object token: `[E(card) ⊕ onehot(zone, NZ) ⊕ feat(F)] → H → H`;
//! - per pooling group (my hand, my battlefield, opponent's battlefield, the rest): mean ⊕ max of the
//!   object vectors (an empty group pools to zeros);
//! - state `h = MLP([global(G) ⊕ pools(8H)] → S → H)`;
//! - action token: `[onehot(kind, NK) ⊕ E(src_card) ⊕ objH[src] ⊕ E(tgt_card) ⊕ objH[tgt] ⊕
//!   onehot(tgt_player, 3) ⊕ feat(A)] → H → H` (objH of a missing object is zeros; card 0 embeds to zeros);
//! - `logit_i = Linear(H→1)(ReLU(Linear(2H→H)([h ⊕ a_i])))`, a masked softmax over the decision's actions;
//! - `v = tanh(Linear(H→1)(ReLU(Linear(H→H)(h))))`, from the acting player's seat (P(win) = (1 + v) / 2).
//!
//! File format `dzk-net-v1` (written by `py/model.py::export_dzkn`): `b"DZKNET01"`, a little-endian u64
//! header length, a JSON header (`format`, `encoder`, `dims`, `tensors` [{name, shape, offset, numel}],
//! `blob_sha256`, `meta`), then the f32 little-endian blob (PyTorch `state_dict` tensors, row-major). Loading
//! checks the format, the encoder version (`encode::ENC_VERSION`), the encoder dimensions and the blob hash.
//!
//! Speed: the first layers are folded into lookup tables at load time (card embedding times its weight
//! block, zone and kind one-hots as weight columns), weights are cached transposed so every layer is a sum
//! of input-scaled weight rows over output tiles held in registers, zero inputs (ReLU outputs, sparse
//! features) are skipped, identical object tokens are computed once, and `objH[src]`/`objH[tgt]` are
//! projected once per referenced object.

use crate::encode::{self, Encoded};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex, OnceLock};

pub const NET_FORMAT: &str = "dzk-net-v1";
const MAGIC: &[u8; 8] = b"DZKNET01";

/// Transposed dense layer: `wt[i * out + o]` = PyTorch `weight[o][i]`.
#[derive(Debug, Clone)]
struct Dense {
    n_in: usize,
    n_out: usize,
    wt: Vec<f32>,
    b: Vec<f32>,
}

impl Dense {
    fn from_torch(w: &[f32], b: &[f32], n_out: usize, n_in: usize) -> Dense {
        Dense { n_in, n_out, wt: transpose(w, n_out, n_in), b: b.to_vec() }
    }

    /// `out = b + W x`.
    fn apply(&self, x: &[f32], out: &mut [f32]) {
        debug_assert!(x.len() == self.n_in && out.len() == self.n_out);
        out.copy_from_slice(&self.b);
        axpy_rows(out, &self.wt, x);
    }

    /// Row by row over `rows` inputs (`x` is rows x n_in): `out = b + W x_r` for each row.
    fn apply_rows(&self, x: &[f32], rows: usize, out: &mut [f32]) {
        debug_assert!(x.len() == rows * self.n_in && out.len() == rows * self.n_out);
        for r in 0..rows {
            out[r * self.n_out..(r + 1) * self.n_out].copy_from_slice(&self.b);
        }
        gemm_rows(out, x, rows, self.n_in, &self.wt);
    }
}

fn transpose(w: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut t = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            t[c * rows + r] = w[r * cols + c];
        }
    }
    t
}

/// The matrix kernels, with and without fused multiply-add. aarch64 always has FMA; on x86-64 the FMA
/// variant is compiled for AVX2+FMA and chosen at run time when the CPU has them (the release build targets
/// baseline x86-64). Each output's sum runs in a fixed order, so results are deterministic on a machine
/// (they differ from another kernel's in the last bits).
mod kern {
    #[inline(always)]
    fn madd<const FMA: bool>(a: f32, b: f32, c: f32) -> f32 {
        if FMA {
            a.mul_add(b, c)
        } else {
            c + a * b
        }
    }

    /// `out += Σ_i x[i] · wt[i]` over output tiles of 32 held in registers; zero inputs skipped.
    #[inline(always)]
    pub fn axpy<const FMA: bool>(out: &mut [f32], wt: &[f32], x: &[f32]) {
        let n = out.len();
        debug_assert_eq!(wt.len(), x.len() * n);
        const T: usize = 32;
        let mut c = 0;
        while c + T <= n {
            let mut acc = [0.0f32; T];
            acc.copy_from_slice(&out[c..c + T]);
            for (i, &xi) in x.iter().enumerate() {
                if xi == 0.0 {
                    continue;
                }
                let row: &[f32; T] = wt[i * n + c..i * n + c + T].try_into().unwrap();
                for j in 0..T {
                    acc[j] = madd::<FMA>(xi, row[j], acc[j]);
                }
            }
            out[c..c + T].copy_from_slice(&acc);
            c += T;
        }
        if c < n {
            for (i, &xi) in x.iter().enumerate() {
                if xi == 0.0 {
                    continue;
                }
                let row = &wt[i * n + c..i * n + n];
                for (o, w) in out[c..].iter_mut().zip(row) {
                    *o = madd::<FMA>(xi, *w, *o);
                }
            }
        }
    }

    /// `out[r] += Σ_i x[r][i] · wt[i]` for `rows` rows of `x` (k wide), four rows per pass over the weights
    /// (output tiles of 16); the rest row by row.
    #[inline(always)]
    pub fn gemm<const FMA: bool>(out: &mut [f32], x: &[f32], rows: usize, k: usize, wt: &[f32]) {
        if rows == 0 {
            return;
        }
        let n = out.len() / rows;
        debug_assert_eq!(wt.len(), k * n);
        const U: usize = 16;
        let mut r = 0;
        if n.is_multiple_of(U) {
            while r + 4 <= rows {
                let mut c = 0;
                while c + U <= n {
                    let mut acc = [[0.0f32; U]; 4];
                    for (q, a) in acc.iter_mut().enumerate() {
                        a.copy_from_slice(&out[(r + q) * n + c..(r + q) * n + c + U]);
                    }
                    for i in 0..k {
                        let xs = [x[r * k + i], x[(r + 1) * k + i], x[(r + 2) * k + i], x[(r + 3) * k + i]];
                        if xs == [0.0; 4] {
                            continue;
                        }
                        let w: &[f32; U] = wt[i * n + c..i * n + c + U].try_into().unwrap();
                        for q in 0..4 {
                            for j in 0..U {
                                acc[q][j] = madd::<FMA>(xs[q], w[j], acc[q][j]);
                            }
                        }
                    }
                    for (q, a) in acc.iter().enumerate() {
                        out[(r + q) * n + c..(r + q) * n + c + U].copy_from_slice(a);
                    }
                    c += U;
                }
                r += 4;
            }
        }
        while r < rows {
            axpy::<FMA>(&mut out[r * n..(r + 1) * n], wt, &x[r * k..(r + 1) * k]);
            r += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn axpy_avx2(out: &mut [f32], wt: &[f32], x: &[f32]) {
        axpy::<true>(out, wt, x)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,fma")]
    pub unsafe fn gemm_avx2(out: &mut [f32], x: &[f32], rows: usize, k: usize, wt: &[f32]) {
        gemm::<true>(out, x, rows, k, wt)
    }
}

/// Whether the FMA kernels run on this machine.
pub fn uses_fma() -> bool {
    #[cfg(target_arch = "aarch64")]
    {
        true
    }
    #[cfg(target_arch = "x86_64")]
    {
        std::is_x86_feature_detected!("avx2") && std::is_x86_feature_detected!("fma")
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    {
        false
    }
}

/// `out += Σ_i x[i] · wt[i]` (rows of `wt` are `out.len()` wide); zero inputs skipped.
#[inline]
fn axpy_rows(out: &mut [f32], wt: &[f32], x: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    kern::axpy::<true>(out, wt, x);
    #[cfg(target_arch = "x86_64")]
    {
        if uses_fma() {
            // SAFETY: the CPU has AVX2 and FMA (checked at run time).
            unsafe { kern::axpy_avx2(out, wt, x) }
        } else {
            kern::axpy::<false>(out, wt, x)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    kern::axpy::<false>(out, wt, x);
}

/// `out[r] += Σ_i x[r][i] · wt[i]` for each of `rows` rows (`x` is rows x k).
#[inline]
fn gemm_rows(out: &mut [f32], x: &[f32], rows: usize, k: usize, wt: &[f32]) {
    #[cfg(target_arch = "aarch64")]
    kern::gemm::<true>(out, x, rows, k, wt);
    #[cfg(target_arch = "x86_64")]
    {
        if uses_fma() {
            // SAFETY: the CPU has AVX2 and FMA (checked at run time).
            unsafe { kern::gemm_avx2(out, x, rows, k, wt) }
        } else {
            kern::gemm::<false>(out, x, rows, k, wt)
        }
    }
    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
    kern::gemm::<false>(out, x, rows, k, wt);
}

#[inline]
fn add_into(out: &mut [f32], x: &[f32]) {
    for (o, v) in out.iter_mut().zip(x) {
        *o += v;
    }
}

#[inline]
fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    // 8 partial sums (vectorizes; a fixed order, so deterministic)
    let mut acc = [0.0f32; 8];
    let mut ca = a.chunks_exact(8);
    let mut cb = b.chunks_exact(8);
    for (x, y) in (&mut ca).zip(&mut cb) {
        for j in 0..8 {
            acc[j] += x[j] * y[j];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        s += x * y;
    }
    s
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Dims {
    pub e: usize,
    pub h: usize,
    pub s: usize,
}

/// A loaded network (immutable; share it with `Arc`).
#[derive(Debug)]
pub struct Model {
    pub dims: Dims,
    /// The first 16 hex digits of the blob's sha256.
    pub hash: String,
    pub path: String,
    pub meta: Value,
    // object MLP
    obj_card: Vec<f32>, // [VOCAB][H]: E(card) · W_obj1[:, card part] + b_obj1
    obj_zone: Vec<f32>, // [NZ][H]
    obj_feat: Vec<f32>, // [F][H]
    obj2: Dense,
    // state MLP
    st1: Dense,
    st2: Dense,
    // action MLP
    act_kind: Vec<f32>,     // [NK][H] (+ b_act1)
    act_src_card: Vec<f32>, // [VOCAB][H]
    act_src_obj: Vec<f32>,  // [H][H] transposed block
    act_tgt_card: Vec<f32>,
    act_tgt_obj: Vec<f32>,
    act_tgt_player: Vec<f32>, // [NP][H]
    act_feat: Vec<f32>,       // [A][H]
    act2: Dense,
    // heads
    pol1_h: Dense,    // h part, with the bias
    pol1_a: Vec<f32>, // [H][H] transposed a part
    pol2_w: Vec<f32>, // [H]
    pol2_b: f32,
    val1: Dense,
    val2_w: Vec<f32>,
    val2_b: f32,
}

/// Per-caller buffers (one per bot or thread).
#[derive(Debug, Default, Clone)]
pub struct Scratch {
    /// object -> its unique token; unique token -> first object, its hash
    uniq: Vec<u32>,
    uniq_first: Vec<u32>,
    uniq_hash: Vec<u64>,
    x1: Vec<f32>,
    obj_h: Vec<f32>,
    /// unique token -> row in the src/tgt projections (u32::MAX: none yet)
    src_row: Vec<u32>,
    tgt_row: Vec<u32>,
    src_in: Vec<f32>,
    tgt_in: Vec<f32>,
    src_proj: Vec<f32>,
    tgt_proj: Vec<f32>,
    st_in: Vec<f32>,
    s1: Vec<f32>,
    h: Vec<f32>,
    hp: Vec<f32>,
    a1: Vec<f32>,
    a2: Vec<f32>,
    p1: Vec<f32>,
    v1: Vec<f32>,
    pub logits: Vec<f32>,
    pub value: f32,
    /// Unique object tokens of the last forward (identical tokens are computed once).
    pub unique_objects: usize,
}

fn token_hash(card: u16, zone: u8, feat: &[f32]) -> u64 {
    let mut h = (u64::from(card) << 8 | u64::from(zone)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for v in feat {
        h = (h.rotate_left(5) ^ u64::from(v.to_bits())).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    }
    h
}

struct Tensors {
    map: HashMap<String, (Vec<usize>, Vec<f32>)>,
}

impl Tensors {
    fn get(&self, name: &str, shape: &[usize]) -> Result<&[f32], String> {
        let (s, v) = self.map.get(name).ok_or_else(|| format!("tensor {name} missing"))?;
        if s.as_slice() != shape {
            return Err(format!("tensor {name}: shape {s:?}, expected {shape:?}"));
        }
        Ok(v)
    }
}

impl Model {
    /// Load a `dzk-net-v1` file (checks format, encoder version and dims, blob hash).
    pub fn load(path: &Path) -> Result<Model, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(&bytes, &path.display().to_string())
    }

    pub fn from_bytes(bytes: &[u8], path: &str) -> Result<Model, String> {
        let err = |m: String| format!("{path}: {m}");
        if bytes.len() < 16 || &bytes[..8] != MAGIC {
            return Err(err("not a dzk-net-v1 file (bad magic)".into()));
        }
        let hlen = u64::from_le_bytes(bytes[8..16].try_into().unwrap()) as usize;
        let header_end = 16 + hlen;
        if bytes.len() < header_end {
            return Err(err("truncated header".into()));
        }
        let header: Value = serde_json::from_slice(&bytes[16..header_end]).map_err(|e| err(format!("header: {e}")))?;
        if header["format"] != NET_FORMAT {
            return Err(err(format!("format {} is not {NET_FORMAT}", header["format"])));
        }
        if header["encoder"] != encode::ENC_VERSION {
            return Err(err(format!(
                "encoder {} does not match this build's {}",
                header["encoder"],
                encode::ENC_VERSION
            )));
        }
        let blob = &bytes[header_end..];
        let digest = format!("{:x}", Sha256::digest(blob));
        if header["blob_sha256"].as_str() != Some(digest.as_str()) {
            return Err(err("blob sha256 mismatch (corrupt or edited file)".into()));
        }
        let d = &header["dims"];
        let dim = |k: &str| d[k].as_u64().map(|v| v as usize).ok_or_else(|| err(format!("dims.{k} missing")));
        for (k, want) in [
            ("V", encode::VOCAB),
            ("G", encode::G),
            ("F", encode::F),
            ("A", encode::A),
            ("NZ", encode::NZ),
            ("NK", encode::NK),
            ("NP", encode::NP),
        ] {
            let got = dim(k)?;
            if got != want {
                return Err(err(format!("dims.{k} = {got}, the encoder has {want}")));
            }
        }
        let dims = Dims { e: dim("E")?, h: dim("H")?, s: dim("S")? };
        if !blob.len().is_multiple_of(4) {
            return Err(err("blob length is not a multiple of 4".into()));
        }
        let floats: Vec<f32> = blob.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect();
        let mut map = HashMap::new();
        for t in header["tensors"].as_array().ok_or_else(|| err("tensors missing".into()))? {
            let name = t["name"].as_str().ok_or_else(|| err("tensor name".into()))?.to_string();
            let shape: Vec<usize> = t["shape"]
                .as_array()
                .ok_or_else(|| err("tensor shape".into()))?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let off = t["offset"].as_u64().ok_or_else(|| err("tensor offset".into()))? as usize;
            let n: usize = shape.iter().product();
            let data = floats.get(off..off + n).ok_or_else(|| err(format!("tensor {name} out of the blob")))?;
            map.insert(name, (shape, data.to_vec()));
        }
        let t = Tensors { map };
        Self::build(
            &t,
            dims,
            header["blob_sha256"].as_str().unwrap_or_default()[..16].to_string(),
            path,
            header["meta"].clone(),
        )
        .map_err(err)
    }

    fn build(t: &Tensors, dims: Dims, hash: String, path: &str, meta: Value) -> Result<Model, String> {
        use encode::{A, F, G, NK, NP, NZ, VOCAB};
        let Dims { e, h, s } = dims;
        let emb = t.get("emb.weight", &[VOCAB, e])?;
        let obj_in = e + NZ + F;
        let w1 = t.get("obj1.weight", &[h, obj_in])?;
        let b1 = t.get("obj1.bias", &[h])?;
        let col = |w: &[f32], n_in: usize, c: usize| -> Vec<f32> { (0..h).map(|o| w[o * n_in + c]).collect() };
        // E(card) · W[:, 0:e] (+ bias) for every card
        let card_table = |w: &[f32], n_in: usize, c0: usize, bias: Option<&[f32]>| -> Vec<f32> {
            let mut out = vec![0.0f32; VOCAB * h];
            for v in 0..VOCAB {
                let ev = &emb[v * e..(v + 1) * e];
                for o in 0..h {
                    let mut acc = bias.map_or(0.0, |b| b[o]);
                    for k in 0..e {
                        acc += ev[k] * w[o * n_in + c0 + k];
                    }
                    out[v * h + o] = acc;
                }
            }
            out
        };
        let obj_card = card_table(w1, obj_in, 0, Some(b1));
        let mut obj_zone = Vec::with_capacity(NZ * h);
        for z in 0..NZ {
            obj_zone.extend(col(w1, obj_in, e + z));
        }
        let mut obj_feat = Vec::with_capacity(F * h);
        for f in 0..F {
            obj_feat.extend(col(w1, obj_in, e + NZ + f));
        }
        let obj2 = Dense::from_torch(t.get("obj2.weight", &[h, h])?, t.get("obj2.bias", &[h])?, h, h);
        let st_in = G + 8 * h;
        let st1 = Dense::from_torch(t.get("st1.weight", &[s, st_in])?, t.get("st1.bias", &[s])?, s, st_in);
        let st2 = Dense::from_torch(t.get("st2.weight", &[h, s])?, t.get("st2.bias", &[h])?, h, s);
        let act_in = NK + e + h + e + h + NP + A;
        let aw = t.get("act1.weight", &[h, act_in])?;
        let ab = t.get("act1.bias", &[h])?;
        let mut act_kind = Vec::with_capacity(NK * h);
        for k in 0..NK {
            let mut c = col(aw, act_in, k);
            add_into(&mut c, ab);
            act_kind.extend(c);
        }
        let o_src_card = NK;
        let o_src_obj = o_src_card + e;
        let o_tgt_card = o_src_obj + h;
        let o_tgt_obj = o_tgt_card + e;
        let o_tp = o_tgt_obj + h;
        let o_feat = o_tp + NP;
        let block_t = |c0: usize, n: usize| -> Vec<f32> {
            // rows i (input c0 + i) of the transposed weight
            let mut out = vec![0.0f32; n * h];
            for i in 0..n {
                for o in 0..h {
                    out[i * h + o] = aw[o * act_in + c0 + i];
                }
            }
            out
        };
        let act_src_card = card_table(aw, act_in, o_src_card, None);
        let act_tgt_card = card_table(aw, act_in, o_tgt_card, None);
        let act_src_obj = block_t(o_src_obj, h);
        let act_tgt_obj = block_t(o_tgt_obj, h);
        let act_tgt_player = block_t(o_tp, NP);
        let act_feat = block_t(o_feat, A);
        let act2 = Dense::from_torch(t.get("act2.weight", &[h, h])?, t.get("act2.bias", &[h])?, h, h);
        let pw = t.get("pol1.weight", &[h, 2 * h])?;
        let pb = t.get("pol1.bias", &[h])?;
        let mut pol1_h_w = vec![0.0f32; h * h];
        let mut pol1_a_t = vec![0.0f32; h * h];
        for o in 0..h {
            for i in 0..h {
                pol1_h_w[o * h + i] = pw[o * 2 * h + i];
                pol1_a_t[i * h + o] = pw[o * 2 * h + h + i];
            }
        }
        let pol1_h = Dense::from_torch(&pol1_h_w, pb, h, h);
        let pol2_w = t.get("pol2.weight", &[1, h])?.to_vec();
        let pol2_b = t.get("pol2.bias", &[1])?[0];
        let val1 = Dense::from_torch(t.get("val1.weight", &[h, h])?, t.get("val1.bias", &[h])?, h, h);
        let val2_w = t.get("val2.weight", &[1, h])?.to_vec();
        let val2_b = t.get("val2.bias", &[1])?[0];
        Ok(Model {
            dims,
            hash,
            path: path.to_string(),
            meta,
            obj_card,
            obj_zone,
            obj_feat,
            obj2,
            st1,
            st2,
            act_kind,
            act_src_card,
            act_src_obj,
            act_tgt_card,
            act_tgt_obj,
            act_tgt_player,
            act_feat,
            act2,
            pol1_h,
            pol1_a: pol1_a_t,
            pol2_w,
            pol2_b,
            val1,
            val2_w,
            val2_b,
        })
    }

    /// Forward one decision: `scratch.logits` (one per action) and `scratch.value` (actor's seat, in (-1, 1)).
    pub fn forward(&self, enc: &Encoded, sc: &mut Scratch) {
        use encode::{A, F, NZ};
        let h = self.dims.h;
        let n_obj = enc.n_obj();
        // ---- unique object tokens (identical tokens -- e.g. untapped basic lands -- are computed once)
        sc.uniq.clear();
        sc.uniq_first.clear();
        sc.uniq_hash.clear();
        for i in 0..n_obj {
            let feat = &enc.obj_feat[i * F..(i + 1) * F];
            let hash = token_hash(enc.obj_card[i], enc.obj_zone[i], feat);
            let same = |&j: &u32| {
                let j = j as usize;
                enc.obj_card[j] == enc.obj_card[i]
                    && enc.obj_zone[j] == enc.obj_zone[i]
                    && enc.obj_feat[j * F..(j + 1) * F].iter().zip(feat).all(|(a, b)| a.to_bits() == b.to_bits())
            };
            let found = (0..sc.uniq_first.len()).find(|&u| sc.uniq_hash[u] == hash && same(&sc.uniq_first[u]));
            match found {
                Some(u) => sc.uniq.push(u as u32),
                None => {
                    sc.uniq.push(sc.uniq_first.len() as u32);
                    sc.uniq_first.push(i as u32);
                    sc.uniq_hash.push(hash);
                }
            }
        }
        let nu = sc.uniq_first.len();
        sc.unique_objects = nu;
        // first layer from the tables, then the second layer four tokens at a time
        sc.x1.resize(nu * h, 0.0);
        for u in 0..nu {
            let i = sc.uniq_first[u] as usize;
            let card = enc.obj_card[i] as usize;
            let zone = (enc.obj_zone[i] as usize).min(NZ - 1);
            let x = &mut sc.x1[u * h..(u + 1) * h];
            x.copy_from_slice(&self.obj_card[card * h..(card + 1) * h]);
            add_into(x, &self.obj_zone[zone * h..(zone + 1) * h]);
            axpy_rows(x, &self.obj_feat, &enc.obj_feat[i * F..(i + 1) * F]);
            relu(x);
        }
        sc.obj_h.resize(nu * h, 0.0);
        self.obj2.apply_rows(&sc.x1, nu, &mut sc.obj_h);
        relu(&mut sc.obj_h);
        // ---- pools: [global, (mean, max) per group]
        let g = encode::G;
        sc.st_in.clear();
        sc.st_in.extend_from_slice(&enc.global);
        sc.st_in.resize(g + 8 * h, 0.0);
        let mut counts = [0usize; 4];
        for i in 0..n_obj {
            let grp = encode::zone_group(enc.obj_zone[i]);
            counts[grp] += 1;
            let u = sc.uniq[i] as usize;
            let oh = &sc.obj_h[u * h..(u + 1) * h];
            let base = g + grp * 2 * h;
            let (mean, max) = sc.st_in[base..base + 2 * h].split_at_mut(h);
            for j in 0..h {
                mean[j] += oh[j];
                if oh[j] > max[j] {
                    max[j] = oh[j];
                }
            }
        }
        for (grp, &c) in counts.iter().enumerate() {
            if c > 0 {
                let inv = c as f32;
                for v in &mut sc.st_in[g + grp * 2 * h..g + grp * 2 * h + h] {
                    *v /= inv;
                }
            }
        }
        // ---- state
        sc.s1.resize(self.dims.s, 0.0);
        self.st1.apply(&sc.st_in, &mut sc.s1);
        relu(&mut sc.s1);
        sc.h.resize(h, 0.0);
        self.st2.apply(&sc.s1, &mut sc.h);
        relu(&mut sc.h);
        // ---- value
        sc.v1.resize(h, 0.0);
        self.val1.apply(&sc.h, &mut sc.v1);
        relu(&mut sc.v1);
        sc.value = (dot(&self.val2_w, &sc.v1) + self.val2_b).tanh();
        // ---- actions
        let n_act = enc.n_act();
        sc.hp.resize(h, 0.0);
        self.pol1_h.apply(&sc.h, &mut sc.hp);
        // objH[src] and objH[tgt] projected once per referenced unique token
        for (refs, row, inp, proj, w) in [
            (&enc.act_src, &mut sc.src_row, &mut sc.src_in, &mut sc.src_proj, &self.act_src_obj),
            (&enc.act_tgt, &mut sc.tgt_row, &mut sc.tgt_in, &mut sc.tgt_proj, &self.act_tgt_obj),
        ] {
            row.clear();
            row.resize(nu, u32::MAX);
            inp.clear();
            let mut n = 0u32;
            for &o in refs.iter() {
                if o < 0 || o as usize >= n_obj {
                    continue;
                }
                let u = sc.uniq[o as usize] as usize;
                if row[u] == u32::MAX {
                    row[u] = n;
                    n += 1;
                    inp.extend_from_slice(&sc.obj_h[u * h..(u + 1) * h]);
                }
            }
            proj.clear();
            proj.resize(n as usize * h, 0.0);
            gemm_rows(proj, inp, n as usize, h, w);
        }
        sc.a1.resize(n_act * h, 0.0);
        for k in 0..n_act {
            let a1 = &mut sc.a1[k * h..(k + 1) * h];
            let kind = (enc.act_kind[k] as usize).min(encode::NK - 1);
            a1.copy_from_slice(&self.act_kind[kind * h..(kind + 1) * h]);
            let sc_card = enc.act_src_card[k] as usize;
            add_into(a1, &self.act_src_card[sc_card * h..(sc_card + 1) * h]);
            let tc_card = enc.act_tgt_card[k] as usize;
            add_into(a1, &self.act_tgt_card[tc_card * h..(tc_card + 1) * h]);
            let tp = (enc.act_tgt_player[k] as usize).min(encode::NP - 1);
            add_into(a1, &self.act_tgt_player[tp * h..(tp + 1) * h]);
            axpy_rows(a1, &self.act_feat, &enc.act_feat[k * A..(k + 1) * A]);
            for (idx, row, proj) in
                [(enc.act_src[k], &sc.src_row, &sc.src_proj), (enc.act_tgt[k], &sc.tgt_row, &sc.tgt_proj)]
            {
                if idx < 0 || idx as usize >= n_obj {
                    continue;
                }
                let r = row[sc.uniq[idx as usize] as usize] as usize;
                add_into(a1, &proj[r * h..(r + 1) * h]);
            }
            relu(a1);
        }
        sc.a2.resize(n_act * h, 0.0);
        self.act2.apply_rows(&sc.a1, n_act, &mut sc.a2);
        relu(&mut sc.a2);
        sc.p1.resize(n_act * h, 0.0);
        for k in 0..n_act {
            sc.p1[k * h..(k + 1) * h].copy_from_slice(&sc.hp);
        }
        gemm_rows(&mut sc.p1, &sc.a2, n_act, h, &self.pol1_a);
        relu(&mut sc.p1);
        sc.logits.clear();
        for k in 0..n_act {
            sc.logits.push(dot(&self.pol2_w, &sc.p1[k * h..(k + 1) * h]) + self.pol2_b);
        }
    }
}

/// Softmax of `logits / temperature` (temperature > 0) into `out`.
pub fn softmax(logits: &[f32], temperature: f32, out: &mut Vec<f32>) {
    out.clear();
    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let t = temperature.max(1e-6);
    let mut z = 0.0f32;
    for &l in logits {
        let e = ((l - m) / t).exp();
        out.push(e);
        z += e;
    }
    for v in out.iter_mut() {
        *v /= z;
    }
}

/// Index of the largest logit (ties to the lower index).
pub fn argmax(xs: &[f32]) -> usize {
    let mut best = 0;
    for i in 1..xs.len() {
        if xs[i] > xs[best] {
            best = i;
        }
    }
    best
}

/// A network a bot spec names: loaded once per process per file version (path, size, modification time), shared
/// by every game and thread; a file rewritten under the same path is loaded again (and has another hash).
#[derive(Clone)]
pub struct NetRef {
    pub path: String,
    pub model: Arc<Model>,
}

impl PartialEq for NetRef {
    fn eq(&self, other: &Self) -> bool {
        self.path == other.path && self.model.hash == other.model.hash
    }
}

impl std::fmt::Debug for NetRef {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "NetRef({} #{})", self.path, self.model.hash)
    }
}

impl NetRef {
    pub fn load(path: &str) -> Result<NetRef, String> {
        type Key = (String, u64, Option<std::time::SystemTime>);
        static CACHE: OnceLock<Mutex<HashMap<Key, Arc<Model>>>> = OnceLock::new();
        let md = std::fs::metadata(path).map_err(|e| format!("{path}: {e}"))?;
        let key = (path.to_string(), md.len(), md.modified().ok());
        let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        let mut c = cache.lock().map_err(|_| "model cache poisoned".to_string())?;
        if let Some(m) = c.get(&key) {
            return Ok(NetRef { path: path.to_string(), model: m.clone() });
        }
        let m = Arc::new(Model::load(Path::new(path))?);
        c.insert(key, m.clone());
        Ok(NetRef { path: path.to_string(), model: m })
    }
}
