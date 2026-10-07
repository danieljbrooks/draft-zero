//! Training data from play (`dzk play --data-out DIR`): every real decision of every seat, encoded
//! (`encode.rs`), with its policy target and, once the game ends, the result from the decision's seat.
//!
//! Format `dzk-data-v1`. A shard (`DIR/<run tag>-w<worker>.dzd.gz`, one per worker per run) is a sequence of
//! gzip members, one per finished game, each appended and flushed whole: a killed run leaves at most one
//! torn last member, which readers drop. A member holds `b"DZKG"`, u32 format version (1), u32 header
//! length, the game's JSON header (encoder version, game key, decks, bots, result, record count), then its
//! records, little-endian:
//!
//! ```text
//! u32 n_obj, u32 n_act, u8 seat (0 = P0/A, 1 = P1/B), u8 source, i8 z, u8 flags, u16 turn, u16 0,
//! u32 decision (index among the game's real decisions), u32 chosen (masked index), f32 root_q (NaN if none)
//! f32[G] global
//! u16[n_obj] obj_card, u8[n_obj] obj_zone, f32[n_obj*F] obj_feat
//! u8[n_act] act_kind, i16[n_act] act_src, u16[n_act] act_src_card, i16[n_act] act_tgt,
//! u16[n_act] act_tgt_card, u8[n_act] act_tgt_player, f32[n_act*A] act_feat
//! f32[n_act] target (sums to 1)
//! ```
//!
//! `source`: 0 search (target = root visit distribution), 1 net, 2 random, 3 rule (a `mull=` policy
//! answer), 4 other (one-hot of the chosen action for 1-4). `z` ∈ {+1, -1, 0} from the record's seat;
//! `flags`: 1 truncated game (z = 0), 2 no valid result (halted or engine error; z = 0), 4 draw.
//! `root_q`: the search's root value from the actor's seat (visit-weighted mean), when it searched.
//! `DIR/meta.json` indexes the runs (`DIR/<tag>.meta.json`: encoder version, bots, decks, seeds, counts).

use crate::encode::{self, Encoded};
use crate::record::GameRecord;
use serde_json::{json, Value};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

pub const DATA_FORMAT: &str = "dzk-data-v1";
const MEMBER_MAGIC: &[u8; 4] = b"DZKG";
const MEMBER_VERSION: u32 = 1;

pub const FLAG_TRUNCATED: u8 = 1;
pub const FLAG_INVALID: u8 = 2;
pub const FLAG_DRAW: u8 = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Source {
    Search = 0,
    Net = 1,
    Random = 2,
    Rule = 3,
    Other = 4,
}

pub const SOURCE_NAMES: [&str; 5] = ["search", "net", "random", "rule", "other"];

/// What a bot reports about its last choice, for the data (`Bot::take_target`).
#[derive(Debug, Clone, PartialEq)]
pub struct PolicyTarget {
    pub source: Source,
    /// A distribution over the masked actions (`None`: one-hot of the chosen action).
    pub probs: Option<Vec<f32>>,
    pub root_q: Option<f32>,
}

impl PolicyTarget {
    pub fn onehot(source: Source) -> Self {
        PolicyTarget { source, probs: None, root_q: None }
    }
}

/// One decision's record (decoded or about to be written).
#[derive(Debug, Clone, PartialEq)]
pub struct DecisionRecord {
    pub enc: Encoded,
    pub seat: u8,
    pub source: u8,
    pub z: i8,
    pub flags: u8,
    pub turn: u16,
    pub decision: u32,
    pub chosen: u32,
    pub root_q: f32,
    pub target: Vec<f32>,
}

/// A game's records, collected while it plays.
#[derive(Debug, Default)]
pub struct GameData {
    pub records: Vec<DecisionRecord>,
    pub encode_failures: u64,
    pub by_source: [u64; 5],
}

impl GameData {
    /// Record the current real decision of `game` (before it is played): `chosen` is the masked index.
    pub fn push(&mut self, game: &crate::game::Game, chosen: usize, target: Option<PolicyTarget>) {
        let mut enc = Encoded::default();
        if encode::encode_game(game, &mut enc).is_err() {
            self.encode_failures += 1;
            return;
        }
        let n = enc.n_act();
        let t = target.unwrap_or(PolicyTarget::onehot(Source::Other));
        let probs = match t.probs {
            Some(p) if p.len() == n && p.iter().sum::<f32>() > 0.0 => {
                let s: f32 = p.iter().sum();
                p.into_iter().map(|x| x / s).collect()
            }
            _ => {
                let mut v = vec![0.0; n];
                v[chosen.min(n - 1)] = 1.0;
                v
            }
        };
        self.by_source[t.source as usize] += 1;
        self.records.push(DecisionRecord {
            enc,
            seat: game.actor().map_or(0, |p| p.index() as u8),
            source: t.source as u8,
            z: 0,
            flags: 0,
            turn: game.turns().min(u32::from(u16::MAX)) as u16,
            decision: self.records.len() as u32,
            chosen: chosen as u32,
            root_q: t.root_q.unwrap_or(f32::NAN),
            target: probs,
        });
    }

    /// Fill each record's result from the finished game's record.
    pub fn finish(&mut self, rec: &GameRecord) {
        let invalid = rec.error.is_some();
        let truncated = rec.terminal.classification.as_deref() == Some("truncated");
        let draw = !invalid && !truncated && rec.winner.is_none();
        for r in &mut self.records {
            let seat = if r.seat == 0 { "A" } else { "B" };
            r.flags = if invalid {
                FLAG_INVALID
            } else if truncated {
                FLAG_TRUNCATED
            } else if draw {
                FLAG_DRAW
            } else {
                0
            };
            r.z = match rec.winner.as_deref() {
                Some(w) if !invalid => {
                    if w == seat {
                        1
                    } else {
                        -1
                    }
                }
                _ => 0,
            };
        }
    }

    /// The game's gzip member (header + records), ready to append to a shard.
    pub fn to_member(&self, rec: &GameRecord) -> Vec<u8> {
        let header = json!({
            "format": DATA_FORMAT,
            "encoder": encode::ENC_VERSION,
            "dims": {"G": encode::G, "F": encode::F, "A": encode::A, "V": encode::VOCAB, "NZ": encode::NZ,
                     "NK": encode::NK, "NP": encode::NP},
            "game_key": game_key(rec),
            "pair": rec.pair, "swap": rec.swap, "deck1": rec.deck1, "deck2": rec.deck2,
            "botA": rec.bot_a, "botB": rec.bot_b, "bot1": rec.bot1, "bot2": rec.bot2,
            "bot1_net": rec.bot1_net, "bot2_net": rec.bot2_net,
            "env_seed": rec.env_seed, "winner": rec.winner, "turns": rec.turns,
            "classification": rec.terminal.classification, "error": rec.error,
            "records": self.records.len(), "encode_failures": self.encode_failures,
        });
        let mut raw = Vec::with_capacity(4096 + self.records.len() * 8192);
        let h = serde_json::to_vec(&header).expect("header serializes");
        raw.extend_from_slice(MEMBER_MAGIC);
        raw.extend_from_slice(&MEMBER_VERSION.to_le_bytes());
        raw.extend_from_slice(&(h.len() as u32).to_le_bytes());
        raw.extend_from_slice(&h);
        for r in &self.records {
            write_record(&mut raw, r);
        }
        let mut gz = flate2::write::GzEncoder::new(Vec::with_capacity(raw.len() / 4), flate2::Compression::fast());
        gz.write_all(&raw).expect("in-memory gzip");
        gz.finish().expect("in-memory gzip")
    }
}

/// The identity of a game across runs: bots, seeds, decks and the networks' hashes (a network file rewritten under
/// the same path makes new keys). `dzk play` writes a game's record line before its data, so a resumed run
/// never writes a game's data twice; readers still keep the first of a repeated key.
pub fn game_key(rec: &GameRecord) -> String {
    format!(
        "{}|{}|{}|{}|{}|{}|{}|{}|{}",
        rec.bot1,
        rec.bot2,
        rec.env_seed,
        rec.pair,
        rec.swap,
        rec.deck1,
        rec.deck2,
        rec.bot1_net.as_deref().unwrap_or(""),
        rec.bot2_net.as_deref().unwrap_or("")
    )
}

fn put_f32s(out: &mut Vec<u8>, xs: &[f32]) {
    for x in xs {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

fn write_record(out: &mut Vec<u8>, r: &DecisionRecord) {
    let e = &r.enc;
    out.extend_from_slice(&(e.n_obj() as u32).to_le_bytes());
    out.extend_from_slice(&(e.n_act() as u32).to_le_bytes());
    out.extend_from_slice(&[r.seat, r.source, r.z as u8, r.flags]);
    out.extend_from_slice(&r.turn.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes());
    out.extend_from_slice(&r.decision.to_le_bytes());
    out.extend_from_slice(&r.chosen.to_le_bytes());
    out.extend_from_slice(&r.root_q.to_le_bytes());
    put_f32s(out, &e.global);
    for c in &e.obj_card {
        out.extend_from_slice(&c.to_le_bytes());
    }
    out.extend_from_slice(&e.obj_zone);
    put_f32s(out, &e.obj_feat);
    out.extend_from_slice(&e.act_kind);
    for v in &e.act_src {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in &e.act_src_card {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in &e.act_tgt {
        out.extend_from_slice(&v.to_le_bytes());
    }
    for v in &e.act_tgt_card {
        out.extend_from_slice(&v.to_le_bytes());
    }
    out.extend_from_slice(&e.act_tgt_player);
    put_f32s(out, &e.act_feat);
    put_f32s(out, &r.target);
}

struct Cursor<'a> {
    b: &'a [u8],
    at: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], String> {
        let s = self.b.get(self.at..self.at + n).ok_or("record truncated")?;
        self.at += n;
        Ok(s)
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u16s(&mut self, n: usize) -> Result<Vec<u16>, String> {
        Ok(self.take(2 * n)?.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    }
    fn i16s(&mut self, n: usize) -> Result<Vec<i16>, String> {
        Ok(self.take(2 * n)?.chunks_exact(2).map(|c| i16::from_le_bytes([c[0], c[1]])).collect())
    }
    fn f32s(&mut self, n: usize) -> Result<Vec<f32>, String> {
        Ok(self.take(4 * n)?.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect())
    }
}

/// One decoded game member.
#[derive(Debug, Clone)]
pub struct GameMember {
    pub header: Value,
    pub records: Vec<DecisionRecord>,
}

fn parse_member(raw: &[u8]) -> Result<GameMember, String> {
    let mut c = Cursor { b: raw, at: 0 };
    if c.take(4)? != MEMBER_MAGIC {
        return Err("bad member magic".into());
    }
    let v = c.u32()?;
    if v != MEMBER_VERSION {
        return Err(format!("member version {v}"));
    }
    let hl = c.u32()? as usize;
    let header: Value = serde_json::from_slice(c.take(hl)?).map_err(|e| format!("member header: {e}"))?;
    if header["encoder"] != encode::ENC_VERSION {
        return Err(format!("encoder {} is not {}", header["encoder"], encode::ENC_VERSION));
    }
    let n = header["records"].as_u64().unwrap_or(0) as usize;
    let mut records = Vec::with_capacity(n);
    for _ in 0..n {
        let n_obj = c.u32()? as usize;
        let n_act = c.u32()? as usize;
        let b = c.take(4)?;
        let (seat, source, z, flags) = (b[0], b[1], b[2] as i8, b[3]);
        let t = c.take(4)?;
        let turn = u16::from_le_bytes([t[0], t[1]]);
        let decision = c.u32()?;
        let chosen = c.u32()?;
        let root_q = f32::from_bits(c.u32()?);
        let enc = Encoded {
            global: c.f32s(encode::G)?,
            obj_card: c.u16s(n_obj)?,
            obj_zone: c.take(n_obj)?.to_vec(),
            obj_feat: c.f32s(n_obj * encode::F)?,
            act_kind: c.take(n_act)?.to_vec(),
            act_src: c.i16s(n_act)?,
            act_src_card: c.u16s(n_act)?,
            act_tgt: c.i16s(n_act)?,
            act_tgt_card: c.u16s(n_act)?,
            act_tgt_player: c.take(n_act)?.to_vec(),
            act_feat: c.f32s(n_act * encode::A)?,
            dropped_objects: 0,
            unlinked_refs: 0,
        };
        let target = c.f32s(n_act)?;
        records.push(DecisionRecord { enc, seat, source, z, flags, turn, decision, chosen, root_q, target });
    }
    Ok(GameMember { header, records })
}

/// Every complete game member of a shard (a torn last member is dropped; `.1` says whether one was).
pub fn read_shard(path: &Path) -> Result<(Vec<GameMember>, bool), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut out = Vec::new();
    let mut rest: &[u8] = &bytes;
    while !rest.is_empty() {
        // decode one member and find where it ends
        let mut d = flate2::bufread::GzDecoder::new(rest);
        let mut raw = Vec::new();
        if d.read_to_end(&mut raw).is_err() {
            return Ok((out, true));
        }
        let consumed = rest.len() - d.into_inner().len();
        rest = &rest[consumed..];
        match parse_member(&raw) {
            Ok(m) => out.push(m),
            Err(e) => return Err(format!("{}: {e}", path.display())),
        }
    }
    Ok((out, false))
}

/// The shards of a data directory, sorted.
pub fn shards_in(dir: &Path) -> Result<Vec<PathBuf>, String> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(".dzd.gz"))
        .collect();
    v.sort();
    Ok(v)
}

/// Games, records and records by source of one shard, read member by member without decoding the records (a
/// torn last member is not counted; `.3` says whether there was one).
pub fn count_shard(path: &Path) -> Result<(u64, u64, [u64; 5], bool), String> {
    let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let (mut games, mut records, mut by_source) = (0u64, 0u64, [0u64; 5]);
    let mut rest: &[u8] = &bytes;
    let rec_fixed = 28 + 4 * encode::G;
    let per_obj = 2 + 1 + 4 * encode::F;
    let per_act = 1 + 2 + 2 + 2 + 2 + 1 + 4 * encode::A + 4;
    while !rest.is_empty() {
        let mut d = flate2::bufread::GzDecoder::new(rest);
        let mut raw = Vec::new();
        if d.read_to_end(&mut raw).is_err() {
            return Ok((games, records, by_source, true));
        }
        rest = &rest[rest.len() - d.into_inner().len()..];
        let mut c = Cursor { b: &raw, at: 0 };
        if c.take(4)? != MEMBER_MAGIC || c.u32()? != MEMBER_VERSION {
            return Err(format!("{}: bad member", path.display()));
        }
        let hl = c.u32()? as usize;
        let header: Value = serde_json::from_slice(c.take(hl)?).map_err(|e| format!("member header: {e}"))?;
        for _ in 0..header["records"].as_u64().unwrap_or(0) {
            let (n_obj, n_act) = (c.u32()? as usize, c.u32()? as usize);
            let source = c.take(2)?[1] as usize;
            c.at += rec_fixed - 10 + n_obj * per_obj + n_act * per_act;
            by_source[source.min(4)] += 1;
            records += 1;
        }
        games += 1;
    }
    Ok((games, records, by_source, false))
}

/// Recount the runs of `dir` whose meta says they never finished (a killed `dzk play` saves its meta every 25
/// games only), from their shards; their metas get the exact counts and `recounted: true`.
pub fn recount_incomplete(dir: &Path) -> Result<(), String> {
    let Ok(entries) = std::fs::read_dir(dir) else { return Ok(()) };
    for p in entries.filter_map(|e| e.ok().map(|e| e.path())) {
        let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        let Some(tag) = name.strip_suffix(".meta.json").filter(|t| !t.is_empty() && *t != "meta") else { continue };
        let Ok(mut meta) =
            std::fs::read_to_string(&p).map_err(|_| ()).and_then(|t| serde_json::from_str::<Value>(&t).map_err(|_| ()))
        else {
            continue;
        };
        if meta["complete"] != json!(false) {
            continue;
        }
        let mut shards = Vec::new();
        let (mut games, mut records, mut by_source) = (0u64, 0u64, [0u64; 5]);
        for s in shards_in(dir)? {
            let file = s.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if !file.starts_with(&format!("{tag}-w")) {
                continue;
            }
            let (g, r, b, torn) = count_shard(&s)?;
            games += g;
            records += r;
            for (t, x) in by_source.iter_mut().zip(b) {
                *t += x;
            }
            shards.push(json!({"file": file, "games": g, "records": r, "torn_last_member": torn}));
        }
        meta["games"] = json!(games);
        meta["records"] = json!(records);
        meta["shards"] = json!(shards);
        meta["records_by_source"] =
            Value::Object(SOURCE_NAMES.iter().zip(by_source).map(|(k, v)| (k.to_string(), json!(v))).collect());
        meta["recounted"] = json!(true);
        std::fs::write(&p, serde_json::to_string_pretty(&meta).unwrap() + "\n")
            .map_err(|e| format!("{}: {e}", p.display()))?;
    }
    Ok(())
}

/// Rewrite `DIR/meta.json` from every `DIR/*.meta.json` (atomic rename). `totals` add up what each run wrote; a
/// game is written once (record line first, then its data: a run killed in between loses that game's data
/// instead of replaying it on resume), and a killed run's counts are redone from its shards when the next run
/// starts (`recount_incomplete`), so they match what the readers load unless runs overlap.
pub fn write_index(dir: &Path) -> Result<(), String> {
    let mut runs = Vec::new();
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)
        .map_err(|e| format!("{}: {e}", dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with(".meta.json"))
        .collect();
    entries.sort();
    let mut totals = json!({"games": 0u64, "records": 0u64});
    for p in entries {
        if let Ok(text) = std::fs::read_to_string(&p) {
            if let Ok(v) = serde_json::from_str::<Value>(&text) {
                for k in ["games", "records"] {
                    totals[k] = json!(totals[k].as_u64().unwrap_or(0) + v[k].as_u64().unwrap_or(0));
                }
                runs.push(v);
            }
        }
    }
    let index = json!({
        "format": DATA_FORMAT,
        "encoder": encode::ENC_VERSION,
        "dims": {"G": encode::G, "F": encode::F, "A": encode::A, "V": encode::VOCAB, "NZ": encode::NZ,
                 "NK": encode::NK, "NP": encode::NP},
        "totals": totals,
        "runs": runs,
    });
    let tmp = dir.join(format!(".meta.json.{}", std::process::id()));
    std::fs::write(&tmp, serde_json::to_string_pretty(&index).unwrap() + "\n")
        .map_err(|e| format!("{}: {e}", tmp.display()))?;
    std::fs::rename(&tmp, dir.join("meta.json")).map_err(|e| format!("meta.json: {e}"))
}
