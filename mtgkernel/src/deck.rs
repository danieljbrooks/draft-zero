//! draft-zero `.dck` decks -> mtg-kernel registry ids.
//!
//! Parsing follows mtg-kernel's `python/tools/limited_decks_v1.py::parse_dck`: `COUNT [SET:NUMBER] NAME` or
//! `COUNT NAME` rows in file order, `NAME:` header, `//` and `#` comments, `SB:` rows excluded from the
//! mainboard. Resolution goes through mtg-kernel's own `CustomDeckV1::resolve` (registry lookup by exact
//! name, refusing tokens and anything without full engine support, at least 40 cards), plus one rule of ours:
//! every non-basic card must be in draft-zero's FDN reference list (`assets/reference/FDN_gih.json`), so
//! Pauper-registry cards that are not FDN cards cannot slip into an FDN deck.

use mtg_kernel::card_def::{card_id_by_name, CARD_DEFS, KERNEL_CARDDB_HASH};
use mtg_kernel::limited_session_v1::{CustomCardCountV1, CustomDeckV1};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::OnceLock;

pub const BASICS: [&str; 5] = ["Plains", "Island", "Swamp", "Mountain", "Forest"];

const FDN_REFERENCE_JSON: &str = include_str!("../../assets/reference/FDN_gih.json");

/// The FDN reference card names (the keys of `FDN_gih.json`'s `cards`).
pub fn fdn_reference_names() -> &'static BTreeSet<String> {
    static NAMES: OnceLock<BTreeSet<String>> = OnceLock::new();
    NAMES.get_or_init(|| {
        let v: serde_json::Value = serde_json::from_str(FDN_REFERENCE_JSON).expect("FDN_gih.json parses");
        v["cards"].as_object().expect("FDN_gih.json has a cards object").keys().cloned().collect()
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeckRow {
    pub count: u32,
    pub name: String,
}

#[derive(Debug, Clone)]
pub struct Deck {
    /// The file stem (what the records call the deck).
    pub stem: String,
    pub rows: Vec<DeckRow>,
    /// Registry ids in row/copy order, exactly as `CustomDeckV1::resolve` returns them.
    pub ids: Vec<u16>,
    /// mtg-kernel's custom-deck content identity (`limited_session_v1::content_identity`).
    pub content_id: String,
}

impl Deck {
    /// The first 16 hex digits of the content identity's sha256 (what a game record keeps of it).
    pub fn short_hash(&self) -> String {
        let hex = self.content_id.rsplit(':').next().unwrap_or(&self.content_id);
        hex.chars().take(16).collect()
    }
}

/// The mainboard rows of a `.dck` file, in file order.
pub fn parse_dck(text: &str) -> Result<Vec<DeckRow>, String> {
    let mut rows = Vec::new();
    let mut named = false;
    for (n, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with("//") || line.starts_with('#') {
            continue;
        }
        if let Some(rest) = line.strip_prefix("NAME:") {
            if named || rest.trim().is_empty() {
                return Err(format!("line {}: invalid or repeated NAME header", n + 1));
            }
            named = true;
            continue;
        }
        let (sideboard, body) = match line.strip_prefix("SB:") {
            Some(rest) => (true, rest.trim()),
            None => (false, line),
        };
        let row =
            parse_row(body).ok_or_else(|| format!("line {}: expected COUNT [SET:NUMBER] CARD or COUNT CARD", n + 1))?;
        if !sideboard {
            rows.push(row);
        }
    }
    if rows.is_empty() {
        return Err("deck has no mainboard cards".into());
    }
    Ok(rows)
}

/// `([1-9][0-9]*)\s+(?:\[([^\[\]]+)\]\s+)?(.+)` without a regex crate.
fn parse_row(body: &str) -> Option<DeckRow> {
    let digits = body.bytes().take_while(|b| b.is_ascii_digit()).count();
    if digits == 0 || body.starts_with('0') {
        return None;
    }
    let count: u32 = body[..digits].parse().ok()?;
    let rest = &body[digits..];
    if !rest.starts_with(char::is_whitespace) {
        return None;
    }
    let mut rest = rest.trim_start();
    if let Some(after) = rest.strip_prefix('[') {
        let close = after.find(']')?;
        if close == 0 || after[..close].contains('[') {
            return None;
        }
        let tail = &after[close + 1..];
        if !tail.starts_with(char::is_whitespace) {
            return None;
        }
        rest = tail.trim_start();
    }
    let name = rest.trim();
    if name.is_empty() || name.starts_with('[') {
        return None;
    }
    Some(DeckRow { count, name: name.to_string() })
}

/// Resolve rows to registry ids: mtg-kernel's `CustomDeckV1::resolve`, then our FDN-reference rule.
pub fn resolve_rows(rows: &[DeckRow]) -> Result<Vec<u16>, String> {
    let fdn = fdn_reference_names();
    for row in rows {
        let basic = BASICS.contains(&row.name.as_str());
        if !basic && !fdn.contains(&row.name) {
            return Err(format!(
                "card {:?} is not in draft-zero's FDN reference list (assets/reference/FDN_gih.json)",
                row.name
            ));
        }
    }
    let custom = CustomDeckV1 {
        cards: rows.iter().map(|r| CustomCardCountV1 { name: r.name.clone(), count: r.count }).collect(),
    };
    let ids = custom.resolve()?;
    // Belt and braces: resolve() already refuses these.
    for &id in &ids {
        let def = &CARD_DEFS[id as usize];
        if def.is_token || !def.has_full_support() {
            return Err(format!("card {:?} is a token or lacks full engine support", def.name));
        }
    }
    Ok(ids)
}

/// mtg-kernel's private `limited_session_v1::content_identity`, reproduced byte for byte: the
/// `deck_ids` a `kernel_limited_env` reset would carry. Identity only; it does not affect play.
pub fn content_identity(ids: &[u16]) -> String {
    use sha2::{Digest, Sha256};
    let mut hash = Sha256::new();
    hash.update(b"kernel_custom_deck/v1\0");
    hash.update(KERNEL_CARDDB_HASH.to_le_bytes());
    hash.update((ids.len() as u64).to_le_bytes());
    for id in ids {
        hash.update(id.to_le_bytes());
    }
    format!("custom-v1:{:x}", hash.finalize())
}

/// `dir` may be a search path of several directories joined by `:` (the first one holding `<stem>.dck` wins).
pub fn load_deck(dir: &Path, stem: &str) -> Result<Deck, String> {
    let file = format!("{stem}.dck");
    let dirs: Vec<std::path::PathBuf> = std::env::split_paths(dir).collect();
    let path = dirs.iter().map(|d| d.join(&file)).find(|p| p.exists()).unwrap_or_else(|| dir.join(&file));
    let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
    let rows = parse_dck(&text).map_err(|e| format!("{}: {e}", path.display()))?;
    let ids = resolve_rows(&rows).map_err(|e| format!("{}: {e}", path.display()))?;
    let content_id = content_identity(&ids);
    Ok(Deck { stem: stem.to_string(), rows, ids, content_id })
}

/// A card's registry name (FDN reference names and registry names agree; see the tests).
pub fn card_name(id: u16) -> &'static str {
    CARD_DEFS[id as usize].name
}

pub fn lookup(name: &str) -> Option<u16> {
    card_id_by_name(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rows_parse_like_limited_decks_v1() {
        let rows = parse_dck("NAME:x\n2 [FDN:100] Beast-Kin Ranger\n// c\n8 Forest\nSB: 1 [FDN:1] Abrade\n\n").unwrap();
        assert_eq!(
            rows,
            vec![DeckRow { count: 2, name: "Beast-Kin Ranger".into() }, DeckRow { count: 8, name: "Forest".into() }]
        );
        assert!(parse_dck("0 Forest").is_err());
        assert!(parse_dck("2 [FDN:1]").is_err());
        assert!(parse_dck("NAME:a\nNAME:b\n1 Forest").is_err());
    }
}
