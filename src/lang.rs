//! Text normalisation, port of `ats/lang.py` (`get_lang(lang)` with empty
//! prepend/append/nopend, which is how SubPlz's sync calls it).

use unicode_general_category::{get_general_category, GeneralCategory as G};
use unicode_normalization::UnicodeNormalization;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    Japanese,
    English,
}

impl Lang {
    pub fn from_code(code: &str) -> Lang {
        if code == "ja" {
            Lang::Japanese
        } else {
            Lang::English
        }
    }

    /// `Language.translate`: per-char table then `str.lower()`.
    pub fn translate(self, s: &str) -> String {
        match self {
            Lang::English => s.to_lowercase(),
            Lang::Japanese => s.chars().map(ja_translate).collect::<String>().to_lowercase(),
        }
    }

    pub fn clean(self, s: &str) -> String {
        match self {
            Lang::English => self.translate(s),
            Lang::Japanese => {
                let t = self.translate(s);
                // r1 `(?![。])[class]+`: a run of class chars is removed unless it
                // starts with 。; a 。 inside a run (not first) is removed too.
                let tc: Vec<char> = t.chars().collect();
                let mut kept: Vec<char> = Vec::with_capacity(tc.len());
                let mut i = 0;
                while i < tc.len() {
                    let c = tc[i];
                    if c != '。' && ja_class(c) {
                        while i < tc.len() && ja_class(tc[i]) {
                            i += 1;
                        }
                    } else {
                        kept.push(c);
                        i += 1;
                    }
                }
                // r2: (.)(?=\1+) — collapse runs of the same char to one
                let mut out = String::with_capacity(kept.len() * 3);
                for (i, &c) in kept.iter().enumerate() {
                    if c != '\n' && i + 1 < kept.len() && kept[i + 1] == c {
                        continue;
                    }
                    out.push(c);
                }
                out
            }
        }
    }

    /// `Language.normalize`: NFKD.
    pub fn normalize(self, s: &str) -> String {
        s.nfkd().collect()
    }
}

const KANSUU: &str = "一二三四五六七八九十〇零壱弐参肆伍陸漆捌玖拾";
const ARABIC: [char; 12] = ['１', '２', '３', '４', '５', '６', '７', '８', '９', '１', '０', '０'];

fn ja_translate(c: char) -> char {
    let u = c as u32;
    match c {
        'は' | 'あ' => return 'わ',
        'お' => return 'を',
        'へ' => return 'え',
        _ => {}
    }
    if (0x21..0x7F).contains(&u) {
        return char::from_u32(u + 0xFEE0).unwrap();
    }
    if (0x30A1..0x30A1 + 0x56).contains(&u) {
        return char::from_u32(u - 0x60).unwrap();
    }
    if let Some(i) = KANSUU.chars().position(|k| k == c) {
        return ARABIC[i % ARABIC.len()];
    }
    c
}

/// The r1 character class (includes 。, which is \p{P}).
fn ja_class(c: char) -> bool {
    if matches!(c, 'ー' | '々' | 'ゝ') || c.is_whitespace() {
        return true;
    }
    matches!(
        get_general_category(c),
        G::Control
            | G::Format
            | G::Surrogate
            | G::PrivateUse
            | G::Unassigned
            | G::NonspacingMark
            | G::SpacingMark
            | G::EnclosingMark
            | G::ConnectorPunctuation
            | G::DashPunctuation
            | G::OpenPunctuation
            | G::ClosePunctuation
            | G::InitialPunctuation
            | G::FinalPunctuation
            | G::OtherPunctuation
            | G::MathSymbol
            | G::CurrencySymbol
            | G::ModifierSymbol
            | G::OtherSymbol
            | G::SpaceSeparator
            | G::LineSeparator
            | G::ParagraphSeparator
    )
}
