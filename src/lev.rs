//! Levenshtein editops/opcodes, ported 1:1 from rapidfuzz-cpp (v3.x)
//! `levenshtein_editops` so that ties between equal-cost alignments are broken
//! exactly like `rapidfuzz.distance.Levenshtein.opcodes` in SubPlz.
//!
//! Strings are slices of dense symbol ids (`0..alphabet`), see [`encode`].

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tag {
    Equal,
    Replace,
    Insert,
    Delete,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EditOp {
    pub tag: Tag,
    pub src: usize,
    pub dest: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Opcode {
    pub tag: Tag,
    pub src_start: usize,
    pub src_end: usize,
    pub dest_start: usize,
    pub dest_end: usize,
}

const NONE: u32 = u32::MAX;

/// Map two char strings onto a shared dense alphabet.
pub fn encode(a: &[char], b: &[char]) -> (Vec<u32>, Vec<u32>, usize) {
    let mut map = std::collections::HashMap::new();
    let mut id = |c: char| {
        let n = map.len() as u32;
        *map.entry(c).or_insert(n)
    };
    let ea: Vec<u32> = a.iter().map(|&c| id(c)).collect();
    let eb: Vec<u32> = b.iter().map(|&c| id(c)).collect();
    (ea, eb, map.len())
}

pub fn opcodes_chars(a: &[char], b: &[char]) -> Vec<Opcode> {
    let (ea, eb, n) = encode(a, b);
    opcodes(&ea, &eb, n)
}

pub fn opcodes(s1: &[u32], s2: &[u32], alphabet: usize) -> Vec<Opcode> {
    let ops = editops(s1, s2, alphabet);
    to_opcodes(&ops, s1.len(), s2.len())
}

pub fn to_opcodes(ops: &[EditOp], src_len: usize, dest_len: usize) -> Vec<Opcode> {
    let mut blocks = Vec::new();
    let (mut src_pos, mut dest_pos, mut i) = (0, 0, 0);
    while i < ops.len() {
        if src_pos < ops[i].src || dest_pos < ops[i].dest {
            blocks.push(Opcode {
                tag: Tag::Equal,
                src_start: src_pos,
                src_end: ops[i].src,
                dest_start: dest_pos,
                dest_end: ops[i].dest,
            });
            src_pos = ops[i].src;
            dest_pos = ops[i].dest;
        }
        let (src_begin, dest_begin, tag) = (src_pos, dest_pos, ops[i].tag);
        while i < ops.len() && ops[i].tag == tag && src_pos == ops[i].src && dest_pos == ops[i].dest
        {
            match tag {
                Tag::Replace => {
                    src_pos += 1;
                    dest_pos += 1
                }
                Tag::Insert => dest_pos += 1,
                Tag::Delete => src_pos += 1,
                Tag::Equal => unreachable!(),
            }
            i += 1;
        }
        blocks.push(Opcode {
            tag,
            src_start: src_begin,
            src_end: src_pos,
            dest_start: dest_begin,
            dest_end: dest_pos,
        });
    }
    if src_pos < src_len || dest_pos < dest_len {
        blocks.push(Opcode {
            tag: Tag::Equal,
            src_start: src_pos,
            src_end: src_len,
            dest_start: dest_pos,
            dest_end: dest_len,
        });
    }
    blocks
}

/// `rapidfuzz::levenshtein_editops` with `score_hint = None` (SIZE_MAX).
pub fn editops(s1: &[u32], s2: &[u32], alphabet: usize) -> Vec<EditOp> {
    let mut ctx = Ctx {
        ops: Vec::new(),
        local: vec![NONE; alphabet],
    };
    let max = s1.len().max(s2.len());
    ctx.align_hirschberg(s1, s2, 0, 0, 0, max);
    ctx.ops
}

#[inline]
fn shr64(a: u64, shift: isize) -> u64 {
    if shift < 0 {
        // only reached with a == 0 in rapidfuzz (default-initialised entries)
        if a == 0 {
            0
        } else {
            a << (-shift) as u32
        }
    } else if shift < 64 {
        a >> shift
    } else {
        0
    }
}

/// Recorded VP/VN rows with a per-row column offset (rapidfuzz ShiftedBitMatrix).
struct BitMatrix {
    cols: usize,
    vp: Vec<u64>,
    vn: Vec<u64>,
    offsets: Vec<isize>,
}

impl BitMatrix {
    fn new(rows: usize, cols: usize) -> Self {
        BitMatrix {
            cols,
            vp: vec![!0u64; rows * cols],
            vn: vec![0u64; rows * cols],
            offsets: vec![0; rows],
        }
    }
    #[inline]
    fn test(&self, bits: &[u64], row: usize, col: usize) -> bool {
        let off = self.offsets[row];
        let col = if off < 0 {
            col + (-off) as usize
        } else if col >= off as usize {
            col - off as usize
        } else {
            return false;
        };
        let w = col / 64;
        debug_assert!(w < self.cols);
        bits[row * self.cols + w] & (1u64 << (col % 64)) != 0
    }
}

struct MatrixResult {
    m: Option<BitMatrix>,
    dist: usize,
}

struct BitRow {
    first_block: usize,
    last_block: usize,
    prev_score: usize,
    vecs: Vec<(u64, u64)>,
    dist: usize,
}

/// Block pattern match vector over a call-local alphabet.
struct BlockPM {
    words: usize,
    table: Vec<u64>,
}

struct Ctx {
    ops: Vec<EditOp>,
    /// scratch: global id -> local id, always reset to NONE after use
    local: Vec<u32>,
}

impl Ctx {
    /// Build PM for s1 and the local ids of s2 (NONE when absent from s1).
    fn block_pm(&mut self, s1: &[u32], s2: &[u32]) -> (BlockPM, Vec<u32>) {
        let mut n = 0u32;
        for &c in s1 {
            if self.local[c as usize] == NONE {
                self.local[c as usize] = n;
                n += 1;
            }
        }
        let words = s1.len().div_ceil(64);
        let mut table = vec![0u64; n as usize * words];
        for (i, &c) in s1.iter().enumerate() {
            let l = self.local[c as usize] as usize;
            table[l * words + i / 64] |= 1u64 << (i % 64);
        }
        let s2l: Vec<u32> = s2.iter().map(|&c| self.local[c as usize]).collect();
        for &c in s1 {
            self.local[c as usize] = NONE;
        }
        (BlockPM { words, table }, s2l)
    }

    fn ensure_ops(&mut self, n: usize) {
        if self.ops.is_empty() {
            self.ops = vec![
                EditOp {
                    tag: Tag::Equal,
                    src: 0,
                    dest: 0
                };
                n
            ];
        }
    }

    fn align_hirschberg(
        &mut self,
        s1: &[u32],
        s2: &[u32],
        mut src_pos: usize,
        mut dest_pos: usize,
        editop_pos: usize,
        max: usize,
    ) {
        let (s1, s2, prefix) = remove_common_affix(s1, s2);
        src_pos += prefix;
        dest_pos += prefix;

        let max = max.min(s1.len().max(s2.len()));
        let full_band = s1.len().min(2 * max + 1);
        let matrix_size = 2 * full_band * s2.len() / 8;
        if matrix_size < 1024 * 1024 || s1.len() < 65 || s2.len() < 10 {
            self.levenshtein_align(s1, s2, max, src_pos, dest_pos, editop_pos);
        } else {
            let h = self.find_hirschberg_pos(s1, s2, max);
            self.ensure_ops(h.left_score + h.right_score);
            self.align_hirschberg(
                &s1[..h.s1_mid],
                &s2[..h.s2_mid],
                src_pos,
                dest_pos,
                editop_pos,
                h.left_score,
            );
            self.align_hirschberg(
                &s1[h.s1_mid..],
                &s2[h.s2_mid..],
                src_pos + h.s1_mid,
                dest_pos + h.s2_mid,
                editop_pos + h.left_score,
                h.right_score,
            );
        }
    }

    fn levenshtein_align(
        &mut self,
        s1: &[u32],
        s2: &[u32],
        max: usize,
        src_pos: usize,
        dest_pos: usize,
        editop_pos: usize,
    ) {
        let max = max.min(s1.len().max(s2.len()));
        let full_band = s1.len().min(2 * max + 1);
        let res = if s1.is_empty() || s2.is_empty() {
            MatrixResult {
                m: None,
                dist: s1.len() + s2.len(),
            }
        } else if s1.len() <= 64 {
            self.hyrroe2003_matrix(s1, s2)
        } else if full_band <= 64 {
            self.small_band_matrix(s1, s2, max)
        } else {
            let (pm, s2l) = self.block_pm(s1, s2);
            match block(&pm, s1.len(), &s2l, max, true, usize::MAX) {
                BlockOut::Matrix(m) => m,
                _ => unreachable!(),
            }
        };
        debug_assert!(res.dist <= max);
        if res.dist != 0 {
            self.ensure_ops(res.dist);
            self.recover_alignment(s1, s2, &res, src_pos, dest_pos, editop_pos);
        }
    }

    fn recover_alignment(
        &mut self,
        s1: &[u32],
        s2: &[u32],
        res: &MatrixResult,
        src_pos: usize,
        dest_pos: usize,
        editop_pos: usize,
    ) {
        let mut dist = res.dist;
        let mut col = s1.len();
        let mut row = s2.len();
        let ops = &mut self.ops;
        let mut put = |dist: usize, tag: Tag, col: usize, row: usize| {
            ops[editop_pos + dist] = EditOp {
                tag,
                src: col + src_pos,
                dest: row + dest_pos,
            };
        };
        if let Some(m) = &res.m {
            while row != 0 && col != 0 {
                if m.test(&m.vp, row - 1, col - 1) {
                    dist -= 1;
                    col -= 1;
                    put(dist, Tag::Delete, col, row);
                } else {
                    row -= 1;
                    if row != 0 && m.test(&m.vn, row - 1, col - 1) {
                        dist -= 1;
                        put(dist, Tag::Insert, col, row);
                    } else {
                        col -= 1;
                        if s1[col] != s2[row] {
                            dist -= 1;
                            put(dist, Tag::Replace, col, row);
                        }
                    }
                }
            }
        }
        while col != 0 {
            dist -= 1;
            col -= 1;
            put(dist, Tag::Delete, col, row);
        }
        while row != 0 {
            dist -= 1;
            row -= 1;
            put(dist, Tag::Insert, col, row);
        }
    }

    /// levenshtein_hyrroe2003<RecordMatrix=true> (s1.len() <= 64)
    fn hyrroe2003_matrix(&mut self, s1: &[u32], s2: &[u32]) -> MatrixResult {
        let (pm, s2l) = self.block_pm(s1, s2);
        let mut vp = !0u64;
        let mut vn = 0u64;
        let mut dist = s1.len();
        let mut m = BitMatrix::new(s2.len(), 1);
        let mask = 1u64 << (s1.len() - 1);
        for (i, &c) in s2l.iter().enumerate() {
            let x = if c == NONE { 0 } else { pm.table[c as usize] };
            let d0 = (((x & vp).wrapping_add(vp)) ^ vp) | x | vn;
            let mut hp = vn | !(d0 | vp);
            let mut hn = d0 & vp;
            dist += (hp & mask != 0) as usize;
            dist -= (hn & mask != 0) as usize;
            hp = (hp << 1) | 1;
            hn <<= 1;
            vp = hn | !(d0 | hp);
            vn = hp & d0;
            m.vp[i] = vp;
            m.vn[i] = vn;
        }
        MatrixResult { m: Some(m), dist }
    }

    /// levenshtein_hyrroe2003_small_band<RecordMatrix=true>
    fn small_band_matrix(&mut self, s1: &[u32], s2: &[u32], max: usize) -> MatrixResult {
        // online PM: per local char (last position, bits)
        let mut n = 0u32;
        for &c in s1 {
            if self.local[c as usize] == NONE {
                self.local[c as usize] = n;
                n += 1;
            }
        }
        let s1l: Vec<u32> = s1.iter().map(|&c| self.local[c as usize]).collect();
        let s2l: Vec<u32> = s2.iter().map(|&c| self.local[c as usize]).collect();
        for &c in s1 {
            self.local[c as usize] = NONE;
        }
        let mut pm: Vec<(isize, u64)> = vec![(0, 0); n as usize];

        let mut vp: u64 = !0u64 << (64 - max - 1);
        let mut vn: u64 = 0;
        let mut dist = max;
        let mut m = BitMatrix::new(s2.len(), 1);
        let start_offset = max as isize + 2 - 64;
        for i in 0..s2.len() {
            m.offsets[i] = start_offset + i as isize;
        }
        let diagonal_mask = 1u64 << 63;
        let mut horizontal_mask = 1u64 << 62;
        let break_score = 2 * max + s2.len() - s1.len();

        let mut it1 = 0usize;
        let mut j = -(max as isize);
        while j < 0 {
            let x = &mut pm[s1l[it1] as usize];
            x.1 = shr64(x.1, j - x.0) | (1u64 << 63);
            x.0 = j;
            it1 += 1;
            j += 1;
        }

        let get = |pm: &Vec<(isize, u64)>, c: u32, i: usize| -> u64 {
            if c == NONE {
                0
            } else {
                let x = pm[c as usize];
                shr64(x.1, i as isize - x.0)
            }
        };

        let mut i = 0usize;
        while i < s1.len() - max {
            {
                let x = &mut pm[s1l[it1] as usize];
                x.1 = shr64(x.1, i as isize - x.0) | (1u64 << 63);
                x.0 = i as isize;
            }
            let pm_j = get(&pm, s2l[i], i);
            let x = pm_j;
            let d0 = (((x & vp).wrapping_add(vp)) ^ vp) | x | vn;
            let hp = vn | !(d0 | vp);
            let hn = d0 & vp;
            dist += (d0 & diagonal_mask == 0) as usize;
            if dist > break_score {
                return MatrixResult {
                    m: Some(m),
                    dist: max + 1,
                };
            }
            vp = hn | !((d0 >> 1) | hp);
            vn = (d0 >> 1) & hp;
            m.vp[i] = vp;
            m.vn[i] = vn;
            i += 1;
            it1 += 1;
        }
        while i < s2.len() {
            if it1 < s1.len() {
                let x = &mut pm[s1l[it1] as usize];
                x.1 = shr64(x.1, i as isize - x.0) | (1u64 << 63);
                x.0 = i as isize;
                it1 += 1;
            }
            let pm_j = get(&pm, s2l[i], i);
            let x = pm_j;
            let d0 = (((x & vp).wrapping_add(vp)) ^ vp) | x | vn;
            let hp = vn | !(d0 | vp);
            let hn = d0 & vp;
            dist += (hp & horizontal_mask != 0) as usize;
            dist -= (hn & horizontal_mask != 0) as usize;
            horizontal_mask >>= 1;
            if dist > break_score {
                return MatrixResult {
                    m: Some(m),
                    dist: max + 1,
                };
            }
            vp = hn | !((d0 >> 1) | hp);
            vn = (d0 >> 1) & hp;
            m.vp[i] = vp;
            m.vn[i] = vn;
            i += 1;
        }
        if dist > max {
            dist = max + 1;
        }
        MatrixResult { m: Some(m), dist }
    }

    fn levenshtein_row(&mut self, s1: &[u32], s2: &[u32], max: usize, stop_row: usize) -> BitRow {
        let (pm, s2l) = self.block_pm(s1, s2);
        match block(&pm, s1.len(), &s2l, max, false, stop_row) {
            BlockOut::Row(r) => r,
            BlockOut::Dist(d) => BitRow {
                first_block: 0,
                last_block: 0,
                prev_score: 0,
                vecs: Vec::new(),
                dist: d,
            },
            BlockOut::Matrix(_) => unreachable!(),
        }
    }

    fn find_hirschberg_pos(&mut self, s1: &[u32], s2: &[u32], max: usize) -> HirschbergPos {
        let mut hpos = HirschbergPos::default();
        let left_size = s2.len() / 2;
        let right_size = s2.len() - left_size;
        hpos.s2_mid = left_size;
        let s1_len = s1.len();
        let mut best_score = usize::MAX;

        let r1: Vec<u32> = s1.iter().rev().copied().collect();
        let r2: Vec<u32> = s2.iter().rev().copied().collect();
        let right_row = self.levenshtein_row(&r1, &r2, max, right_size - 1);
        if right_row.dist > max {
            return self.find_hirschberg_pos(s1, s2, max * 2);
        }
        let right_first_pos = right_row.first_block * 64;
        let right_last_pos = s1_len.min(right_row.last_block * 64 + 64);
        let mut right_scores = vec![0usize; right_last_pos - right_first_pos + 1];
        right_scores[0] = right_row.prev_score;
        for i in right_first_pos..right_last_pos {
            let col_mask = 1u64 << (i % 64);
            let (vp, vn) = right_row.vecs[i / 64];
            let k = i - right_first_pos;
            right_scores[k + 1] = right_scores[k];
            right_scores[k + 1] -= (vn & col_mask != 0) as usize;
            right_scores[k + 1] += (vp & col_mask != 0) as usize;
        }

        let left_row = self.levenshtein_row(s1, s2, max, left_size - 1);
        if left_row.dist > max {
            return self.find_hirschberg_pos(s1, s2, max * 2);
        }
        let left_first_pos = left_row.first_block * 64;
        let left_last_pos = s1_len.min(left_row.last_block * 64 + 64);
        let mut left_score = left_row.prev_score;
        if s1_len >= left_first_pos + right_first_pos {
            let right_index = s1_len - left_first_pos - right_first_pos;
            if right_index < right_scores.len() {
                best_score = right_scores[right_index] + left_score;
                hpos.left_score = left_score;
                hpos.right_score = right_scores[right_index];
                hpos.s1_mid = left_first_pos;
            }
        }
        for i in left_first_pos..left_last_pos {
            let col_mask = 1u64 << (i % 64);
            let (vp, vn) = left_row.vecs[i / 64];
            left_score -= (vn & col_mask != 0) as usize;
            left_score += (vp & col_mask != 0) as usize;
            if s1_len < i + 1 + right_first_pos {
                continue;
            }
            let right_index = s1_len - i - 1 - right_first_pos;
            if right_index >= right_scores.len() {
                continue;
            }
            if right_scores[right_index] + left_score < best_score {
                best_score = right_scores[right_index] + left_score;
                hpos.left_score = left_score;
                hpos.right_score = right_scores[right_index];
                hpos.s1_mid = i + 1;
            }
        }
        if hpos.left_score + hpos.right_score > max {
            self.find_hirschberg_pos(s1, s2, max * 2)
        } else {
            hpos
        }
    }
}

#[derive(Default, Debug)]
struct HirschbergPos {
    left_score: usize,
    right_score: usize,
    s1_mid: usize,
    s2_mid: usize,
}

fn remove_common_affix<'a>(s1: &'a [u32], s2: &'a [u32]) -> (&'a [u32], &'a [u32], usize) {
    let prefix = s1.iter().zip(s2).take_while(|(a, b)| a == b).count();
    let (s1, s2) = (&s1[prefix..], &s2[prefix..]);
    let suffix = s1
        .iter()
        .rev()
        .zip(s2.iter().rev())
        .take_while(|(a, b)| a == b)
        .count();
    (&s1[..s1.len() - suffix], &s2[..s2.len() - suffix], prefix)
}

enum BlockOut {
    Matrix(MatrixResult),
    Row(BitRow),
    Dist(usize),
}

/// levenshtein_hyrroe2003_block. `record_matrix` selects RecordMatrix, otherwise
/// RecordBitRow at `stop_row`.
fn block(
    pm: &BlockPM,
    s1_len: usize,
    s2l: &[u32],
    mut max: usize,
    record_matrix: bool,
    stop_row: usize,
) -> BlockOut {
    let s2_len = s2l.len();
    let fail = |max: usize| -> BlockOut {
        if record_matrix {
            BlockOut::Matrix(MatrixResult {
                m: None,
                dist: max + 1,
            })
        } else {
            BlockOut::Dist(max + 1)
        }
    };
    if max < s1_len.abs_diff(s2_len) {
        return fail(max);
    }
    const WS: usize = 64;
    let words = pm.words;
    let mut vecs: Vec<(u64, u64)> = vec![(!0u64, 0u64); words];
    let mut scores = vec![0usize; words];
    let last: u64 = 1u64 << ((s1_len - 1) % WS);
    for (i, s) in scores.iter_mut().enumerate().take(words - 1) {
        *s = (i + 1) * WS;
    }
    scores[words - 1] = s1_len;

    let mut mat = if record_matrix {
        let full_band = s1_len.min(2 * max + 1);
        let full_band_words = words.min(full_band / WS + 2);
        Some(BitMatrix::new(s2_len, full_band_words))
    } else {
        None
    };

    max = max.min(s1_len.max(s2_len));
    let mut first_block: usize = 0;
    let mut last_block: isize = {
        let a = max.min(((max + s1_len) as isize - s2_len as isize) as usize / 2);
        (words.min((a + 1).div_ceil(WS)) as isize) - 1
    };

    let get_row_num = |word: usize| -> usize {
        if word + 1 == words {
            s1_len - 1
        } else {
            (word + 1) * WS - 1
        }
    };

    for row in 0..s2_len {
        let mut hp_carry: u64 = 1;
        let mut hn_carry: u64 = 0;
        if let Some(m) = mat.as_mut() {
            m.offsets[row] = (first_block * WS) as isize;
        }
        let ch = s2l[row];

        let mut advance_block = |word: usize,
                                 vecs: &mut Vec<(u64, u64)>,
                                 hp_carry: &mut u64,
                                 hn_carry: &mut u64,
                                 first_block: usize|
         -> i64 {
            let pm_j = if ch == NONE {
                0
            } else {
                pm.table[ch as usize * words + word]
            };
            let (vp, vn) = vecs[word];
            let x = pm_j | *hn_carry;
            let d0 = (((x & vp).wrapping_add(vp)) ^ vp) | x | vn;
            let mut hp = vn | !(d0 | vp);
            let mut hn = d0 & vp;
            let hp_carry_temp = *hp_carry;
            let hn_carry_temp = *hn_carry;
            if word < words - 1 {
                *hp_carry = hp >> 63;
                *hn_carry = hn >> 63;
            } else {
                *hp_carry = (hp & last != 0) as u64;
                *hn_carry = (hn & last != 0) as u64;
            }
            hp = (hp << 1) | hp_carry_temp;
            hn = (hn << 1) | hn_carry_temp;
            let nvp = hn | !(d0 | hp);
            let nvn = hp & d0;
            vecs[word] = (nvp, nvn);
            if let Some(m) = mat.as_mut() {
                let idx = row * m.cols + (word - first_block);
                m.vp[idx] = nvp;
                m.vn[idx] = nvn;
            }
            *hp_carry as i64 - *hn_carry as i64
        };

        let mut word = first_block as isize;
        while word <= last_block {
            let w = word as usize;
            scores[w] = (scores[w] as i64
                + advance_block(w, &mut vecs, &mut hp_carry, &mut hn_carry, first_block))
                as usize;
            word += 1;
        }

        let lb = last_block as usize;
        max = (max as isize).min(
            scores[lb] as isize
                + (s2_len as isize - row as isize - 1)
                    .max(s1_len as isize - (((1 + lb) * WS) as isize - 1) - 1),
        ) as usize;

        if (last_block as usize) + 1 < words {
            let lb = last_block as usize;
            let cond = (max + 2 * WS + row + s1_len) as isize - (scores[lb] + 2 + s2_len) as isize;
            if (get_row_num(lb) as isize) < cond {
                last_block += 1;
                let lb = last_block as usize;
                vecs[lb] = (!0u64, 0);
                let chars_in_block = if lb + 1 == words {
                    (s1_len - 1) % WS + 1
                } else {
                    64
                };
                scores[lb] = scores[lb - 1] + chars_in_block - hp_carry as usize + hn_carry as usize;
                scores[lb] = (scores[lb] as i64
                    + advance_block(lb, &mut vecs, &mut hp_carry, &mut hn_carry, first_block))
                    as usize;
            }
        }

        while last_block >= first_block as isize {
            let lb = last_block as usize;
            let c1 = scores[lb] < max + WS;
            let cond =
                (max + 2 * WS + row + s1_len + 1) as isize - (scores[lb] + 2 + s2_len) as isize;
            let c2 = get_row_num(lb) as isize <= cond;
            if c1 && c2 {
                break;
            }
            last_block -= 1;
        }

        while first_block as isize <= last_block {
            let c1 = scores[first_block] < max + WS;
            let cond = (scores[first_block] + s1_len + row) as isize - (max + s2_len) as isize;
            let c2 = get_row_num(first_block) as isize >= cond;
            if c1 && c2 {
                break;
            }
            first_block += 1;
        }

        if last_block < first_block as isize {
            return fail(max);
        }

        if !record_matrix && row == stop_row {
            let prev_score = if first_block == 0 {
                stop_row + 1
            } else {
                let relevant_bits = ((first_block + 1) * 64).min(s1_len) % 64;
                let mut mask = !0u64;
                if relevant_bits != 0 {
                    mask >>= 64 - relevant_bits;
                }
                let (vp, vn) = vecs[first_block];
                scores[first_block] + (vn & mask).count_ones() as usize
                    - (vp & mask).count_ones() as usize
            };
            return BlockOut::Row(BitRow {
                first_block,
                last_block: last_block as usize,
                prev_score,
                vecs,
                dist: 0,
            });
        }
    }

    let mut dist = scores[words - 1];
    if dist > max {
        dist = max + 1;
    }
    if record_matrix {
        BlockOut::Matrix(MatrixResult { m: mat, dist })
    } else {
        BlockOut::Dist(dist)
    }
}

/// Indel (LCS-based) distance, used for `fuzz.ratio`.
pub fn indel_distance(a: &[char], b: &[char]) -> usize {
    let lcs = lcs_len(a, b);
    a.len() + b.len() - 2 * lcs
}

/// Bit-parallel LCS length (Hyyrö).
pub fn lcs_len(a: &[char], b: &[char]) -> usize {
    let (s1, s2) = if a.len() <= b.len() { (b, a) } else { (a, b) };
    if s1.is_empty() || s2.is_empty() {
        return 0;
    }
    let (e1, e2, n) = encode(s1, s2);
    let words = e1.len().div_ceil(64);
    let mut table = vec![0u64; n * words];
    for (i, &c) in e1.iter().enumerate() {
        table[c as usize * words + i / 64] |= 1u64 << (i % 64);
    }
    let mut s = vec![!0u64; words];
    for &c in &e2 {
        let mut carry = 0u64;
        let row = &table[c as usize * words..c as usize * words + words];
        for w in 0..words {
            let u = s[w] & row[w];
            let (t1, c1) = s[w].overflowing_add(u);
            let (t2, c2) = t1.overflowing_add(carry);
            carry = (c1 || c2) as u64;
            s[w] = t2 | (s[w] - u);
        }
    }
    // bits beyond len in the last word stay 1 (never matched), count zeros in range
    let mut lcs = 0usize;
    for (w, &v) in s.iter().enumerate() {
        let bits = if w + 1 == words && e1.len() % 64 != 0 {
            e1.len() % 64
        } else {
            64
        };
        let mask = if bits == 64 { !0u64 } else { (1u64 << bits) - 1 };
        lcs += (!v & mask).count_ones() as usize;
    }
    lcs
}
