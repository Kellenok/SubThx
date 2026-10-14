//! Parser for the Python-literal (`repr()`) transcripts SubPlz keeps in its cache
//! (`<audio>.<cid>.<model>.subs`), into `serde_json::Value`.

use serde_json::{Map, Number, Value};

pub fn parse(src: &str) -> Result<Value, String> {
    let mut p = P { s: src.as_bytes(), src, i: 0 };
    let v = p.value()?;
    p.ws();
    if p.i != p.s.len() {
        return Err(format!("trailing data at {}", p.i));
    }
    Ok(v)
}

struct P<'a> {
    s: &'a [u8],
    src: &'a str,
    i: usize,
}

impl P<'_> {
    fn ws(&mut self) {
        while self.i < self.s.len() && self.s[self.i].is_ascii_whitespace() {
            self.i += 1;
        }
    }

    fn err<T>(&self, what: &str) -> Result<T, String> {
        Err(format!("{what} at byte {}", self.i))
    }

    fn value(&mut self) -> Result<Value, String> {
        self.ws();
        let Some(&c) = self.s.get(self.i) else { return self.err("unexpected end") };
        match c {
            b'{' => {
                self.i += 1;
                let mut m = Map::new();
                loop {
                    self.ws();
                    if self.s.get(self.i) == Some(&b'}') {
                        self.i += 1;
                        return Ok(Value::Object(m));
                    }
                    let k = match self.value()? {
                        Value::String(k) => k,
                        other => other.to_string(),
                    };
                    self.ws();
                    if self.s.get(self.i) != Some(&b':') {
                        return self.err("expected ':'");
                    }
                    self.i += 1;
                    let v = self.value()?;
                    m.insert(k, v);
                    self.ws();
                    if self.s.get(self.i) == Some(&b',') {
                        self.i += 1;
                    }
                }
            }
            b'[' | b'(' => {
                let close = if c == b'[' { b']' } else { b')' };
                self.i += 1;
                let mut v = Vec::new();
                loop {
                    self.ws();
                    if self.s.get(self.i) == Some(&close) {
                        self.i += 1;
                        return Ok(Value::Array(v));
                    }
                    v.push(self.value()?);
                    self.ws();
                    if self.s.get(self.i) == Some(&b',') {
                        self.i += 1;
                    }
                }
            }
            b'\'' | b'"' => self.string(c).map(Value::String),
            _ => {
                let start = self.i;
                while self.i < self.s.len() && !b",:]})".contains(&self.s[self.i]) && !self.s[self.i].is_ascii_whitespace() {
                    self.i += 1;
                }
                let tok = &self.src[start..self.i];
                match tok {
                    "None" => Ok(Value::Null),
                    "True" => Ok(Value::Bool(true)),
                    "False" => Ok(Value::Bool(false)),
                    _ => {
                        if let Ok(n) = tok.parse::<i64>() {
                            Ok(Value::Number(n.into()))
                        } else if let Ok(f) = tok.parse::<f64>() {
                            Number::from_f64(f).map(Value::Number).ok_or_else(|| format!("bad float {tok}"))
                        } else {
                            Err(format!("bad token {tok:?} at {start}"))
                        }
                    }
                }
            }
        }
    }

    fn string(&mut self, q: u8) -> Result<String, String> {
        self.i += 1;
        let mut out = String::new();
        loop {
            let rest = &self.src[self.i..];
            let Some(j) = rest.find(|c: char| c == q as char || c == '\\') else {
                return self.err("unterminated string");
            };
            out.push_str(&rest[..j]);
            self.i += j;
            if self.s[self.i] == q {
                self.i += 1;
                return Ok(out);
            }
            // escape
            let e = *self.s.get(self.i + 1).ok_or("bad escape")?;
            self.i += 2;
            let hex = |p: &mut Self, n: usize| -> Result<char, String> {
                let h = p.src.get(p.i..p.i + n).ok_or("short escape")?;
                p.i += n;
                let v = u32::from_str_radix(h, 16).map_err(|e| e.to_string())?;
                char::from_u32(v).ok_or_else(|| format!("bad codepoint {v:x}"))
            };
            match e {
                b'\\' => out.push('\\'),
                b'\'' => out.push('\''),
                b'"' => out.push('"'),
                b'n' => out.push('\n'),
                b'r' => out.push('\r'),
                b't' => out.push('\t'),
                b'0' => out.push('\0'),
                b'x' => out.push(hex(self, 2)?),
                b'u' => out.push(hex(self, 4)?),
                b'U' => out.push(hex(self, 8)?),
                other => {
                    out.push('\\');
                    out.push(other as char);
                }
            }
        }
    }
}
