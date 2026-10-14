//! Minimal lenient (X)HTML tree builder, enough to mirror what SubPlz gets
//! from BeautifulSoup(html.parser) on EPUB content documents.

#[derive(Debug)]
pub enum Node {
    Element(Element),
    Text(String),
}

#[derive(Debug, Default)]
pub struct Element {
    /// lowercase tag name as written (may contain a prefix, e.g. `epub:switch`)
    pub name: String,
    /// lowercase attribute names
    pub attrs: Vec<(String, String)>,
    pub children: Vec<Node>,
}

const VOID: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr",
];

impl Element {
    pub fn attr(&self, name: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == name).map(|(_, v)| v.as_str())
    }

    pub fn has_attr(&self, name: &str) -> bool {
        self.attrs.iter().any(|(k, _)| k == name)
    }

    /// lxml `text_content()`: every descendant text node
    pub fn text(&self) -> String {
        let mut s = String::new();
        self.collect_text(&mut s);
        s
    }

    /// bs4 `"".join(el.strings)` / `get_text()`: strings inside rt/rp/script/
    /// style/template are special string classes that `.strings` skips.
    pub fn bs4_text(&self) -> String {
        let mut s = String::new();
        self.collect_bs4(&mut s);
        s
    }

    fn collect_bs4(&self, out: &mut String) {
        for c in &self.children {
            match c {
                Node::Text(t) => out.push_str(t),
                Node::Element(e) => {
                    if !matches!(e.name.as_str(), "rt" | "rp" | "script" | "style" | "template") {
                        e.collect_bs4(out)
                    }
                }
            }
        }
    }

    fn collect_text(&self, out: &mut String) {
        for c in &self.children {
            match c {
                Node::Text(t) => out.push_str(t),
                Node::Element(e) => e.collect_text(out),
            }
        }
    }

    /// Pre-order search over descendants (bs4 `find_all`, recursive).
    pub fn find_all<'a>(&'a self, pred: &dyn Fn(&Element) -> bool, out: &mut Vec<&'a Element>) {
        for c in &self.children {
            if let Node::Element(e) = c {
                if pred(e) {
                    out.push(e);
                }
                e.find_all(pred, out);
            }
        }
    }

    pub fn find<'a>(&'a self, pred: &dyn Fn(&Element) -> bool) -> Option<&'a Element> {
        for c in &self.children {
            if let Node::Element(e) = c {
                if pred(e) {
                    return Some(e);
                }
                if let Some(f) = e.find(pred) {
                    return Some(f);
                }
            }
        }
        None
    }

    /// Direct children elements.
    pub fn child_elements(&self) -> impl Iterator<Item = &Element> {
        self.children.iter().filter_map(|c| match c {
            Node::Element(e) => Some(e),
            _ => None,
        })
    }
}

fn decode(s: &str) -> String {
    if s.contains('&') {
        html_escape::decode_html_entities(s).into_owned()
    } else {
        s.to_string()
    }
}

/// Parse into a synthetic root element.
pub fn parse(src: &str) -> Element {
    let src = src.strip_prefix('\u{feff}').unwrap_or(src);
    let b = src.as_bytes();
    let mut stack: Vec<Element> = vec![Element::default()];
    let mut i = 0;
    let mut text_start = 0;

    let flush = |stack: &mut Vec<Element>, from: usize, to: usize| {
        if to > from {
            let t = decode(&src[from..to]);
            let top = stack.last_mut().unwrap();
            if let Some(Node::Text(prev)) = top.children.last_mut() {
                prev.push_str(&t);
            } else {
                top.children.push(Node::Text(t));
            }
        }
    };

    while i < b.len() {
        if b[i] != b'<' {
            i += 1;
            continue;
        }
        let rest = &src[i..];
        if rest.starts_with("<!--") {
            flush(&mut stack, text_start, i);
            let end = rest.find("-->").map(|e| i + e + 3).unwrap_or(b.len());
            i = end;
            text_start = i;
        } else if rest.starts_with("<![CDATA[") {
            flush(&mut stack, text_start, i);
            let end = rest.find("]]>").map(|e| i + e).unwrap_or(b.len());
            let t = src[i + 9..end].to_string();
            stack.last_mut().unwrap().children.push(Node::Text(t));
            i = (end + 3).min(b.len());
            text_start = i;
        } else if rest.starts_with("<!") || rest.starts_with("<?") {
            flush(&mut stack, text_start, i);
            i = rest.find('>').map(|e| i + e + 1).unwrap_or(b.len());
            text_start = i;
        } else if rest.starts_with("</") {
            flush(&mut stack, text_start, i);
            let end = rest.find('>').map(|e| i + e).unwrap_or(b.len());
            let name = src[i + 2..end].trim().to_ascii_lowercase();
            // pop to the matching open element if any
            if let Some(pos) = stack.iter().rposition(|e| e.name == name) {
                if pos > 0 {
                    while stack.len() > pos {
                        let e = stack.pop().unwrap();
                        stack.last_mut().unwrap().children.push(Node::Element(e));
                    }
                }
            }
            i = (end + 1).min(b.len());
            text_start = i;
        } else if rest.len() > 1 && (b[i + 1] as char).is_ascii_alphabetic() {
            flush(&mut stack, text_start, i);
            let (el, self_closing, next) = parse_tag(src, i);
            i = next;
            text_start = i;
            if self_closing || VOID.contains(&el.name.as_str()) {
                stack.last_mut().unwrap().children.push(Node::Element(el));
            } else {
                stack.push(el);
            }
        } else {
            i += 1;
        }
    }
    flush(&mut stack, text_start, b.len());
    while stack.len() > 1 {
        let e = stack.pop().unwrap();
        stack.last_mut().unwrap().children.push(Node::Element(e));
    }
    stack.pop().unwrap()
}

/// Parse `<name attr="v" ...>` starting at `start` ('<'). Returns (element, self_closing, next index).
fn parse_tag(src: &str, start: usize) -> (Element, bool, usize) {
    let b = src.as_bytes();
    let mut i = start + 1;
    let name_start = i;
    while i < b.len() && !b[i].is_ascii_whitespace() && b[i] != b'>' && b[i] != b'/' {
        i += 1;
    }
    let mut el = Element {
        name: src[name_start..i].to_ascii_lowercase(),
        ..Default::default()
    };
    loop {
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        if i >= b.len() {
            return (el, false, i);
        }
        if b[i] == b'>' {
            return (el, false, i + 1);
        }
        if b[i] == b'/' {
            i += 1;
            while i < b.len() && b[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < b.len() && b[i] == b'>' {
                return (el, true, i + 1);
            }
            continue;
        }
        let an = i;
        while i < b.len() && !b[i].is_ascii_whitespace() && b[i] != b'=' && b[i] != b'>' && b[i] != b'/' {
            i += 1;
        }
        let aname = src[an..i].to_ascii_lowercase();
        while i < b.len() && b[i].is_ascii_whitespace() {
            i += 1;
        }
        let mut val = String::new();
        if i < b.len() && b[i] == b'=' {
            i += 1;
            while i < b.len() && b[i].is_ascii_whitespace() {
                i += 1;
            }
            if i < b.len() && (b[i] == b'"' || b[i] == b'\'') {
                let q = b[i];
                let vs = i + 1;
                let ve = src[vs..].find(q as char).map(|e| vs + e).unwrap_or(b.len());
                val = decode(&src[vs..ve]);
                i = (ve + 1).min(b.len());
            } else {
                let vs = i;
                while i < b.len() && !b[i].is_ascii_whitespace() && b[i] != b'>' {
                    i += 1;
                }
                val = decode(&src[vs..i]);
            }
        }
        if !aname.is_empty() {
            el.attrs.push((aname, val));
        }
    }
}
