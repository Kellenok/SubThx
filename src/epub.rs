//! EPUB reading, port of SubPlz `subplz/text.py::Epub.from_file` on top of
//! ebooklib `read_epub(path, {"ignore_ncx": True})`.

use crate::html::{self, Element};
use std::io::Read;
use std::path::Path;

#[derive(Debug, Clone, serde::Serialize)]
pub struct Chapter {
    /// spine index
    pub idx: usize,
    pub title: String,
    pub paragraphs: Vec<String>,
}

#[derive(Debug)]
pub struct Epub {
    pub title: String,
    pub chapters: Vec<Chapter>,
}

struct Item {
    id: String,
    /// unquoted manifest href (ebooklib `file_name`), relative to the OPF dir
    file_name: String,
    media_type: String,
    properties: Vec<String>,
}

struct Link {
    href: String,
    title: String,
}

const PARA_TAGS: &[&str] = &["p", "li", "blockquote", "h1", "h2", "h3", "h4", "h5", "h6"];

fn unquote(s: &str) -> String {
    percent_encoding::percent_decode_str(s).decode_utf8_lossy().into_owned()
}

/// posixpath.normpath
fn normpath(p: &str) -> String {
    let abs = p.starts_with('/');
    let mut parts: Vec<&str> = Vec::new();
    for c in p.split('/') {
        match c {
            "" | "." => {}
            ".." => {
                if parts.last().is_some_and(|l| *l != "..") {
                    parts.pop();
                } else if !abs {
                    parts.push("..");
                }
            }
            c => parts.push(c),
        }
    }
    let j = parts.join("/");
    if abs {
        format!("/{j}")
    } else if j.is_empty() {
        ".".into()
    } else {
        j
    }
}

/// posixpath.join for two parts
fn join(a: &str, b: &str) -> String {
    if b.starts_with('/') || a.is_empty() {
        b.to_string()
    } else if a.ends_with('/') {
        format!("{a}{b}")
    } else {
        format!("{a}/{b}")
    }
}

fn dirname(p: &str) -> &str {
    match p.rfind('/') {
        Some(i) => p[..i].trim_end_matches('/'),
        None => "",
    }
}

fn py_strip(s: &str) -> &str {
    crate::batch::py_strip(s)
}

pub struct Archive {
    zip: zip::ZipArchive<std::fs::File>,
}

impl Archive {
    fn read(&mut self, name: &str) -> Result<Vec<u8>, String> {
        let name = normpath(name);
        let mut f = self.zip.by_name(&name).map_err(|e| format!("{name}: {e}"))?;
        let mut v = Vec::new();
        f.read_to_end(&mut v).map_err(|e| e.to_string())?;
        Ok(v)
    }
    fn read_str(&mut self, name: &str) -> Result<String, String> {
        Ok(String::from_utf8_lossy(&self.read(name)?).into_owned())
    }
}

/// The cover image of an EPUB: the manifest item with `properties="cover-image"`
/// (EPUB 3), else the one `<meta name="cover" content="id">` points at (EPUB 2).
pub fn cover(path: &Path) -> Result<Option<(Vec<u8>, String)>, String> {
    let f = std::fs::File::open(path).map_err(|e| e.to_string())?;
    let mut ar = Archive {
        zip: zip::ZipArchive::new(f).map_err(|e| e.to_string())?,
    };
    let container = ar.read_str("META-INF/container.xml")?;
    let cdoc = roxmltree::Document::parse(&container).map_err(|e| e.to_string())?;
    let Some(opf_file) = cdoc.descendants().find(|n| local(n) == "rootfile").and_then(|n| n.attribute("full-path")) else {
        return Ok(None);
    };
    let opf_file = opf_file.to_string();
    let opf = ar.read_str(&opf_file)?;
    let odoc = roxmltree::Document::parse_with_options(
        &opf,
        roxmltree::ParsingOptions {
            allow_dtd: true,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())?;
    let items: Vec<roxmltree::Node> = odoc.descendants().filter(|n| local(n) == "item").collect();
    let by_prop = items
        .iter()
        .find(|n| n.attribute("properties").is_some_and(|p| p.split(' ').any(|x| x == "cover-image")));
    let by_meta = || {
        let id = odoc
            .descendants()
            .find(|n| local(n) == "meta" && n.attribute("name") == Some("cover"))?
            .attribute("content")?;
        items.iter().find(|n| n.attribute("id") == Some(id))
    };
    let Some(item) = by_prop.or_else(by_meta) else { return Ok(None) };
    let href = unquote(item.attribute("href").unwrap_or(""));
    let mime = item.attribute("media-type").unwrap_or("image/jpeg").to_string();
    let name = if dirname(&opf_file).is_empty() { href } else { format!("{}/{href}", dirname(&opf_file)) };
    Ok(ar.read(&name).ok().map(|b| (b, mime)))
}

fn local<'a>(n: &roxmltree::Node<'a, 'a>) -> &'a str {
    n.tag_name().name()
}

impl Epub {
    pub fn open(path: &Path) -> Result<Epub, String> {
        let f = std::fs::File::open(path).map_err(|e| e.to_string())?;
        let mut ar = Archive {
            zip: zip::ZipArchive::new(f).map_err(|e| e.to_string())?,
        };

        let container = ar.read_str("META-INF/container.xml")?;
        let cdoc = roxmltree::Document::parse(&container).map_err(|e| e.to_string())?;
        let opf_file = cdoc
            .descendants()
            .find(|n| local(n) == "rootfile" && n.attribute("media-type") == Some("application/oebps-package+xml"))
            .and_then(|n| n.attribute("full-path"))
            .ok_or("no rootfile")?
            .to_string();
        let opf_dir = dirname(&opf_file).to_string();

        let opf = ar.read_str(&opf_file)?;
        let odoc = roxmltree::Document::parse_with_options(
            &opf,
            roxmltree::ParsingOptions {
                allow_dtd: true,
                ..Default::default()
            },
        )
        .map_err(|e| e.to_string())?;

        let book_title = odoc
            .descendants()
            .find(|n| local(n) == "title" && n.tag_name().namespace() == Some("http://purl.org/dc/elements/1.1/"))
            .and_then(|n| n.text())
            .unwrap_or("")
            .to_string();

        let mut items: Vec<Item> = Vec::new();
        if let Some(manifest) = odoc.descendants().find(|n| local(n) == "manifest") {
            for r in manifest.children().filter(|n| n.is_element() && local(n) == "item") {
                items.push(Item {
                    id: r.attribute("id").unwrap_or("").to_string(),
                    file_name: unquote(r.attribute("href").unwrap_or("")),
                    media_type: r.attribute("media-type").unwrap_or("").to_string(),
                    properties: r
                        .attribute("properties")
                        .unwrap_or("")
                        .split(' ')
                        .filter(|s| !s.is_empty())
                        .map(String::from)
                        .collect(),
                });
            }
        }
        let spine_node = odoc.descendants().find(|n| local(n) == "spine").ok_or("no spine")?;
        let spine: Vec<String> = spine_node
            .children()
            .filter(|n| n.is_element())
            .map(|n| n.attribute("idref").unwrap_or("").to_string())
            .collect();
        let toc_id = spine_node.attribute("toc").unwrap_or("");

        let nav_item = items
            .iter()
            .find(|i| i.media_type == "application/xhtml+xml" && i.properties.iter().any(|p| p == "nav"));

        // ignore_ncx=True: nav document wins, NCX only when there is no nav
        let mut flat_toc: Vec<Link> = Vec::new();
        if let Some(nav) = nav_item {
            let content = ar.read_str(&join(&opf_dir, &nav.file_name))?;
            flat_toc = parse_nav(&content, dirname(&nav.file_name));
        } else if !toc_id.is_empty() {
            let ncx = items.iter().find(|i| i.id == toc_id).ok_or("Can not find ncx file.")?;
            let content = ar.read_str(&join(&opf_dir, &ncx.file_name))?;
            flat_toc = parse_ncx(&content)?;
        }

        // m = {item.id: toc_index} (last wins)
        let mut m: std::collections::HashMap<String, usize> = Default::default();
        for (i, e) in flat_toc.iter().enumerate() {
            let href = unquote(e.href.split('#').next().unwrap_or(""));
            if let Some(it) = items.iter().find(|it| it.file_name == href) {
                m.insert(it.id.clone(), i);
            }
        }

        let mut chapters = Vec::new();
        let mut prev_title = String::new();
        for (i, idref) in spine.iter().enumerate() {
            let Some(item) = items.iter().find(|it| &it.id == idref) else {
                continue;
            };
            let mut title = m.get(idref).map(|&k| flat_toc[k].title.clone()).unwrap_or_default();
            if item.media_type != "application/xhtml+xml" {
                if !title.is_empty() {
                    prev_title = title;
                }
                continue;
            }
            let content = ar.read_str(&join(&opf_dir, &item.file_name))?;
            let doc = html::parse(&content);
            let Some(body) = doc.find(&|e| e.name == "body") else {
                continue;
            };
            let mut r: Vec<&Element> = Vec::new();
            body.find_all(&|e| PARA_TAGS.contains(&e.name.as_str()), &mut r);
            let first = r.iter().position(|e| !py_strip(&e.bs4_text()).is_empty());
            let Some(first) = first else {
                if !title.is_empty() {
                    prev_title = title;
                }
                continue;
            };
            if title.is_empty() {
                let pt = py_strip(&prev_title).to_string();
                if !pt.is_empty() {
                    title = pt;
                    prev_title.clear();
                } else {
                    let t = py_strip(&r[first].bs4_text()).to_string();
                    if t.chars().count() < 25 {
                        title = t;
                    } else {
                        title = item.file_name.clone();
                    }
                }
            }
            let paragraphs = r.iter().filter(|e| !e.has_attr("id")).map(|e| e.bs4_text()).collect();
            chapters.push(Chapter { idx: i, title, paragraphs });
        }

        let title = {
            let t = py_strip(&book_title).to_string();
            if t.is_empty() {
                path.file_name().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default()
            } else {
                t
            }
        };
        Ok(Epub { title, chapters })
    }
}

/// ebooklib `_parse_nav(..., "toc")` flattened to Links (Sections are dropped
/// by SubPlz's `flatten`, only their children survive).
fn parse_nav(content: &str, base_path: &str) -> Vec<Link> {
    let doc = html::parse(content);
    let Some(nav) = doc.find(&|e| e.name == "nav" && e.attrs.iter().any(|(_, v)| v == "toc")) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    if let Some(ol) = nav.child_elements().find(|e| e.name == "ol") {
        parse_list(ol, base_path, &mut out);
    }
    out
}

fn parse_list(list: &Element, base_path: &str, out: &mut Vec<Link>) {
    for li in list.child_elements().filter(|e| e.name == "li") {
        let sub = li.child_elements().find(|e| e.name == "ol");
        let link = li.child_elements().find(|e| e.name == "a");
        if let Some(sub) = sub {
            parse_list(sub, base_path, out);
        } else if let Some(a) = link {
            if let Some(href) = a.attr("href").filter(|h| !h.is_empty()) {
                out.push(Link {
                    href: normpath(&join(base_path, href)),
                    title: a.text(),
                });
            }
        }
    }
}

/// ebooklib `_parse_ncx`, flattened to leaf navPoints.
fn parse_ncx(content: &str) -> Result<Vec<Link>, String> {
    let doc = roxmltree::Document::parse_with_options(
        content,
        roxmltree::ParsingOptions {
            allow_dtd: true,
            ..Default::default()
        },
    )
    .map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    let Some(nav_map) = doc.descendants().find(|n| local(n) == "navMap") else {
        return Ok(out);
    };
    fn walk(n: roxmltree::Node, out: &mut Vec<Link>) {
        let mut label = String::new();
        let mut content = String::new();
        let mut children = Vec::new();
        for a in n.children().filter(|c| c.is_element()) {
            match local(&a) {
                "navLabel" => {
                    label = a
                        .children()
                        .find(|c| c.is_element())
                        .and_then(|c| c.text())
                        .unwrap_or("")
                        .to_string()
                }
                "content" => content = a.attribute("src").unwrap_or("").to_string(),
                "navPoint" => children.push(a),
                _ => {}
            }
        }
        if children.is_empty() {
            out.push(Link { href: content, title: label });
        } else {
            for c in children {
                walk(c, out);
            }
        }
    }
    for np in nav_map.children().filter(|c| c.is_element() && local(c) == "navPoint") {
        walk(np, &mut out);
    }
    Ok(out)
}
