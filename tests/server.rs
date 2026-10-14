//! HTTP server: a sync job over the API must give the same subtitles as the
//! CLI (fixtures/e2e/tantei.kitsune-p0.05b.srt); the second, identical job is
//! served from the transcript cache. Needs CUDA (SUBTHX_DEVICE to override).

use clap::Parser;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};

const DIR: &str = "E:/audiobooks/Audiobook Collection/[二語 十] 探偵はもう、死んでいる。/test";

#[derive(Parser)]
struct Cli {
    #[command(flatten)]
    serve: subthx::server::ServeArgs,
}

fn request(addr: SocketAddr, method: &str, path: &str, body: &str) -> (u16, String) {
    let mut s = TcpStream::connect(addr).unwrap();
    write!(
        s,
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .unwrap();
    let mut buf = Vec::new();
    s.read_to_end(&mut buf).unwrap();
    let text = String::from_utf8(buf).unwrap();
    let (head, body) = text.split_once("\r\n\r\n").unwrap();
    let code = head.split(' ').nth(1).unwrap().parse().unwrap();
    (code, body.to_string())
}

#[test]
fn sync_job_matches_cli() {
    let device = std::env::var("SUBTHX_DEVICE").unwrap_or_else(|_| "cuda".into());
    let args = Cli::parse_from(["x", "--device", &device]).serve;
    let addr = subthx::server::spawn_for_tests(args).unwrap();

    let (code, health) = request(addr, "GET", "/v1/health", "");
    assert_eq!(code, 200, "{health}");

    let audio = format!("{DIR}/[01] 探偵はもう、死んでいる。 [B0DQCZ9TNQ].m4b");
    let text = format!("{DIR}/探偵はもう、死んでいる。【電子特典付き】.epub");
    let body = serde_json::json!({ "audio": audio, "text": text }).to_string();

    let (code, err) = request(addr, "POST", "/v1/jobs", r#"{"audio":"nope.m4b","text":"nope.epub"}"#);
    assert_eq!(code, 400, "{err}");

    let t = std::time::Instant::now();
    let (code, srt) = request(addr, "POST", "/v1/jobs?wait=true&format=srt", &body);
    let first = t.elapsed().as_secs_f64();
    assert_eq!(code, 200, "{srt}");
    let want = std::fs::read_to_string("fixtures/e2e/tantei.kitsune-p0.05b.srt").unwrap().replace("\r\n", "\n");
    assert!(srt == want, "API subtitles differ from the CLI's");

    let t = std::time::Instant::now();
    let (code, srt2) = request(addr, "POST", "/v1/jobs?wait=true&format=srt", &body);
    let second = t.elapsed().as_secs_f64();
    assert_eq!(code, 200);
    assert_eq!(srt2, srt);
    eprintln!("first job {first:.2}s, cached {second:.2}s");
    assert!(second < 1.0, "second job should hit the transcript cache");

    let (_, list) = request(addr, "GET", "/v1/jobs", "");
    let jobs: Vec<serde_json::Value> = serde_json::from_str(&list).unwrap();
    assert_eq!(jobs.len(), 2);
    assert!(jobs.iter().all(|j| j["state"] == "done"));
}
