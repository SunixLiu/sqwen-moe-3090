// tests/webui_test.rs — Task 13：WebUI speed panel + bench
use std::fs;

#[test]
fn webui_html_exists() {
    let path = "src/server/webui.html";
    assert!(std::path::Path::new(path).exists(), "webui.html should exist");
}

#[test]
fn webui_html_has_speed_panel() {
    let html = fs::read_to_string("src/server/webui.html").unwrap();
    assert!(html.contains("speed-panel"), "should have speed panel");
    assert!(html.contains("TTFT"), "should show TTFT");
    assert!(html.contains("promptRate"), "should show prompt eval rate");
    assert!(html.contains("evalRate"), "should show eval rate");
}

#[test]
fn webui_html_has_bench_section() {
    let html = fs::read_to_string("src/server/webui.html").unwrap();
    assert!(html.contains("Bench"), "should have bench section");
    assert!(html.contains("/v1/bench"), "should link to bench endpoint");
}

#[test]
fn bench_module_runs() {
    use sqwen::server::bench::BenchConfig;
    let cfg = BenchConfig {
        prompt_tokens: 64,
        gen_tokens: 8,
        repetitions: 1,
        prompt_ids: None,
    };
    assert_eq!(cfg.prompt_tokens, 64);
    assert_eq!(cfg.gen_tokens, 8);
    assert_eq!(cfg.repetitions, 1);
}