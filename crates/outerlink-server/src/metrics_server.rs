//! Prometheus metrics exporter for the OutterLink server.
//!
//! Installs a `metrics-exporter-prometheus` recorder and starts an HTTP
//! listener that serves the `/metrics` scrape endpoint.

use metrics_exporter_prometheus::PrometheusBuilder;
use std::net::SocketAddr;

/// Default address for the Prometheus metrics endpoint.
pub const DEFAULT_METRICS_ADDR: &str = "0.0.0.0:9464";

/// Start the Prometheus metrics HTTP server.
///
/// Listens on the given address (e.g. `0.0.0.0:9464`) and serves the
/// standard `/metrics` endpoint. The recorder is installed globally so all
/// `metrics` macro calls from any crate will be captured.
///
/// This function returns immediately after binding the listener; the HTTP
/// server runs on a background thread managed by the exporter.
pub fn start_metrics_server(addr: SocketAddr) -> Result<(), Box<dyn std::error::Error>> {
    PrometheusBuilder::new()
        .with_http_listener(addr)
        .install()?;

    tracing::info!(%addr, "Prometheus metrics server started");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    fn get_free_port() -> u16 {
        TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap()
            .port()
    }

    #[test]
    fn test_default_metrics_addr_parses() {
        let addr: Result<SocketAddr, _> = DEFAULT_METRICS_ADDR.parse();
        assert!(addr.is_ok(), "DEFAULT_METRICS_ADDR must be a valid SocketAddr");
    }

    #[test]
    fn test_metrics_server_starts() {
        let port = get_free_port();
        let addr: SocketAddr = format!("127.0.0.1:{}", port).parse().unwrap();
        let result = start_metrics_server(addr);
        assert!(result.is_ok(), "metrics server should start successfully");
    }
}
