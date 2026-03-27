//! Tests for the security module: certificates, JWT, quota, audit events.

use outerlink_common::security::{
    AuditEvent, CaConfig, NodeCertConfig, OutterLinkClaims, QuotaConfig,
    create_token, generate_ca, generate_node_cert, validate_token,
};
use std::time::Duration;

// ===== CaConfig tests =====

#[test]
fn ca_config_default_org_name() {
    let config = CaConfig::default();
    assert_eq!(config.org_name, "OutterLink Cluster");
}

#[test]
fn ca_config_default_validity_one_year() {
    let config = CaConfig::default();
    assert_eq!(config.validity, Duration::from_secs(365 * 24 * 3600));
}

// ===== NodeCertConfig tests =====

#[test]
fn node_cert_config_new_sets_hostname_and_id() {
    let config = NodeCertConfig::new("gpu-node-1".into(), "node-abc-123".into());
    assert_eq!(config.hostname, "gpu-node-1");
    assert_eq!(config.node_id, "node-abc-123");
}

#[test]
fn node_cert_config_new_default_validity_24h() {
    let config = NodeCertConfig::new("host".into(), "id".into());
    assert_eq!(config.validity, Duration::from_secs(24 * 3600));
}

#[test]
fn node_cert_config_new_empty_gpu_uuids() {
    let config = NodeCertConfig::new("host".into(), "id".into());
    assert!(config.gpu_uuids.is_empty());
}

// ===== generate_ca tests =====

#[test]
fn generate_ca_produces_valid_pem_cert() {
    let config = CaConfig::default();
    let pair = generate_ca(&config).expect("CA generation should succeed");
    assert!(pair.cert_pem.starts_with("-----BEGIN CERTIFICATE-----"));
    assert!(pair.cert_pem.contains("-----END CERTIFICATE-----"));
}

#[test]
fn generate_ca_produces_valid_pem_key() {
    let config = CaConfig::default();
    let pair = generate_ca(&config).expect("CA generation should succeed");
    assert!(pair.key_pem.contains("PRIVATE KEY"));
}

#[test]
fn generate_ca_cert_and_key_are_different() {
    let config = CaConfig::default();
    let pair = generate_ca(&config).expect("CA generation should succeed");
    assert_ne!(pair.cert_pem, pair.key_pem);
}

// ===== generate_node_cert tests =====

#[test]
fn generate_node_cert_produces_valid_pem() {
    let ca = generate_ca(&CaConfig::default()).expect("CA gen");
    let node_config = NodeCertConfig::new("worker-1".into(), "node-001".into());
    let node_pair =
        generate_node_cert(&ca.cert_pem, &ca.key_pem, &node_config).expect("node cert gen");
    assert!(node_pair.cert_pem.starts_with("-----BEGIN CERTIFICATE-----"));
    assert!(node_pair.key_pem.contains("PRIVATE KEY"));
}

#[test]
fn generate_node_cert_different_from_ca() {
    let ca = generate_ca(&CaConfig::default()).expect("CA gen");
    let node_config = NodeCertConfig::new("worker-1".into(), "node-001".into());
    let node_pair =
        generate_node_cert(&ca.cert_pem, &ca.key_pem, &node_config).expect("node cert gen");
    assert_ne!(node_pair.cert_pem, ca.cert_pem);
    assert_ne!(node_pair.key_pem, ca.key_pem);
}

// ===== OutterLinkClaims tests =====

#[test]
fn claims_new_sets_subject_and_gpu_access() {
    let claims = OutterLinkClaims::new(
        "node-42".into(),
        vec![0, 1, 2],
        8 * 1024 * 1024 * 1024,
        Duration::from_secs(3600),
    );
    assert_eq!(claims.sub, "node-42");
    assert_eq!(claims.gpu_access, vec![0, 1, 2]);
    assert_eq!(claims.max_vram, 8 * 1024 * 1024 * 1024);
}

#[test]
fn claims_new_sets_timestamps() {
    let claims = OutterLinkClaims::new("x".into(), vec![], 0, Duration::from_secs(3600));
    assert!(claims.iat > 0);
    assert_eq!(claims.exp, claims.iat + 3600);
}

#[test]
fn claims_new_generates_unique_jti() {
    let c1 = OutterLinkClaims::new("a".into(), vec![], 0, Duration::from_secs(60));
    let c2 = OutterLinkClaims::new("a".into(), vec![], 0, Duration::from_secs(60));
    assert_ne!(c1.jti, c2.jti);
}

#[test]
fn claims_has_gpu_access_specific_gpu() {
    let claims = OutterLinkClaims::new("n".into(), vec![0, 2], 0, Duration::from_secs(60));
    assert!(claims.has_gpu_access(0));
    assert!(!claims.has_gpu_access(1));
    assert!(claims.has_gpu_access(2));
}

#[test]
fn claims_has_gpu_access_empty_means_unlimited() {
    let claims = OutterLinkClaims::new("n".into(), vec![], 0, Duration::from_secs(60));
    assert!(claims.has_gpu_access(0));
    assert!(claims.has_gpu_access(42));
    assert!(claims.has_gpu_access(u32::MAX));
}

#[test]
fn claims_is_expired_with_future_expiry() {
    let claims = OutterLinkClaims::new("n".into(), vec![], 0, Duration::from_secs(3600));
    assert!(!claims.is_expired());
}

#[test]
fn claims_is_expired_with_past_expiry() {
    let mut claims = OutterLinkClaims::new("n".into(), vec![], 0, Duration::from_secs(1));
    // Force expiry to the past
    claims.exp = 1;
    assert!(claims.is_expired());
}

// ===== JWT roundtrip tests =====

#[test]
fn create_and_validate_token_roundtrip() {
    let claims = OutterLinkClaims::new(
        "test-node".into(),
        vec![0, 1],
        4 * 1024 * 1024 * 1024,
        Duration::from_secs(3600),
    );
    let secret = b"super-secret-key-for-testing";

    let token = create_token(&claims, secret).expect("token creation");
    assert!(!token.is_empty());

    let decoded = validate_token(&token, secret).expect("token validation");
    assert_eq!(decoded.sub, "test-node");
    assert_eq!(decoded.gpu_access, vec![0, 1]);
    assert_eq!(decoded.max_vram, 4 * 1024 * 1024 * 1024);
    assert_eq!(decoded.jti, claims.jti);
}

#[test]
fn validate_token_wrong_secret_fails() {
    let claims = OutterLinkClaims::new("n".into(), vec![], 0, Duration::from_secs(3600));
    let token = create_token(&claims, b"correct-secret").expect("token creation");
    let result = validate_token(&token, b"wrong-secret");
    assert!(result.is_err());
}

#[test]
fn validate_token_expired_fails() {
    let mut claims = OutterLinkClaims::new("n".into(), vec![], 0, Duration::from_secs(1));
    // Force token to be already expired
    claims.exp = 1;
    claims.iat = 0;
    let secret = b"test-secret";
    let token = create_token(&claims, secret).expect("token creation");
    let result = validate_token(&token, secret);
    assert!(result.is_err());
}

// ===== QuotaConfig tests =====

#[test]
fn quota_config_default_values() {
    let config = QuotaConfig::default();
    assert_eq!(config.max_cuda_calls_per_sec, 10_000);
    assert_eq!(config.max_vram_bytes, 24 * 1024 * 1024 * 1024);
    assert_eq!(config.max_connections, 16);
    assert_eq!(config.max_bandwidth_bytes_per_sec, 0);
}

// ===== AuditEvent tests =====

#[test]
fn audit_event_display_node_joined() {
    assert_eq!(AuditEvent::NodeJoined.to_string(), "node_joined");
}

#[test]
fn audit_event_display_node_left() {
    assert_eq!(AuditEvent::NodeLeft.to_string(), "node_left");
}

#[test]
fn audit_event_display_client_auth() {
    assert_eq!(AuditEvent::ClientAuth.to_string(), "client_auth");
}

#[test]
fn audit_event_display_client_auth_failed() {
    assert_eq!(AuditEvent::ClientAuthFailed.to_string(), "client_auth_failed");
}

#[test]
fn audit_event_display_gpu_allocated() {
    assert_eq!(AuditEvent::GpuAllocated.to_string(), "gpu_allocated");
}

#[test]
fn audit_event_display_gpu_released() {
    assert_eq!(AuditEvent::GpuReleased.to_string(), "gpu_released");
}

#[test]
fn audit_event_display_quota_exceeded() {
    assert_eq!(AuditEvent::QuotaExceeded.to_string(), "quota_exceeded");
}

#[test]
fn audit_event_display_cert_rotated() {
    assert_eq!(AuditEvent::CertRotated.to_string(), "cert_rotated");
}

#[test]
fn audit_event_equality() {
    assert_eq!(AuditEvent::NodeJoined, AuditEvent::NodeJoined);
    assert_ne!(AuditEvent::NodeJoined, AuditEvent::NodeLeft);
}
