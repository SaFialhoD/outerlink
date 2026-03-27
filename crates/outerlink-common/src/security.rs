//! Security and authentication types.
//!
//! Provides certificate generation (cluster CA + node certs via rcgen),
//! JWT token creation/validation (via jsonwebtoken), and quota definitions.
//! This module contains only crypto operations and types -- no networking code.

use std::time::{Duration, SystemTime, UNIX_EPOCH};

// ===== Certificate types =====

/// Certificate authority configuration.
#[derive(Debug, Clone)]
pub struct CaConfig {
    /// Organization name in the CA certificate.
    pub org_name: String,
    /// CA certificate validity duration.
    pub validity: Duration,
}

impl Default for CaConfig {
    fn default() -> Self {
        Self {
            org_name: "OutterLink Cluster".into(),
            validity: Duration::from_secs(365 * 24 * 3600), // 1 year
        }
    }
}

/// Node certificate configuration.
#[derive(Debug, Clone)]
pub struct NodeCertConfig {
    /// Node hostname.
    pub hostname: String,
    /// Node UUID.
    pub node_id: String,
    /// GPU UUIDs on this node (added as SANs).
    pub gpu_uuids: Vec<String>,
    /// Certificate validity duration.
    pub validity: Duration,
}

impl NodeCertConfig {
    pub fn new(hostname: String, node_id: String) -> Self {
        Self {
            hostname,
            node_id,
            gpu_uuids: Vec::new(),
            validity: Duration::from_secs(24 * 3600), // 24 hours
        }
    }
}

/// Generated certificate pair (PEM-encoded).
#[derive(Debug, Clone)]
pub struct CertPair {
    /// PEM-encoded certificate.
    pub cert_pem: String,
    /// PEM-encoded private key.
    pub key_pem: String,
}

/// Generate a self-signed CA certificate.
pub fn generate_ca(config: &CaConfig) -> Result<CertPair, String> {
    use rcgen::{BasicConstraints, CertificateParams, DnType, IsCa, KeyPair, KeyUsagePurpose};

    let mut params = CertificateParams::default();
    params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    params.key_usages = vec![
        KeyUsagePurpose::KeyCertSign,
        KeyUsagePurpose::CrlSign,
    ];

    // Set distinguished name
    params.distinguished_name = rcgen::DistinguishedName::new();
    params.distinguished_name.push(DnType::OrganizationName, &config.org_name);
    params.distinguished_name.push(DnType::CommonName, "OutterLink Cluster CA");

    // Set validity window
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let end_secs = now_secs + config.validity.as_secs();
    params.not_before = offset_from_epoch(now_secs.saturating_sub(3600)); // 1h grace
    params.not_after = offset_from_epoch(end_secs);

    let key_pair = KeyPair::generate().map_err(|e| e.to_string())?;
    let cert = params.self_signed(&key_pair).map_err(|e| e.to_string())?;

    Ok(CertPair {
        cert_pem: cert.pem(),
        key_pem: key_pair.serialize_pem(),
    })
}

/// Generate a node certificate signed by the CA.
pub fn generate_node_cert(
    ca_cert_pem: &str,
    ca_key_pem: &str,
    config: &NodeCertConfig,
) -> Result<CertPair, String> {
    use rcgen::{CertificateParams, DnType, Issuer, KeyPair, SanType};

    // Reconstruct CA issuer from PEM
    let ca_key = KeyPair::from_pem(ca_key_pem).map_err(|e| e.to_string())?;
    let ca_issuer =
        Issuer::from_ca_cert_pem(ca_cert_pem, ca_key).map_err(|e| e.to_string())?;

    // Build SANs: hostname as DNS + node_id and GPU UUIDs as URIs
    let mut params =
        CertificateParams::new(vec![config.hostname.clone()]).map_err(|e| e.to_string())?;

    // Embed node_id as a URI SAN for machine identity
    let node_uri = format!("urn:outerlink:node:{}", config.node_id);
    let node_ia5: rcgen::string::Ia5String = node_uri.try_into().map_err(|e| format!("{e:?}"))?;
    params.subject_alt_names.push(SanType::URI(node_ia5));

    // Embed GPU UUIDs as URI SANs (not DNS — they're not hostnames)
    for uuid in &config.gpu_uuids {
        let gpu_uri = format!("urn:outerlink:gpu:{uuid}");
        let gpu_ia5: rcgen::string::Ia5String = gpu_uri.try_into().map_err(|e| format!("{e:?}"))?;
        params.subject_alt_names.push(SanType::URI(gpu_ia5));
    }

    params.distinguished_name = rcgen::DistinguishedName::new();
    params.distinguished_name.push(DnType::CommonName, &config.hostname);
    params
        .distinguished_name
        .push(DnType::OrganizationName, "OutterLink Node");

    // Set validity window
    let now_secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let end_secs = now_secs + config.validity.as_secs();
    params.not_before = offset_from_epoch(now_secs.saturating_sub(60)); // 1min grace
    params.not_after = offset_from_epoch(end_secs);

    let node_key = KeyPair::generate().map_err(|e| e.to_string())?;
    let node_cert = params
        .signed_by(&node_key, &ca_issuer)
        .map_err(|e| e.to_string())?;

    Ok(CertPair {
        cert_pem: node_cert.pem(),
        key_pem: node_key.serialize_pem(),
    })
}

/// Convert seconds-since-epoch to `time::OffsetDateTime` (used by rcgen).
fn offset_from_epoch(secs: u64) -> time::OffsetDateTime {
    time::OffsetDateTime::from_unix_timestamp(secs as i64)
        .unwrap_or_else(|_| rcgen::date_time_ymd(2030, 1, 1))
}

// ===== JWT types =====

/// JWT claims for OutterLink access tokens.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct OutterLinkClaims {
    /// Subject (node ID or user ID).
    pub sub: String,
    /// Issued at (Unix timestamp).
    pub iat: u64,
    /// Expiration (Unix timestamp).
    pub exp: u64,
    /// Token ID for replay prevention.
    pub jti: String,
    /// Allowed GPU indices.
    pub gpu_access: Vec<u32>,
    /// Maximum VRAM allocation in bytes (0 = unlimited).
    pub max_vram: u64,
}

impl OutterLinkClaims {
    /// Create new claims with a given expiry duration.
    pub fn new(subject: String, gpu_access: Vec<u32>, max_vram: u64, expiry: Duration) -> Self {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        Self {
            sub: subject,
            iat: now,
            exp: now + expiry.as_secs(),
            jti: uuid::Uuid::new_v4().to_string(),
            gpu_access,
            max_vram,
        }
    }

    /// Check if the token grants access to a specific GPU.
    pub fn has_gpu_access(&self, gpu_index: u32) -> bool {
        self.gpu_access.is_empty() || self.gpu_access.contains(&gpu_index)
    }

    /// Check if the token has expired.
    pub fn is_expired(&self) -> bool {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();
        now >= self.exp
    }
}

/// Create a signed JWT token.
pub fn create_token(claims: &OutterLinkClaims, secret: &[u8]) -> Result<String, String> {
    let key = jsonwebtoken::EncodingKey::from_secret(secret);
    let header = jsonwebtoken::Header::default();
    jsonwebtoken::encode(&header, claims, &key).map_err(|e| e.to_string())
}

/// Validate and decode a JWT token.
pub fn validate_token(token: &str, secret: &[u8]) -> Result<OutterLinkClaims, String> {
    let key = jsonwebtoken::DecodingKey::from_secret(secret);
    let mut validation = jsonwebtoken::Validation::default();
    validation.validate_exp = true;

    let data = jsonwebtoken::decode::<OutterLinkClaims>(token, &key, &validation)
        .map_err(|e| e.to_string())?;
    Ok(data.claims)
}

// ===== Quota types =====

/// Quota configuration for a node or user.
#[derive(Debug, Clone)]
pub struct QuotaConfig {
    /// Maximum CUDA API calls per second.
    pub max_cuda_calls_per_sec: u32,
    /// Maximum concurrent VRAM allocations in bytes.
    pub max_vram_bytes: u64,
    /// Maximum concurrent connections.
    pub max_connections: u32,
    /// Maximum transfer bandwidth in bytes/sec (0 = unlimited).
    pub max_bandwidth_bytes_per_sec: u64,
}

impl Default for QuotaConfig {
    fn default() -> Self {
        Self {
            max_cuda_calls_per_sec: 10_000,
            max_vram_bytes: 24 * 1024 * 1024 * 1024, // 24 GB
            max_connections: 16,
            max_bandwidth_bytes_per_sec: 0, // unlimited
        }
    }
}

/// Audit event types for structured logging.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuditEvent {
    /// Node joined the cluster.
    NodeJoined,
    /// Node left the cluster.
    NodeLeft,
    /// Client authenticated.
    ClientAuth,
    /// Client authentication failed.
    ClientAuthFailed,
    /// GPU allocated to client.
    GpuAllocated,
    /// GPU released by client.
    GpuReleased,
    /// Quota exceeded.
    QuotaExceeded,
    /// Certificate rotated.
    CertRotated,
}

impl std::fmt::Display for AuditEvent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NodeJoined => write!(f, "node_joined"),
            Self::NodeLeft => write!(f, "node_left"),
            Self::ClientAuth => write!(f, "client_auth"),
            Self::ClientAuthFailed => write!(f, "client_auth_failed"),
            Self::GpuAllocated => write!(f, "gpu_allocated"),
            Self::GpuReleased => write!(f, "gpu_released"),
            Self::QuotaExceeded => write!(f, "quota_exceeded"),
            Self::CertRotated => write!(f, "cert_rotated"),
        }
    }
}
