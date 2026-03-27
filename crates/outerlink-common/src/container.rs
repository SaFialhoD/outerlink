//! Container and Docker CDI integration types for OuterLink.
//!
//! Provides pure data types and config generation for container GPU access:
//! - CDI (Container Device Interface) device specifications
//! - Per-container VRAM limits and compute allocation
//! - LD_PRELOAD injection configuration
//! - OCI hook and mount definitions
//!
//! This module contains NO Docker API calls -- only types and config builders.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Core config: what GPU resources a container gets
// ---------------------------------------------------------------------------

/// Configuration for a single container's GPU access through OuterLink.
///
/// Defines which GPUs the container can see, optional VRAM/compute limits,
/// and any extra environment variables to inject.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContainerGpuConfig {
    /// Unique container identifier (e.g. Docker container ID or name).
    pub container_id: String,

    /// Optional VRAM limit in bytes. `None` means no limit (full VRAM available).
    pub vram_limit_bytes: Option<u64>,

    /// Optional compute percentage (0.0 - 100.0). `None` means no throttle.
    pub compute_percent: Option<f64>,

    /// GPU indices this container is allowed to use.
    pub gpu_indices: Vec<u32>,

    /// Extra environment variables to inject into the container.
    pub env_vars: HashMap<String, String>,
}

impl ContainerGpuConfig {
    /// Create a new config with only a container ID and defaults.
    pub fn new(container_id: impl Into<String>) -> Self {
        Self {
            container_id: container_id.into(),
            vram_limit_bytes: None,
            compute_percent: None,
            gpu_indices: Vec::new(),
            env_vars: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// CDI device spec
// ---------------------------------------------------------------------------

/// A single CDI device entry representing one OuterLink virtual GPU.
///
/// CDI device names follow the format `<vendor>/<class>=<name>`,
/// e.g. `outerlink/gpu=gpu0`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CdiDeviceSpec {
    /// Vendor identifier. Default: `"outerlink"`.
    pub vendor: String,

    /// Device class. Default: `"gpu"`.
    pub class: String,

    /// Device name (unique within this vendor/class). E.g. `"gpu0"`.
    pub name: String,

    /// Arbitrary annotations attached to this device.
    pub annotations: HashMap<String, String>,
}

impl CdiDeviceSpec {
    /// Create a new CDI device spec with default vendor/class.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            vendor: "outerlink".to_string(),
            class: "gpu".to_string(),
            name: name.into(),
            annotations: HashMap::new(),
        }
    }

    /// Returns the fully-qualified CDI device name.
    ///
    /// Format: `<vendor>/<class>=<name>`, e.g. `"outerlink/gpu=gpu0"`.
    pub fn device_name(&self) -> String {
        format!("{}/{}={}", self.vendor, self.class, self.name)
    }
}

// ---------------------------------------------------------------------------
// Full CDI spec
// ---------------------------------------------------------------------------

/// A complete CDI specification file for OuterLink GPUs.
///
/// Serialises to the JSON format expected by CDI-aware container runtimes
/// (containerd, CRI-O).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CdiSpec {
    /// CDI spec version. Default: `"0.8.0"`.
    pub version: String,

    /// Kind identifier. Default: `"outerlink/gpu"`.
    pub kind: String,

    /// Individual device entries.
    pub devices: Vec<CdiDeviceSpec>,

    /// Global container edits applied when any device from this spec is requested.
    pub container_edits: ContainerEdits,
}

impl CdiSpec {
    /// Create a new CDI spec with defaults and the given devices/edits.
    pub fn new(devices: Vec<CdiDeviceSpec>, container_edits: ContainerEdits) -> Self {
        Self {
            version: "0.8.0".to_string(),
            kind: "outerlink/gpu".to_string(),
            devices,
            container_edits,
        }
    }
}

// ---------------------------------------------------------------------------
// Container edits (injections)
// ---------------------------------------------------------------------------

/// Edits applied to a container when an OuterLink CDI device is requested.
///
/// These map directly to the CDI `containerEdits` object.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContainerEdits {
    /// Environment variables to set (in `KEY=value` format).
    pub env: Vec<String>,

    /// OCI lifecycle hooks to install.
    pub hooks: Vec<ContainerHook>,

    /// Bind mounts to add.
    pub mounts: Vec<Mount>,
}

impl ContainerEdits {
    /// Create empty container edits.
    pub fn new() -> Self {
        Self {
            env: Vec::new(),
            hooks: Vec::new(),
            mounts: Vec::new(),
        }
    }
}

impl Default for ContainerEdits {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// OCI hook
// ---------------------------------------------------------------------------

/// An OCI lifecycle hook that runs at container creation time.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ContainerHook {
    /// OCI hook point name. Default: `"createRuntime"`.
    pub hook_name: String,

    /// Path to the hook binary on the host.
    pub path: String,

    /// Arguments passed to the hook binary.
    pub args: Vec<String>,
}

impl ContainerHook {
    /// Create a new hook with the default `createRuntime` hook point.
    pub fn new(path: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            hook_name: "createRuntime".to_string(),
            path: path.into(),
            args,
        }
    }
}

// ---------------------------------------------------------------------------
// Bind mount
// ---------------------------------------------------------------------------

/// A bind mount injected into the container.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Mount {
    /// Source path on the host.
    pub host_path: String,

    /// Destination path inside the container.
    pub container_path: String,

    /// Mount options. Default: `["ro", "nosuid", "nodev"]`.
    pub options: Vec<String>,
}

impl Mount {
    /// Create a new mount with default options (read-only, nosuid, nodev).
    pub fn new(host_path: impl Into<String>, container_path: impl Into<String>) -> Self {
        Self {
            host_path: host_path.into(),
            container_path: container_path.into(),
            options: vec![
                "ro".to_string(),
                "nosuid".to_string(),
                "nodev".to_string(),
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// Injection method
// ---------------------------------------------------------------------------

/// How OuterLink is injected into a container's runtime environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InjectionMethod {
    /// Classic LD_PRELOAD interception -- the OuterLink client .so is preloaded.
    LdPreload,

    /// CDI plugin -- a CDI-aware runtime (containerd, CRI-O) provisions the device.
    CdiPlugin,

    /// OCI runtime hook -- a createRuntime hook sets up the OuterLink environment.
    RuntimeHook,
}

// ---------------------------------------------------------------------------
// Builder functions
// ---------------------------------------------------------------------------

/// Build the `LD_PRELOAD` environment variable value for OuterLink.
///
/// If `existing_ld_preload` is non-empty the OuterLink library is appended
/// (colon-separated) so existing preloads are preserved.
pub fn build_ld_preload_env(library_path: &str, existing_ld_preload: Option<&str>) -> String {
    match existing_ld_preload {
        Some(existing) if !existing.is_empty() => {
            format!("{}:{}", existing, library_path)
        }
        _ => library_path.to_string(),
    }
}

/// Generate the full set of environment variables for a container.
///
/// Returns key-value pairs for:
/// - `LD_PRELOAD` (with the OuterLink client library)
/// - `OUTERLINK_SERVER` (caller must place this in `config.env_vars`)
/// - `OUTERLINK_VRAM_LIMIT` (if `vram_limit_bytes` is set)
/// - `OUTERLINK_GPU_INDICES` (comma-separated, if non-empty)
/// - Any additional entries from `config.env_vars`
pub fn build_container_env(
    config: &ContainerGpuConfig,
    library_path: &str,
) -> Vec<(String, String)> {
    let mut env: Vec<(String, String)> = Vec::new();

    // LD_PRELOAD -- always set
    let existing = config.env_vars.get("LD_PRELOAD").map(|s| s.as_str());
    env.push((
        "LD_PRELOAD".to_string(),
        build_ld_preload_env(library_path, existing),
    ));

    // VRAM limit
    if let Some(limit) = config.vram_limit_bytes {
        env.push(("OUTERLINK_VRAM_LIMIT".to_string(), limit.to_string()));
    }

    // Compute percent
    if let Some(pct) = config.compute_percent {
        env.push(("OUTERLINK_COMPUTE_PERCENT".to_string(), pct.to_string()));
    }

    // GPU indices
    if !config.gpu_indices.is_empty() {
        let indices: Vec<String> = config.gpu_indices.iter().map(|i| i.to_string()).collect();
        env.push(("OUTERLINK_GPU_INDICES".to_string(), indices.join(",")));
    }

    // Pass through remaining env_vars, skipping keys already handled structurally
    const MANAGED_KEYS: &[&str] = &[
        "LD_PRELOAD",
        "OUTERLINK_VRAM_LIMIT",
        "OUTERLINK_COMPUTE_PERCENT",
        "OUTERLINK_GPU_INDICES",
    ];
    for (k, v) in &config.env_vars {
        if !MANAGED_KEYS.contains(&k.as_str()) {
            env.push((k.clone(), v.clone()));
        }
    }

    env
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- ContainerGpuConfig -------------------------------------------------

    #[test]
    fn container_gpu_config_new_defaults() {
        let cfg = ContainerGpuConfig::new("abc123");
        assert_eq!(cfg.container_id, "abc123");
        assert_eq!(cfg.vram_limit_bytes, None);
        assert_eq!(cfg.compute_percent, None);
        assert!(cfg.gpu_indices.is_empty());
        assert!(cfg.env_vars.is_empty());
    }

    #[test]
    fn container_gpu_config_with_limits() {
        let cfg = ContainerGpuConfig {
            container_id: "ctr1".to_string(),
            vram_limit_bytes: Some(4 * 1024 * 1024 * 1024), // 4 GiB
            compute_percent: Some(50.0),
            gpu_indices: vec![0, 2],
            env_vars: HashMap::new(),
        };
        assert_eq!(cfg.vram_limit_bytes, Some(4_294_967_296));
        assert_eq!(cfg.compute_percent, Some(50.0));
        assert_eq!(cfg.gpu_indices, vec![0, 2]);
    }

    #[test]
    fn container_gpu_config_serde_roundtrip() {
        let cfg = ContainerGpuConfig::new("round");
        let json = serde_json::to_string(&cfg).unwrap();
        let back: ContainerGpuConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(cfg, back);
    }

    // -- CdiDeviceSpec ------------------------------------------------------

    #[test]
    fn cdi_device_spec_defaults() {
        let dev = CdiDeviceSpec::new("gpu0");
        assert_eq!(dev.vendor, "outerlink");
        assert_eq!(dev.class, "gpu");
        assert_eq!(dev.name, "gpu0");
        assert!(dev.annotations.is_empty());
    }

    #[test]
    fn cdi_device_name_format() {
        let dev = CdiDeviceSpec::new("gpu0");
        assert_eq!(dev.device_name(), "outerlink/gpu=gpu0");
    }

    #[test]
    fn cdi_device_name_custom_vendor() {
        let dev = CdiDeviceSpec {
            vendor: "custom".to_string(),
            class: "accel".to_string(),
            name: "dev1".to_string(),
            annotations: HashMap::new(),
        };
        assert_eq!(dev.device_name(), "custom/accel=dev1");
    }

    #[test]
    fn cdi_device_spec_with_annotations() {
        let mut dev = CdiDeviceSpec::new("gpu1");
        dev.annotations
            .insert("pci-slot".to_string(), "00:1e.0".to_string());
        assert_eq!(
            dev.annotations.get("pci-slot"),
            Some(&"00:1e.0".to_string())
        );
    }

    #[test]
    fn cdi_device_spec_serde_roundtrip() {
        let dev = CdiDeviceSpec::new("gpu2");
        let json = serde_json::to_string(&dev).unwrap();
        let back: CdiDeviceSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(dev, back);
    }

    // -- CdiSpec ------------------------------------------------------------

    #[test]
    fn cdi_spec_defaults() {
        let spec = CdiSpec::new(vec![], ContainerEdits::new());
        assert_eq!(spec.version, "0.8.0");
        assert_eq!(spec.kind, "outerlink/gpu");
        assert!(spec.devices.is_empty());
    }

    #[test]
    fn cdi_spec_with_devices() {
        let devices = vec![CdiDeviceSpec::new("gpu0"), CdiDeviceSpec::new("gpu1")];
        let spec = CdiSpec::new(devices, ContainerEdits::new());
        assert_eq!(spec.devices.len(), 2);
        assert_eq!(spec.devices[0].device_name(), "outerlink/gpu=gpu0");
        assert_eq!(spec.devices[1].device_name(), "outerlink/gpu=gpu1");
    }

    #[test]
    fn cdi_spec_serde_roundtrip() {
        let spec = CdiSpec::new(
            vec![CdiDeviceSpec::new("gpu0")],
            ContainerEdits {
                env: vec!["FOO=bar".to_string()],
                hooks: vec![],
                mounts: vec![],
            },
        );
        let json = serde_json::to_string(&spec).unwrap();
        let back: CdiSpec = serde_json::from_str(&json).unwrap();
        assert_eq!(spec, back);
    }

    // -- ContainerEdits -----------------------------------------------------

    #[test]
    fn container_edits_default_is_empty() {
        let edits = ContainerEdits::default();
        assert!(edits.env.is_empty());
        assert!(edits.hooks.is_empty());
        assert!(edits.mounts.is_empty());
    }

    #[test]
    fn container_edits_with_env_and_mounts() {
        let edits = ContainerEdits {
            env: vec!["LD_PRELOAD=/usr/lib/outerlink.so".to_string()],
            hooks: vec![],
            mounts: vec![Mount::new("/usr/lib/outerlink.so", "/usr/lib/outerlink.so")],
        };
        assert_eq!(edits.env.len(), 1);
        assert_eq!(edits.mounts.len(), 1);
    }

    // -- ContainerHook ------------------------------------------------------

    #[test]
    fn container_hook_defaults() {
        let hook = ContainerHook::new("/usr/bin/outerlink-hook", vec!["setup".to_string()]);
        assert_eq!(hook.hook_name, "createRuntime");
        assert_eq!(hook.path, "/usr/bin/outerlink-hook");
        assert_eq!(hook.args, vec!["setup"]);
    }

    #[test]
    fn container_hook_custom_name() {
        let hook = ContainerHook {
            hook_name: "poststop".to_string(),
            path: "/usr/bin/cleanup".to_string(),
            args: vec![],
        };
        assert_eq!(hook.hook_name, "poststop");
    }

    #[test]
    fn container_hook_serde_roundtrip() {
        let hook = ContainerHook::new("/bin/hook", vec!["--verbose".to_string()]);
        let json = serde_json::to_string(&hook).unwrap();
        let back: ContainerHook = serde_json::from_str(&json).unwrap();
        assert_eq!(hook, back);
    }

    // -- Mount --------------------------------------------------------------

    #[test]
    fn mount_default_options() {
        let m = Mount::new("/host/lib.so", "/container/lib.so");
        assert_eq!(m.host_path, "/host/lib.so");
        assert_eq!(m.container_path, "/container/lib.so");
        assert_eq!(m.options, vec!["ro", "nosuid", "nodev"]);
    }

    #[test]
    fn mount_custom_options() {
        let m = Mount {
            host_path: "/dev/nvidia0".to_string(),
            container_path: "/dev/nvidia0".to_string(),
            options: vec!["rw".to_string()],
        };
        assert_eq!(m.options, vec!["rw"]);
    }

    #[test]
    fn mount_serde_roundtrip() {
        let m = Mount::new("/a", "/b");
        let json = serde_json::to_string(&m).unwrap();
        let back: Mount = serde_json::from_str(&json).unwrap();
        assert_eq!(m, back);
    }

    // -- InjectionMethod ----------------------------------------------------

    #[test]
    fn injection_method_variants() {
        let methods = [
            InjectionMethod::LdPreload,
            InjectionMethod::CdiPlugin,
            InjectionMethod::RuntimeHook,
        ];
        // All three are distinct
        assert_ne!(methods[0], methods[1]);
        assert_ne!(methods[1], methods[2]);
        assert_ne!(methods[0], methods[2]);
    }

    #[test]
    fn injection_method_serde_roundtrip() {
        for method in &[
            InjectionMethod::LdPreload,
            InjectionMethod::CdiPlugin,
            InjectionMethod::RuntimeHook,
        ] {
            let json = serde_json::to_string(method).unwrap();
            let back: InjectionMethod = serde_json::from_str(&json).unwrap();
            assert_eq!(*method, back);
        }
    }

    // -- build_ld_preload_env -----------------------------------------------

    #[test]
    fn ld_preload_no_existing() {
        let result = build_ld_preload_env("/usr/lib/outerlink-client.so", None);
        assert_eq!(result, "/usr/lib/outerlink-client.so");
    }

    #[test]
    fn ld_preload_empty_existing() {
        let result = build_ld_preload_env("/usr/lib/outerlink-client.so", Some(""));
        assert_eq!(result, "/usr/lib/outerlink-client.so");
    }

    #[test]
    fn ld_preload_with_existing() {
        let result =
            build_ld_preload_env("/usr/lib/outerlink-client.so", Some("/usr/lib/libfoo.so"));
        assert_eq!(
            result,
            "/usr/lib/libfoo.so:/usr/lib/outerlink-client.so"
        );
    }

    #[test]
    fn ld_preload_preserves_multiple_existing() {
        let result = build_ld_preload_env(
            "/usr/lib/outerlink-client.so",
            Some("/usr/lib/a.so:/usr/lib/b.so"),
        );
        assert_eq!(
            result,
            "/usr/lib/a.so:/usr/lib/b.so:/usr/lib/outerlink-client.so"
        );
    }

    // -- build_container_env ------------------------------------------------

    #[test]
    fn build_env_minimal_config() {
        let cfg = ContainerGpuConfig::new("ctr");
        let env = build_container_env(&cfg, "/usr/lib/outerlink-client.so");

        let map: HashMap<String, String> = env.into_iter().collect();
        assert_eq!(
            map.get("LD_PRELOAD"),
            Some(&"/usr/lib/outerlink-client.so".to_string())
        );
        // No VRAM limit, no compute, no indices
        assert!(!map.contains_key("OUTERLINK_VRAM_LIMIT"));
        assert!(!map.contains_key("OUTERLINK_COMPUTE_PERCENT"));
        assert!(!map.contains_key("OUTERLINK_GPU_INDICES"));
    }

    #[test]
    fn build_env_with_vram_limit() {
        let cfg = ContainerGpuConfig {
            container_id: "x".to_string(),
            vram_limit_bytes: Some(2_147_483_648),
            compute_percent: None,
            gpu_indices: vec![],
            env_vars: HashMap::new(),
        };
        let env = build_container_env(&cfg, "/lib/ol.so");
        let map: HashMap<String, String> = env.into_iter().collect();
        assert_eq!(
            map.get("OUTERLINK_VRAM_LIMIT"),
            Some(&"2147483648".to_string())
        );
    }

    #[test]
    fn build_env_with_compute_percent() {
        let cfg = ContainerGpuConfig {
            container_id: "x".to_string(),
            vram_limit_bytes: None,
            compute_percent: Some(75.5),
            gpu_indices: vec![],
            env_vars: HashMap::new(),
        };
        let env = build_container_env(&cfg, "/lib/ol.so");
        let map: HashMap<String, String> = env.into_iter().collect();
        assert_eq!(
            map.get("OUTERLINK_COMPUTE_PERCENT"),
            Some(&"75.5".to_string())
        );
    }

    #[test]
    fn build_env_with_gpu_indices() {
        let cfg = ContainerGpuConfig {
            container_id: "x".to_string(),
            vram_limit_bytes: None,
            compute_percent: None,
            gpu_indices: vec![0, 1, 3],
            env_vars: HashMap::new(),
        };
        let env = build_container_env(&cfg, "/lib/ol.so");
        let map: HashMap<String, String> = env.into_iter().collect();
        assert_eq!(
            map.get("OUTERLINK_GPU_INDICES"),
            Some(&"0,1,3".to_string())
        );
    }

    #[test]
    fn build_env_passes_through_extra_vars() {
        let mut extra = HashMap::new();
        extra.insert("OUTERLINK_SERVER".to_string(), "10.0.0.1:9000".to_string());
        let cfg = ContainerGpuConfig {
            container_id: "x".to_string(),
            vram_limit_bytes: None,
            compute_percent: None,
            gpu_indices: vec![],
            env_vars: extra,
        };
        let env = build_container_env(&cfg, "/lib/ol.so");
        let map: HashMap<String, String> = env.into_iter().collect();
        assert_eq!(
            map.get("OUTERLINK_SERVER"),
            Some(&"10.0.0.1:9000".to_string())
        );
    }

    #[test]
    fn build_env_appends_to_existing_ld_preload() {
        let mut extra = HashMap::new();
        extra.insert("LD_PRELOAD".to_string(), "/usr/lib/other.so".to_string());
        let cfg = ContainerGpuConfig {
            container_id: "x".to_string(),
            vram_limit_bytes: None,
            compute_percent: None,
            gpu_indices: vec![],
            env_vars: extra,
        };
        let env = build_container_env(&cfg, "/lib/outerlink.so");
        let map: HashMap<String, String> = env.into_iter().collect();
        assert_eq!(
            map.get("LD_PRELOAD"),
            Some(&"/usr/lib/other.so:/lib/outerlink.so".to_string())
        );
    }

    #[test]
    fn build_env_full_config() {
        let mut extra = HashMap::new();
        extra.insert("OUTERLINK_SERVER".to_string(), "192.168.1.10:8080".to_string());
        let cfg = ContainerGpuConfig {
            container_id: "full".to_string(),
            vram_limit_bytes: Some(8_589_934_592),
            compute_percent: Some(100.0),
            gpu_indices: vec![0],
            env_vars: extra,
        };
        let env = build_container_env(&cfg, "/usr/lib/outerlink-client.so");
        let map: HashMap<String, String> = env.into_iter().collect();
        assert!(map.contains_key("LD_PRELOAD"));
        assert!(map.contains_key("OUTERLINK_VRAM_LIMIT"));
        assert!(map.contains_key("OUTERLINK_COMPUTE_PERCENT"));
        assert!(map.contains_key("OUTERLINK_GPU_INDICES"));
        assert!(map.contains_key("OUTERLINK_SERVER"));
        assert_eq!(map.len(), 5);
    }
}
