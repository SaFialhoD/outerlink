//! Retry and reconnect configuration for OuterLink transports.
//!
//! Provides [`RetryConfig`] which controls per-request retry attempts and
//! connection-level reconnect behaviour with exponential backoff.

use std::time::Duration;

/// Configuration for retry and reconnect behaviour.
///
/// Used by the client to transparently retry failed requests on transient
/// transport errors and to reconnect when the underlying TCP connection is
/// lost.
#[derive(Debug, Clone)]
pub struct RetryConfig {
    /// Maximum number of times to retry a single request on transient error.
    /// Does NOT include the initial attempt (i.e. 3 means up to 4 total attempts).
    pub max_retries: u32,

    /// Delay sequence for per-request retries. If the number of retries
    /// exceeds the length of this vec, the last value is reused.
    pub retry_delays: Vec<Duration>,

    /// Maximum number of reconnect attempts before giving up.
    pub max_reconnect_attempts: u32,

    /// Initial delay before the first reconnect attempt.
    pub reconnect_initial_delay: Duration,

    /// Maximum delay between reconnect attempts (caps exponential growth).
    pub reconnect_max_delay: Duration,
}

impl Default for RetryConfig {
    fn default() -> Self {
        Self {
            max_retries: 3,
            retry_delays: vec![
                Duration::from_millis(100),
                Duration::from_millis(500),
                Duration::from_millis(1000),
            ],
            max_reconnect_attempts: 5,
            reconnect_initial_delay: Duration::from_secs(1),
            reconnect_max_delay: Duration::from_secs(30),
        }
    }
}

impl RetryConfig {
    /// Get the delay for retry attempt `n` (0-indexed).
    ///
    /// Returns the n-th element of `retry_delays`, or the last element if
    /// `n` exceeds the list length. Returns `Duration::ZERO` if
    /// `retry_delays` is empty.
    pub fn retry_delay(&self, n: u32) -> Duration {
        if self.retry_delays.is_empty() {
            return Duration::ZERO;
        }
        let idx = (n as usize).min(self.retry_delays.len() - 1);
        self.retry_delays[idx]
    }

    /// Compute the reconnect delay for attempt `n` (0-indexed) using
    /// exponential backoff capped at `reconnect_max_delay`.
    pub fn reconnect_delay(&self, n: u32) -> Duration {
        let base = self.reconnect_initial_delay.as_millis() as u64;
        let multiplier = 1u64.checked_shl(n).unwrap_or(u64::MAX);
        let delay_ms = base.saturating_mul(multiplier);
        let max_ms = self.reconnect_max_delay.as_millis() as u64;
        Duration::from_millis(delay_ms.min(max_ms))
    }

    /// Create a config that disables all retries and reconnects.
    /// Useful for tests that want immediate failure.
    pub fn no_retry() -> Self {
        Self {
            max_retries: 0,
            retry_delays: vec![Duration::ZERO],
            max_reconnect_attempts: 0,
            reconnect_initial_delay: Duration::ZERO,
            reconnect_max_delay: Duration::ZERO,
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config_values() {
        let cfg = RetryConfig::default();
        assert_eq!(cfg.max_retries, 3);
        assert_eq!(cfg.retry_delays.len(), 3);
        assert_eq!(cfg.max_reconnect_attempts, 5);
        assert_eq!(cfg.reconnect_initial_delay, Duration::from_secs(1));
        assert_eq!(cfg.reconnect_max_delay, Duration::from_secs(30));
    }

    #[test]
    fn test_retry_delay_returns_correct_values() {
        let cfg = RetryConfig::default();
        assert_eq!(cfg.retry_delay(0), Duration::from_millis(100));
        assert_eq!(cfg.retry_delay(1), Duration::from_millis(500));
        assert_eq!(cfg.retry_delay(2), Duration::from_millis(1000));
        // Beyond list length: clamps to last
        assert_eq!(cfg.retry_delay(10), Duration::from_millis(1000));
    }

    #[test]
    fn test_reconnect_delay_exponential_backoff() {
        let cfg = RetryConfig::default();
        // base = 1000ms
        assert_eq!(cfg.reconnect_delay(0), Duration::from_secs(1));
        assert_eq!(cfg.reconnect_delay(1), Duration::from_secs(2));
        assert_eq!(cfg.reconnect_delay(2), Duration::from_secs(4));
        assert_eq!(cfg.reconnect_delay(3), Duration::from_secs(8));
        assert_eq!(cfg.reconnect_delay(4), Duration::from_secs(16));
    }

    #[test]
    fn test_reconnect_delay_capped_at_max() {
        let cfg = RetryConfig::default();
        // 2^5 * 1000 = 32000ms > 30000ms max
        assert_eq!(cfg.reconnect_delay(5), Duration::from_secs(30));
        assert_eq!(cfg.reconnect_delay(20), Duration::from_secs(30));
    }

    #[test]
    fn test_no_retry_config() {
        let cfg = RetryConfig::no_retry();
        assert_eq!(cfg.max_retries, 0);
        assert_eq!(cfg.max_reconnect_attempts, 0);
    }

    #[test]
    fn test_retry_delay_empty_vec_returns_zero() {
        let cfg = RetryConfig {
            retry_delays: vec![],
            ..RetryConfig::default()
        };
        // Should not panic, should return Duration::ZERO
        assert_eq!(cfg.retry_delay(0), Duration::from_millis(0));
        assert_eq!(cfg.retry_delay(5), Duration::from_millis(0));
    }

    #[test]
    fn test_is_retryable_on_errors() {
        use crate::error::OuterLinkError;
        use crate::cuda_types::CuResult;

        // Retryable errors
        assert!(OuterLinkError::Transport("broken pipe".into()).is_retryable());
        assert!(OuterLinkError::Connection("refused".into()).is_retryable());
        assert!(OuterLinkError::ConnectionClosed.is_retryable());
        assert!(OuterLinkError::Timeout(5000).is_retryable());
        assert!(OuterLinkError::Io(std::io::Error::new(
            std::io::ErrorKind::BrokenPipe, "test"
        )).is_retryable());

        // Non-retryable errors
        assert!(!OuterLinkError::Cuda(CuResult::OutOfMemory).is_retryable());
        assert!(!OuterLinkError::Protocol("bad magic".into()).is_retryable());
        assert!(!OuterLinkError::HandleNotFound(0x42).is_retryable());
        assert!(!OuterLinkError::NotReady("starting".into()).is_retryable());
        assert!(!OuterLinkError::Config("bad".into()).is_retryable());
    }
}
