//! Tests for coherency directory, fault handler, and thrashing detection.

#[cfg(test)]
mod coherency_tests {
    use crate::memory::coherency::*;

    #[test]
    fn coherency_read_from_invalid() {
        let dir = CoherencyDirectory::new(0);
        // First read of a page that doesn't exist yet -> should become Shared
        let resp = dir.handle_read_request(100, 1);
        match resp {
            ReadResponse::Granted { data_source } => {
                // Home node is the one that provides data for a fresh page
                assert_eq!(data_source, 0, "data should come from home node (0)");
            }
            _ => panic!("expected Granted for first read from invalid state"),
        }
        assert_eq!(dir.get_state(100), Some(PageState::Shared));
        assert!(dir.get_sharers(100).contains(&1));
    }

    #[test]
    fn coherency_read_shared() {
        let dir = CoherencyDirectory::new(0);
        // First reader
        dir.handle_read_request(200, 1);
        // Second reader -> added to sharers
        let resp = dir.handle_read_request(200, 2);
        match resp {
            ReadResponse::Granted { data_source } => {
                // Can read from home or any existing sharer
                assert!(data_source == 0 || data_source == 1);
            }
            _ => panic!("expected Granted for second read"),
        }
        let sharers = dir.get_sharers(200);
        assert!(sharers.contains(&1));
        assert!(sharers.contains(&2));
        assert_eq!(sharers.len(), 2);
    }

    #[test]
    fn coherency_read_shared_already_sharer() {
        let dir = CoherencyDirectory::new(0);
        dir.handle_read_request(200, 1);
        // Same node requests read again -> AlreadyShared
        let resp = dir.handle_read_request(200, 1);
        assert!(matches!(resp, ReadResponse::AlreadyShared));
    }

    #[test]
    fn coherency_write_invalidates_sharers() {
        let dir = CoherencyDirectory::new(0);
        // Two readers
        dir.handle_read_request(300, 1);
        dir.handle_read_request(300, 2);
        // Node 3 requests write -> should invalidate both sharers
        let resp = dir.handle_write_request(300, 3);
        match resp {
            WriteResponse::Granted { invalidated } => {
                assert!(invalidated.contains(&1));
                assert!(invalidated.contains(&2));
                assert_eq!(invalidated.len(), 2);
            }
            _ => panic!("expected Granted with invalidation"),
        }
        assert_eq!(dir.get_state(300), Some(PageState::Exclusive));
        assert!(dir.get_sharers(300).is_empty());

        let stats = dir.stats();
        assert!(stats.invalidations_sent >= 2);
    }

    #[test]
    fn coherency_exclusive_to_shared() {
        let dir = CoherencyDirectory::new(0);
        // Node 1 takes exclusive ownership
        dir.handle_write_request(400, 1);
        assert_eq!(dir.get_state(400), Some(PageState::Exclusive));

        // Node 2 requests read -> downgrade owner to shared
        let resp = dir.handle_read_request(400, 2);
        match resp {
            ReadResponse::Granted { data_source } => {
                assert_eq!(data_source, 1, "data should come from exclusive owner");
            }
            _ => panic!("expected Granted when downgrading exclusive to shared"),
        }
        assert_eq!(dir.get_state(400), Some(PageState::Shared));
        let sharers = dir.get_sharers(400);
        assert!(sharers.contains(&1), "previous owner should be sharer");
        assert!(sharers.contains(&2), "new reader should be sharer");

        let stats = dir.stats();
        assert!(stats.downgrades >= 1);
    }

    #[test]
    fn coherency_evict_cleans_up() {
        let dir = CoherencyDirectory::new(0);
        dir.handle_read_request(500, 1);
        dir.handle_read_request(500, 2);
        assert_eq!(dir.get_sharers(500).len(), 2);

        // Node 1 evicts
        dir.handle_evict_notify(500, 1);
        let sharers = dir.get_sharers(500);
        assert!(!sharers.contains(&1));
        assert!(sharers.contains(&2));

        // Node 2 evicts -> page becomes invalid
        dir.handle_evict_notify(500, 2);
        assert_eq!(dir.get_state(500), Some(PageState::Invalid));
    }

    #[test]
    fn coherency_evict_exclusive_owner() {
        let dir = CoherencyDirectory::new(0);
        dir.handle_write_request(550, 1);
        assert_eq!(dir.get_state(550), Some(PageState::Exclusive));

        dir.handle_evict_notify(550, 1);
        assert_eq!(dir.get_state(550), Some(PageState::Invalid));
    }

    #[test]
    fn coherency_write_from_invalid() {
        let dir = CoherencyDirectory::new(0);
        let resp = dir.handle_write_request(600, 1);
        match resp {
            WriteResponse::Granted { invalidated } => {
                assert!(invalidated.is_empty(), "nothing to invalidate from invalid state");
            }
            _ => panic!("expected Granted for write from invalid"),
        }
        assert_eq!(dir.get_state(600), Some(PageState::Exclusive));
    }

    #[test]
    fn coherency_write_already_exclusive() {
        let dir = CoherencyDirectory::new(0);
        dir.handle_write_request(700, 1);
        // Same node requests write again -> AlreadyExclusive
        let resp = dir.handle_write_request(700, 1);
        assert!(matches!(resp, WriteResponse::AlreadyExclusive));
    }

    #[test]
    fn coherency_write_steals_exclusive() {
        let dir = CoherencyDirectory::new(0);
        dir.handle_write_request(800, 1);
        // Node 2 writes -> invalidate node 1, take exclusive
        let resp = dir.handle_write_request(800, 2);
        match resp {
            WriteResponse::Granted { invalidated } => {
                assert_eq!(invalidated, vec![1]);
            }
            _ => panic!("expected Granted"),
        }
        assert_eq!(dir.get_state(800), Some(PageState::Exclusive));
    }

    #[test]
    fn coherency_upgrade_shared_to_exclusive() {
        let dir = CoherencyDirectory::new(0);
        // Node 1 reads, then wants to write (upgrade)
        dir.handle_read_request(900, 1);
        dir.handle_read_request(900, 2);
        let resp = dir.handle_write_request(900, 1);
        match resp {
            WriteResponse::Granted { invalidated } => {
                // Node 2 should be invalidated; node 1 is the requester
                assert_eq!(invalidated, vec![2]);
            }
            _ => panic!("expected Granted for upgrade"),
        }
        assert_eq!(dir.get_state(900), Some(PageState::Exclusive));

        let stats = dir.stats();
        assert!(stats.upgrades >= 1);
    }

    #[test]
    fn coherency_stats_tracking() {
        let dir = CoherencyDirectory::new(0);
        // Generate various operations
        dir.handle_read_request(1000, 1);   // read_requests +1
        dir.handle_read_request(1000, 2);   // read_requests +1
        dir.handle_write_request(1000, 3);  // write_requests +1, invalidations_sent +2
        dir.handle_read_request(1000, 1);   // read_requests +1, downgrades +1
        dir.handle_write_request(1000, 1);  // write_requests +1, upgrades +1 (sharer -> exclusive)

        let stats = dir.stats();
        assert_eq!(stats.read_requests, 3);
        assert_eq!(stats.write_requests, 2);
        assert!(stats.invalidations_sent >= 2);
        assert!(stats.downgrades >= 1);
        assert!(stats.upgrades >= 1);
    }
}

#[cfg(test)]
mod fault_handler_tests {
    use crate::memory::fault_handler::*;

    #[test]
    fn fault_handler_request_pages() {
        let handler = FaultHandler::new(FaultConfig::default());
        let results = handler.request_pages(&[10, 20, 30]);
        assert_eq!(results.len(), 3);
        for r in &results {
            assert!(matches!(r, FaultResult::Fetching));
        }
        assert_eq!(handler.pending_count(), 3);
        assert!(handler.is_fault_pending(10));
        assert!(handler.is_fault_pending(20));
        assert!(handler.is_fault_pending(30));
    }

    #[test]
    fn fault_handler_complete() {
        let handler = FaultHandler::new(FaultConfig::default());
        handler.request_pages(&[40]);
        assert!(handler.is_fault_pending(40));

        handler.on_fault_complete(40);
        assert!(!handler.is_fault_pending(40));
        assert_eq!(handler.pending_count(), 0);

        let stats = handler.stats();
        assert_eq!(stats.total_faults, 1);
        assert_eq!(stats.faults_resolved, 1);
        assert_eq!(stats.faults_failed, 0);
    }

    #[test]
    fn fault_handler_failed() {
        let handler = FaultHandler::new(FaultConfig::default());
        handler.request_pages(&[50]);
        handler.on_fault_failed(50, "node unreachable".to_string());

        assert!(!handler.is_fault_pending(50));
        let stats = handler.stats();
        assert_eq!(stats.total_faults, 1);
        assert_eq!(stats.faults_failed, 1);
    }

    #[test]
    fn fault_handler_dedup_requests() {
        let handler = FaultHandler::new(FaultConfig::default());
        // Request same VPN twice in same batch
        let results = handler.request_pages(&[60, 60, 70]);
        // First 60 -> Fetching, second 60 -> AlreadyLocal (deduped), 70 -> Fetching
        assert!(matches!(results[0], FaultResult::Fetching));
        assert!(matches!(results[1], FaultResult::AlreadyLocal));
        assert!(matches!(results[2], FaultResult::Fetching));
        assert_eq!(handler.pending_count(), 2); // only 60 and 70

        // Request 60 again in a new call while still pending
        let results2 = handler.request_pages(&[60]);
        assert!(matches!(results2[0], FaultResult::AlreadyLocal));
        assert_eq!(handler.pending_count(), 2); // unchanged
    }

    #[test]
    fn fault_handler_default_config() {
        let config = FaultConfig::default();
        assert_eq!(config.max_concurrent_faults, 64);
        assert_eq!(config.max_retry_count, 3);
    }

    #[test]
    fn fault_handler_capacity_limit() {
        let config = FaultConfig {
            max_concurrent_faults: 2,
            max_retry_count: 3,
        };
        let handler = FaultHandler::new(config);
        let results = handler.request_pages(&[80, 90, 100]);
        // Only 2 should be Fetching, third should fail
        let fetching_count = results.iter().filter(|r| matches!(r, FaultResult::Fetching)).count();
        let failed_count = results.iter().filter(|r| matches!(r, FaultResult::Failed(_))).count();
        assert_eq!(fetching_count, 2);
        assert_eq!(failed_count, 1);
    }
}

#[cfg(test)]
mod thrashing_tests {
    use crate::memory::fault_handler::*;

    #[test]
    fn thrashing_level_escalation() {
        let detector = ThrashingDetector::new(ThrashingConfig::default());
        let vpn = 1000;

        // Below level1 threshold (5) -> None
        for _ in 0..4 {
            let level = detector.record_bounce(vpn);
            assert_eq!(level, ThrashingLevel::None);
        }
        // 5th bounce -> Level1
        assert_eq!(detector.record_bounce(vpn), ThrashingLevel::Level1);

        // Continue to level2 threshold (10)
        for _ in 6..10 {
            let level = detector.record_bounce(vpn);
            assert_eq!(level, ThrashingLevel::Level1, "bounces 6-9 should be Level1");
        }
        // 10th bounce -> Level2
        assert_eq!(detector.record_bounce(vpn), ThrashingLevel::Level2);

        // Continue to level3 threshold (20)
        for _ in 11..20 {
            let level = detector.record_bounce(vpn);
            assert_eq!(level, ThrashingLevel::Level2, "bounces 11-19 should be Level2");
        }
        // 20th bounce -> Level3
        assert_eq!(detector.record_bounce(vpn), ThrashingLevel::Level3);
    }

    #[test]
    fn thrashing_window_reset() {
        let config = ThrashingConfig {
            window_ms: 1, // 1ms window for fast test
            level1_threshold: 5,
            level2_threshold: 10,
            level3_threshold: 20,
            pin_duration_ms: 20_000,
        };
        let detector = ThrashingDetector::new(config);
        let vpn = 2000;

        // Record 4 bounces
        for _ in 0..4 {
            detector.record_bounce(vpn);
        }

        // Sleep past the window
        std::thread::sleep(std::time::Duration::from_millis(5));

        // Next bounce should reset the counter (window expired), so back to 1
        let level = detector.record_bounce(vpn);
        assert_eq!(level, ThrashingLevel::None);
    }

    #[test]
    fn thrashing_check_level_without_recording() {
        let detector = ThrashingDetector::new(ThrashingConfig::default());
        // Check level for unknown VPN -> None
        assert_eq!(detector.check_level(9999), ThrashingLevel::None);

        // Record some bounces and check
        for _ in 0..5 {
            detector.record_bounce(3000);
        }
        assert_eq!(detector.check_level(3000), ThrashingLevel::Level1);
    }
}
