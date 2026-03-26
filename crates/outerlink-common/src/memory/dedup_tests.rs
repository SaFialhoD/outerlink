//! Tests for the memory deduplication subsystem.

#[cfg(test)]
mod tests {
    use crate::memory::dedup::DedupManager;
    use crate::memory::traits::DedupHook;

    /// Helper: create a page of repeated bytes.
    fn make_page(byte: u8, size: usize) -> Vec<u8> {
        vec![byte; size]
    }

    #[test]
    fn dedup_identical_pages_detected() {
        let mgr = DedupManager::with_defaults();
        let page = make_page(0xAA, 4096);

        // First page: unique, becomes canonical.
        let result1 = mgr.check_dedup(100, &page);
        assert_eq!(result1, None, "first page should be unique");

        // Second page with identical content: should match canonical.
        let result2 = mgr.check_dedup(200, &page);
        assert_eq!(result2, Some(100), "identical page should return canonical vpn");
    }

    #[test]
    fn dedup_different_pages_unique() {
        let mgr = DedupManager::with_defaults();
        let page_a = make_page(0xAA, 4096);
        let page_b = make_page(0xBB, 4096);

        let r1 = mgr.check_dedup(100, &page_a);
        let r2 = mgr.check_dedup(200, &page_b);

        assert_eq!(r1, None);
        assert_eq!(r2, None, "different content should not match");
    }

    #[test]
    fn dedup_ref_count_tracking() {
        let mgr = DedupManager::with_defaults();
        let page = make_page(0xCC, 4096);

        mgr.check_dedup(100, &page); // canonical
        mgr.check_dedup(200, &page); // ref 1
        mgr.check_dedup(300, &page); // ref 2
        mgr.check_dedup(400, &page); // ref 3

        // 1 canonical + 3 references = ref_count 4
        assert!(mgr.is_canonical(100));
        assert!(mgr.is_reference(200));
        assert!(mgr.is_reference(300));
        assert!(mgr.is_reference(400));
        assert_eq!(mgr.table_size(), 1, "only one hash entry");
    }

    #[test]
    fn dedup_notify_free_reference() {
        let mgr = DedupManager::with_defaults();
        let page = make_page(0xDD, 4096);

        mgr.check_dedup(100, &page);
        mgr.check_dedup(200, &page);
        mgr.check_dedup(300, &page);

        // Free a reference.
        mgr.notify_free(200);

        assert!(mgr.is_canonical(100), "canonical unchanged");
        assert!(!mgr.is_reference(200), "freed ref should be gone");
        assert!(mgr.is_reference(300), "other ref still present");
        assert_eq!(mgr.table_size(), 1);
    }

    #[test]
    fn dedup_notify_free_canonical_promotes() {
        let mgr = DedupManager::with_defaults();
        let page = make_page(0xEE, 4096);

        mgr.check_dedup(100, &page); // canonical
        mgr.check_dedup(200, &page); // ref 1
        mgr.check_dedup(300, &page); // ref 2

        // Free the canonical.
        mgr.notify_free(100);

        // First reference (200) should be promoted to canonical.
        assert!(!mgr.is_canonical(100), "old canonical should be gone");
        assert!(mgr.is_canonical(200), "200 should be promoted to canonical");
        assert!(mgr.is_reference(300), "300 still a reference");
        assert_eq!(mgr.table_size(), 1, "entry still exists");
    }

    #[test]
    fn dedup_notify_free_last_page() {
        let mgr = DedupManager::with_defaults();
        let page = make_page(0xFF, 4096);

        mgr.check_dedup(100, &page);

        // Free the only page.
        mgr.notify_free(100);

        assert!(!mgr.is_canonical(100));
        assert_eq!(mgr.table_size(), 0, "entry should be removed entirely");
    }

    #[test]
    fn dedup_break_dedup() {
        let mgr = DedupManager::with_defaults();
        let page = make_page(0x11, 4096);

        mgr.check_dedup(100, &page); // canonical
        mgr.check_dedup(200, &page); // reference

        // Break CoW on the reference.
        let hash = mgr.break_dedup(200);
        assert!(hash.is_some(), "should return the hash");

        assert!(!mgr.is_reference(200), "200 no longer a reference");
        assert!(mgr.is_canonical(100), "canonical untouched");
        assert_eq!(mgr.get_canonical_for(200), None, "200 has no canonical mapping");
    }

    #[test]
    fn dedup_is_canonical_is_reference() {
        let mgr = DedupManager::with_defaults();
        let page = make_page(0x22, 4096);

        // Before any insertion.
        assert!(!mgr.is_canonical(999));
        assert!(!mgr.is_reference(999));

        mgr.check_dedup(100, &page);
        assert!(mgr.is_canonical(100));
        assert!(!mgr.is_reference(100));

        mgr.check_dedup(200, &page);
        assert!(!mgr.is_canonical(200));
        assert!(mgr.is_reference(200));
    }

    #[test]
    fn dedup_stats_tracking() {
        let mgr = DedupManager::with_defaults();
        let page_a = make_page(0x33, 4096);
        let page_b = make_page(0x44, 4096);

        mgr.check_dedup(100, &page_a); // unique
        mgr.check_dedup(200, &page_a); // match
        mgr.check_dedup(300, &page_b); // unique
        mgr.check_dedup(400, &page_a); // match

        let s = mgr.stats();
        assert_eq!(s.total_checks, 4);
        assert_eq!(s.total_unique, 2);
        assert_eq!(s.total_matches, 2);
        assert_eq!(s.pages_saved, 2);
        assert_eq!(s.bytes_saved, 2 * 4096);
    }

    #[test]
    fn dedup_hook_trait() {
        // Verify DedupManager can be used as Box<dyn DedupHook>.
        let mgr = DedupManager::with_defaults();
        let hook: Box<dyn DedupHook> = Box::new(mgr);

        let page = make_page(0x55, 4096);
        assert_eq!(hook.check_dedup(100, &page), None);
        assert_eq!(hook.check_dedup(200, &page), Some(100));

        hook.notify_free(200);
        // After freeing the reference, a new check should still find canonical.
        assert_eq!(hook.check_dedup(300, &page), Some(100));
    }

    #[test]
    fn dedup_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let mgr = Arc::new(DedupManager::with_defaults());
        let page = make_page(0x66, 4096);

        // Spawn multiple threads that all check the same page content.
        let handles: Vec<_> = (0..8)
            .map(|i| {
                let mgr = Arc::clone(&mgr);
                let page = page.clone();
                thread::spawn(move || {
                    let vpn = 1000 + i as u64;
                    mgr.check_dedup(vpn, &page);
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        // One should be canonical, the rest references.
        let mut canonical_count = 0;
        let mut reference_count = 0;
        for i in 0..8u64 {
            let vpn = 1000 + i;
            if mgr.is_canonical(vpn) {
                canonical_count += 1;
            } else if mgr.is_reference(vpn) {
                reference_count += 1;
            }
        }
        assert_eq!(canonical_count, 1, "exactly one canonical");
        assert_eq!(reference_count, 7, "seven references");
        assert_eq!(mgr.table_size(), 1);
    }
}
