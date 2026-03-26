//! Tests for eviction policies (ARC, CAR, CLOCK).

use super::*;
use crate::memory::traits::EvictionPolicy;

// ---------------------------------------------------------------------------
// ARC tests
// ---------------------------------------------------------------------------

#[test]
fn arc_basic_insert_evict() {
    let mut arc = ArcPolicy::new(4, 0);

    // Insert 4 pages -- fills capacity.
    for vpn in 0..4 {
        arc.record_insert(vpn);
    }
    assert_eq!(arc.tracked_count(), 4);

    // Insert a 5th page -- should trigger internal replacement.
    arc.record_insert(4);
    // After replacement, we should be back at max_size.
    assert_eq!(arc.tracked_count(), 4);

    // select_victim should return candidates.
    let mut arc2 = ArcPolicy::new(3, 0);
    arc2.record_insert(10);
    arc2.record_insert(11);
    arc2.record_insert(12);
    let victims = arc2.select_victim(1);
    assert_eq!(victims.len(), 1);
    let v = &victims[0];
    assert_eq!(v.tier_id, 0);
    // LRU of T1 should be 10 (first inserted, never re-accessed).
    assert_eq!(v.vpn, 10);
    // select_victim is non-destructive, tracked_count unchanged.
    assert_eq!(arc2.tracked_count(), 3);
}

#[test]
fn arc_promotes_to_t2_on_reaccess() {
    let mut arc = ArcPolicy::new(4, 0);

    arc.record_insert(1);
    arc.record_insert(2);
    arc.record_insert(3);

    // Access page 1 again -- should promote from T1 to T2.
    arc.record_access(1);

    // Select victims. Pages 2 and 3 are in T1, page 1 is in T2.
    // With p starting at 0, T1 should be preferred for eviction.
    // |T1| = 2 > p = 0, so evict from T1. LRU of T1 is 2.
    let victims = arc.select_victim(3);
    assert!(victims.len() >= 2);
    assert_eq!(victims[0].vpn, 2);
    assert_eq!(victims[1].vpn, 3);
    // Page 1 (in T2) should be last.
    if victims.len() >= 3 {
        assert_eq!(victims[2].vpn, 1);
    }
}

#[test]
fn arc_ghost_hit_adapts_p() {
    // Use capacity 4 so the directory cap (T1+B1 >= max_size) doesn't
    // immediately evict the ghost entry when we insert a replacement page.
    // With max_size=4: fill 4 pages, insert 5th triggers eviction of page 1
    // to B1. Then T1=[2,3,4,5], B1=[1], T1+B1=5 > 4 so B1 is trimmed.
    // Instead: fill 4, then use record_remove to free a slot before inserting,
    // so we don't trigger the directory cap.
    let mut arc = ArcPolicy::new(4, 0);

    // Fill to capacity.
    arc.record_insert(1);
    arc.record_insert(2);
    arc.record_insert(3);
    arc.record_insert(4);
    assert_eq!(arc.tracked_count(), 4);

    // Manually remove page 4 to make room, then insert page 5.
    // This way page 1 stays as LRU of T1 and T1+B1 stays under control.
    // Actually, let's use a different approach: insert 5 which triggers
    // replace() evicting page 1 to B1. Then T1=[2,3,4,5], B1=[1].
    // T1+B1 = 4+1 = 5 > 4, so the code pops B1 front (page 1).
    //
    // The real fix: remove a page first to keep T1 small.
    arc.record_remove(4); // T1=[1,2,3], count=3
    arc.record_remove(3); // T1=[1,2], count=2

    // Now insert 5 and 6 to fill back to capacity. replace() triggers
    // eviction of page 1 to B1 at capacity.
    arc.record_insert(5); // T1=[1,2,5], count=3
    arc.record_insert(6); // T1=[1,2,5,6], count=4
    assert_eq!(arc.tracked_count(), 4);

    // Insert page 7 triggers replace(). Page 1 (LRU of T1) -> B1.
    // After replace: T1=[2,5,6], B1=[1]. Then new-page path checks
    // T1+B1 = 3+1 = 4 >= 4, pops B1 front... page 1 again.
    //
    // This approach still has the directory cap issue.
    // The fundamental issue: ARC's directory cap means T1+B1 can't exceed
    // max_size. So after evicting to B1, the very next insert may trim B1.
    //
    // Solution: after the eviction, re-insert page 1 IMMEDIATELY before
    // inserting any other new page. This way we catch the ghost hit before
    // the directory trim happens.

    // Reset and try a clean approach.
    arc.reset();

    arc.record_insert(1);
    arc.record_insert(2);
    arc.record_insert(3);
    arc.record_insert(4);
    assert_eq!(arc.tracked_count(), 4);

    // Insert 5 -> triggers replace, evicts page 1 to B1, then checks
    // T1+B1 = 3+1 = 4 >= 4, trims B1. So page 1 is lost.
    // The ghost is only alive during the insert call.
    //
    // To test ghost hit adaptation, we need to re-insert the evicted page
    // as the NEXT insert (which triggers the ghost hit check before the
    // directory trim of a subsequent new page).
    //
    // When page 5 is inserted:
    //   1. replace() evicts page 1 from T1 to B1. T1=[2,3,4], B1=[1].
    //   2. Check T1+B1 = 3+1 = 4 >= 4. Pop B1 front (page 1). B1=[].
    //   3. Insert 5 into T1. T1=[2,3,4,5].
    //
    // So page 1 never survives in B1 at max_size=4 either. We need to
    // ensure T1 has fewer entries when we trigger the eviction.
    //
    // Approach: promote some pages to T2 first (via record_access),
    // reducing T1 length. Then after eviction, T1+B1 < max_size.

    arc.record_access(3); // Promote 3: T1->T2. T1=[1,2,4], T2=[3].
    arc.record_access(4); // Promote 4: T1->T2. T1=[1,2], T2=[3,4].

    // Now T1+T2 = 4, at capacity. Insert 5 -> replace().
    // p=0, t1_len=2, 2>0 -> evict from T1. Page 1 -> B1.
    // T1=[2], T2=[3,4], B1=[1]. T1+B1 = 1+1 = 2 < 4. No trim!
    // Then insert 5 into T1. T1=[2,5], T2=[3,4], B1=[1].
    arc.record_insert(5);
    assert_eq!(arc.tracked_count(), 4);

    // Page 1 should be in B1 ghost list.
    assert!(
        arc.is_ghost_hit(1),
        "page 1 should be in B1 ghost list after eviction"
    );

    // Re-insert page 1 -> B1 ghost hit -> increase p.
    let p_before = arc.adaptive_param();
    arc.record_insert(1);
    let p_after = arc.adaptive_param();
    assert!(
        p_after > p_before,
        "B1 ghost hit should increase p: before={}, after={}",
        p_before,
        p_after
    );
}

#[test]
fn arc_pinned_pages_skipped() {
    let mut arc = ArcPolicy::new(4, 0);

    arc.record_insert(1);
    arc.record_insert(2);
    arc.record_insert(3);

    // Pin page 1 (LRU of T1).
    arc.pin(1);

    // Victim should skip page 1 and return page 2.
    let victims = arc.select_victim(1);
    assert_eq!(victims.len(), 1);
    assert_eq!(victims[0].vpn, 2);

    // Pin remaining unpinned pages.
    arc.pin(2);
    arc.pin(3);

    // All pinned. No victim available.
    let victims = arc.select_victim(1);
    assert!(victims.is_empty());
}

#[test]
fn arc_reset_clears_all() {
    let mut arc = ArcPolicy::new(4, 0);

    arc.record_insert(1);
    arc.record_insert(2);
    arc.record_access(1);
    arc.pin(2);

    arc.reset();

    assert_eq!(arc.tracked_count(), 0);
    assert!(!arc.is_ghost_hit(1));
    assert!(arc.select_victim(1).is_empty());
}

// ---------------------------------------------------------------------------
// CAR tests
// ---------------------------------------------------------------------------

#[test]
fn car_basic_insert_evict() {
    let mut car = CarPolicy::new(3, 2);

    car.record_insert(10);
    car.record_insert(11);
    car.record_insert(12);
    assert_eq!(car.tracked_count(), 3);

    let victims = car.select_victim(1);
    assert!(!victims.is_empty());
    let v = &victims[0];
    assert_eq!(v.tier_id, 2);
    // select_victim is non-destructive.
    assert_eq!(car.tracked_count(), 3);
}

#[test]
fn car_reference_bit_second_chance() {
    let mut car = CarPolicy::new(3, 2);

    car.record_insert(10);
    car.record_insert(11);
    car.record_insert(12);

    // Access page 10 to set its reference bit.
    car.record_access(10);

    // When selecting a victim, page 10 should get a second chance
    // (ref bit cleared) and the clock should advance past it.
    let victims = car.select_victim(1);
    assert_eq!(victims.len(), 1);
    // Page 10 had ref bit set, so it should be skipped. 11 should be evicted.
    assert_eq!(victims[0].vpn, 11);
}

#[test]
fn car_promotes_to_t2() {
    let mut car = CarPolicy::new(4, 2);

    car.record_insert(1);
    car.record_insert(2);

    // Access page 1 to set ref bit.
    car.record_access(1);

    // Access again -- with ref bit already set, should promote to T2.
    car.record_access(1);

    // Now select victims: page 2 is in T1 (no ref bit), page 1 is in T2.
    // With p ~ 0, T1 eviction preferred. T1 has only page 2.
    let victims = car.select_victim(2);
    assert!(victims.len() >= 1);
    assert_eq!(victims[0].vpn, 2);

    // Page 1 should also appear as a candidate (from T2).
    if victims.len() >= 2 {
        assert_eq!(victims[1].vpn, 1);
    }
}

#[test]
fn car_ghost_hit_adapts_p() {
    // To observe a ghost hit, we need the evicted page to remain in B1
    // after the eviction. The directory cap trims B1 when T1+B1 >= max_size.
    // Strategy: promote some pages to T2 (via record_access) to reduce T1
    // size, so after eviction T1+B1 < max_size.
    let mut car = CarPolicy::new(4, 2);

    car.record_insert(1);
    car.record_insert(2);
    car.record_insert(3);
    car.record_insert(4);
    assert_eq!(car.tracked_count(), 4);

    // Promote pages 3 and 4 to T2 by accessing them (sets ref bit),
    // then accessing again (ref bit already set -> promote).
    car.record_access(3);
    car.record_access(3); // promotes 3 to T2
    car.record_access(4);
    car.record_access(4); // promotes 4 to T2
    // Now T1=[1,2], T2=[3,4]. T1+T2=4, at capacity.

    // Insert page 5 -> triggers replace(). With p=0, t1_len=2 >= p=0,
    // evict from T1 clock. Page 1 (at hand position) goes to B1.
    // T1=[2], T2=[3,4], B1=[1]. T1+B1 = 1+1 = 2 < 4. No trim!
    car.record_insert(5);
    assert_eq!(car.tracked_count(), 4);

    assert!(
        car.is_ghost_hit(1),
        "page 1 should be in B1 ghost list after eviction"
    );

    // Re-insert page 1 (B1 ghost hit) should increase p.
    let p_before = car.adaptive_param();
    car.record_insert(1);
    let p_after = car.adaptive_param();
    assert!(
        p_after > p_before,
        "B1 ghost hit should increase p: before={}, after={}",
        p_before,
        p_after
    );
}

// ---------------------------------------------------------------------------
// CLOCK tests
// ---------------------------------------------------------------------------

#[test]
fn clock_basic_insert_evict() {
    let mut clock = ClockPolicy::new(3, 4);

    clock.record_insert(100);
    clock.record_insert(101);
    clock.record_insert(102);
    assert_eq!(clock.tracked_count(), 3);

    let victims = clock.select_victim(1);
    assert!(!victims.is_empty());
    let v = &victims[0];
    assert_eq!(v.tier_id, 4);
    // select_victim is non-destructive.
    assert_eq!(clock.tracked_count(), 3);
}

#[test]
fn clock_second_chance() {
    let mut clock = ClockPolicy::new(3, 4);

    clock.record_insert(100);
    clock.record_insert(101);
    clock.record_insert(102);

    // Access page 100 to give it a second chance.
    clock.record_access(100);

    // Victim selection should skip 100, return 101.
    let victims = clock.select_victim(1);
    assert_eq!(victims.len(), 1);
    assert_eq!(victims[0].vpn, 101);
}

#[test]
fn clock_pinned_skipped() {
    let mut clock = ClockPolicy::new(3, 4);

    clock.record_insert(100);
    clock.record_insert(101);
    clock.record_insert(102);

    clock.pin(100);
    clock.pin(101);

    // Only page 102 is evictable.
    let victims = clock.select_victim(1);
    assert_eq!(victims.len(), 1);
    assert_eq!(victims[0].vpn, 102);

    // Pin 102 too -- no more evictable pages.
    clock.pin(102);
    assert!(clock.select_victim(1).is_empty());
}

#[test]
fn clock_remove() {
    let mut clock = ClockPolicy::new(4, 4);

    clock.record_insert(1);
    clock.record_insert(2);
    clock.record_insert(3);
    assert_eq!(clock.tracked_count(), 3);

    clock.record_remove(2);
    assert_eq!(clock.tracked_count(), 2);

    // Selecting victims should not return removed page 2.
    let victims = clock.select_victim(2);
    for v in &victims {
        assert_ne!(v.vpn, 2);
    }
}

// ---------------------------------------------------------------------------
// Cross-policy: trait object compatibility
// ---------------------------------------------------------------------------

#[test]
fn all_policies_implement_trait() {
    let policies: Vec<Box<dyn EvictionPolicy>> = vec![
        Box::new(ArcPolicy::new(8, 0)),
        Box::new(CarPolicy::new(8, 2)),
        Box::new(ClockPolicy::new(8, 4)),
    ];

    for mut policy in policies {
        policy.record_insert(1);
        policy.record_insert(2);
        policy.record_access(1);
        assert_eq!(policy.tracked_count(), 2);
        assert!(policy.memory_overhead() > 0);

        let victims = policy.select_victim(1);
        assert!(!victims.is_empty());

        policy.reset();
        assert_eq!(policy.tracked_count(), 0);
    }
}

// ---------------------------------------------------------------------------
// Multi-victim selection
// ---------------------------------------------------------------------------

#[test]
fn select_victim_returns_multiple() {
    let mut arc = ArcPolicy::new(8, 0);
    for vpn in 0..6 {
        arc.record_insert(vpn);
    }

    let victims = arc.select_victim(3);
    assert_eq!(victims.len(), 3);
    // All VPNs should be unique.
    let vpns: Vec<u64> = victims.iter().map(|v| v.vpn).collect();
    let mut unique = vpns.clone();
    unique.sort();
    unique.dedup();
    assert_eq!(unique.len(), 3);
}

#[test]
fn select_victim_count_exceeds_tracked() {
    let mut clock = ClockPolicy::new(8, 4);
    clock.record_insert(1);
    clock.record_insert(2);

    // Ask for 10, but only 2 tracked.
    let victims = clock.select_victim(10);
    assert_eq!(victims.len(), 2);
}
