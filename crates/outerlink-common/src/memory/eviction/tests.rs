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

    // select_victim should return a page.
    // First let's re-fill to capacity and then check.
    let mut arc2 = ArcPolicy::new(3, 0);
    arc2.record_insert(10);
    arc2.record_insert(11);
    arc2.record_insert(12);
    let victim = arc2.select_victim();
    assert!(victim.is_some());
    let v = victim.unwrap();
    assert_eq!(v.source_tier, 0);
    // LRU of T1 should be 10 (first inserted, never re-accessed).
    assert_eq!(v.vpn, 10);
    assert_eq!(arc2.tracked_count(), 2);
}

#[test]
fn arc_promotes_to_t2_on_reaccess() {
    let mut arc = ArcPolicy::new(4, 0);

    arc.record_insert(1);
    arc.record_insert(2);
    arc.record_insert(3);

    // Access page 1 again -- should promote from T1 to T2.
    arc.record_access(1);

    // Now evict. Pages 2 and 3 are in T1, page 1 is in T2.
    // With p starting at 0, T1 should be preferred for eviction.
    // |T1| = 2 > p = 0, so evict from T1. LRU of T1 is 2.
    let victim = arc.select_victim().unwrap();
    assert_eq!(victim.vpn, 2);

    let victim2 = arc.select_victim().unwrap();
    assert_eq!(victim2.vpn, 3);

    // Only page 1 remains (in T2).
    let victim3 = arc.select_victim().unwrap();
    assert_eq!(victim3.vpn, 1);
}

#[test]
fn arc_ghost_hit_adapts_p() {
    let mut arc = ArcPolicy::new(2, 0);

    // Fill and evict to create ghost entries.
    arc.record_insert(1);
    arc.record_insert(2);
    // Evict page 1 (LRU of T1) -- goes to B1 ghost list.
    let v = arc.select_victim().unwrap();
    assert_eq!(v.vpn, 1);

    // Page 1 is now in B1. Check ghost hit.
    assert!(arc.is_ghost_hit(1));

    // Insert page 1 again -- this is a B1 ghost hit, should increase p.
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
    let victim = arc.select_victim().unwrap();
    assert_eq!(victim.vpn, 2);

    // Pin remaining unpinned pages.
    arc.pin(3);

    // Only page 1 is left, but it's pinned. No victim available.
    let victim = arc.select_victim();
    assert!(victim.is_none());
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
    assert!(arc.select_victim().is_none());
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

    let victim = car.select_victim();
    assert!(victim.is_some());
    let v = victim.unwrap();
    assert_eq!(v.source_tier, 2);
    assert_eq!(car.tracked_count(), 2);
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
    let victim = car.select_victim().unwrap();
    // Page 10 had ref bit set, so it should be skipped. 11 should be evicted.
    assert_eq!(victim.vpn, 11);
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

    // Now evict: page 2 is in T1 (no ref bit), page 1 is in T2.
    // With p ~ 0, T1 eviction preferred. T1 has only page 2.
    let victim = car.select_victim().unwrap();
    assert_eq!(victim.vpn, 2);

    // Page 1 is still tracked (in T2).
    assert_eq!(car.tracked_count(), 1);
    let victim2 = car.select_victim().unwrap();
    assert_eq!(victim2.vpn, 1);
}

#[test]
fn car_ghost_hit_adapts_p() {
    let mut car = CarPolicy::new(2, 2);

    car.record_insert(1);
    car.record_insert(2);

    // Evict page 1 -> goes to B1 ghost.
    let v = car.select_victim().unwrap();
    assert_eq!(v.vpn, 1);
    assert!(car.is_ghost_hit(1));

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

    let victim = clock.select_victim();
    assert!(victim.is_some());
    let v = victim.unwrap();
    assert_eq!(v.source_tier, 4);
    assert_eq!(clock.tracked_count(), 2);
}

#[test]
fn clock_second_chance() {
    let mut clock = ClockPolicy::new(3, 4);

    clock.record_insert(100);
    clock.record_insert(101);
    clock.record_insert(102);

    // Access page 100 to give it a second chance.
    clock.record_access(100);

    // Victim selection should skip 100, evict 101.
    let victim = clock.select_victim().unwrap();
    assert_eq!(victim.vpn, 101);
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
    let victim = clock.select_victim().unwrap();
    assert_eq!(victim.vpn, 102);

    // No more evictable pages.
    assert!(clock.select_victim().is_none());
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

    // Evicting should not return removed page 2.
    let v1 = clock.select_victim().unwrap();
    let v2 = clock.select_victim().unwrap();
    assert_ne!(v1.vpn, 2);
    assert_ne!(v2.vpn, 2);
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

        let victim = policy.select_victim();
        assert!(victim.is_some());

        policy.reset();
        assert_eq!(policy.tracked_count(), 0);
    }
}
