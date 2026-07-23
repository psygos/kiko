#!/usr/bin/env bash
#
# Offline Kiko Nano cold-boot/fault component acceptance.
#
# This runner opens no robot device and starts no systemd unit. It exercises
# the exact production parsers and simulated owner/fault paths. A passing run
# is software evidence only; see the final claims_not_established line.

set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly script_dir
repo_root="$(cd -- "${script_dir}/.." && pwd -P)"
readonly repo_root
readonly cargo_bin="${CARGO:-cargo}"

cd -- "${repo_root}"
export OAK_SYS_CHECK_ONLY="${OAK_SYS_CHECK_ONLY:-1}"

if ! command -v "${cargo_bin}" >/dev/null 2>&1; then
    printf 'acceptance error: cargo executable not found: %s\n' "${cargo_bin}" >&2
    exit 2
fi

run_exact() {
    local case_id="$1"
    local package="$2"
    local features="$3"
    local test_name="$4"
    local target="${5:-lib}"
    local -a command=(
        "${cargo_bin}"
        test
        --locked
        -p
        "${package}"
    )
    if [[ "${target}" == 'lib' ]]; then
        command+=(--lib)
    else
        command+=(--test "${target}")
    fi
    if [[ -n "${features}" ]]; then
        command+=(--features "${features}")
    fi
    command+=("${test_name}" -- --exact)

    printf '\n[%s] %s\n' "${case_id}" "${test_name}"
    local output
    if ! output="$("${command[@]}" 2>&1)"; then
        printf '%s\n' "${output}" >&2
        printf 'acceptance failed: %s\n' "${case_id}" >&2
        exit 1
    fi
    printf '%s\n' "${output}"
    if [[ "${output}" != *"test ${test_name} ... ok"* ]]; then
        printf \
            'acceptance failed: %s did not execute exactly the named test\n' \
            "${case_id}" >&2
        exit 1
    fi
}

printf '%s\n' \
    'Kiko Nano cold-boot/fault acceptance v1' \
    'scope=offline_component_simulation_only'

run_exact \
    'bundle.launch_last_and_content_addressed' \
    'kiko-nano-bundle-renderer' \
    '' \
    'launch_is_written_last_and_every_file_matches_plan_digest' \
    'renderer'
run_exact \
    'boot.qualified_enablement_retains_gate' \
    'kiko-nano-deployment-gate' \
    '' \
    'tests::qualified_enablement_cannot_bypass_exact_prestart_admission'
run_exact \
    'boot.marker_and_bound_byte_drift_fail_closed' \
    'kiko-nano-deployment-gate' \
    '' \
    'tests::marker_round_trip_is_strict_and_file_drift_fails_closed'
run_exact \
    'launch.exact_asset_digest_admission' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::nano_agent_launch::tests::exact_asset_loader_rejects_content_mismatch'
run_exact \
    'startup.single_controller_owner_and_zero_before_promotion' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::nano_production_admission::tests::exact_evidence_orders_controller_check_before_promotion_and_returns_disarmed_runtime'
run_exact \
    'startup.inventory_identity_mismatch_stops_without_promotion' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::nano_production_admission::tests::exact_inventory_identity_mismatch_stops_without_promotion'
run_exact \
    'startup.session_correlated_exact_zero' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::nano_production_admission::tests::initial_zero_must_name_the_pending_session_and_be_exact_zero'
run_exact \
    'startup.readiness_enters_disarmed_only' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::nano_startup::tests::exact_startup_evidence_enters_only_disarmed'
run_exact \
    'startup.control_socket_owner_ready_barrier' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::control_socket::tests::task_start_waits_until_the_child_has_taken_ownership_and_reported_ready'
run_exact \
    'startup.acquired_owner_cleanup_stops_then_closes' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::nano_bootstrap::tests::post_acquisition_cleanup_always_attempts_stop_then_close_and_retains_both'
run_exact \
    'startup.accessory_readiness_and_ordered_shutdown' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::nano_accessory_worker::tests::readiness_follows_both_startup_receipts_and_shutdown_is_ordered'
run_exact \
    'supervisor.boot_inventory_arm_zero_gates' \
    'kiko-supervisor-core' \
    '' \
    'state::tests::boot_inventory_arm_and_zero_are_distinct_gates'
run_exact \
    'supervisor.authority_lease_expiry_requires_zero' \
    'kiko-supervisor-core' \
    '' \
    'state::tests::lease_deadline_is_exclusive_and_requires_confirmed_stop'
run_exact \
    'supervisor.clock_fault_requires_reinventory' \
    'kiko-supervisor-core' \
    '' \
    'state::tests::clock_regression_and_fault_clear_cannot_resume_motion'
run_exact \
    'firmware.watchdog_feed_requires_complete_loop' \
    'embedded' \
    '' \
    'watchdog_gate::tests::feed_requires_every_piece_of_safe_completed_loop_evidence'
run_exact \
    'firmware.motion_lease_and_clock_gap_stop' \
    'embedded' \
    '' \
    'controller::tests::lease_expiry_and_clock_gap_are_fail_closed_across_counter_wrap'
run_exact \
    'firmware.priority_tx_reserves_stop_receipt_capacity' \
    'embedded' \
    '' \
    'transport_scheduler::tests::best_effort_saturation_cannot_consume_stop_or_applied_capacity'
run_exact \
    'controller.candidate_50_hz_baseline_retains_margin_under_100_hz_ceiling' \
    'robot-server' \
    '' \
    'config::tests::exact_115200_8n1_budget_admits_candidate_twenty_ms_fifty_hertz_baseline'
run_exact \
    'controller.continuous_uart_input_cannot_starve_host_or_timer' \
    'robot-server' \
    '' \
    'actuation_v2::tests::continuously_readable_observational_uart_cannot_starve_host_status_or_ack_timer'
run_exact \
    'controller.command_rate_does_not_delay_stop_or_consume_sequence' \
    'robot-server' \
    '' \
    'actuation_v2::tests::declared_command_rate_is_enforced_without_delaying_stop_or_consuming_sequence'
run_exact \
    'controller.partial_uart_shutdown_resynchronizes_and_confirms_stop' \
    'robot-server' \
    '' \
    'actuation_v2::tests::shutdown_during_command_write_resynchronizes_then_reports_exact_stop'
run_exact \
    'console.browser_deadman_applies_exact_zero' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::operator_console_runtime::tests::browser_deadman_truthfully_uses_conservative_global_stop_contract'
run_exact \
    'console.connection_loss_inhibits_and_attempts_release' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::operator_console_http::tests::embedded_console_bounds_requests_and_inhibits_stale_snapshots'
run_exact \
    'console.unified_map_mpc_controller_status_surface' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::operator_console_http::tests::embedded_console_exposes_one_typed_unified_status_surface'
run_exact \
    'console.stale_state_blocks_persistence_not_stop' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::operator_console_http::tests::embedded_console_inhibits_stale_persistence_but_keeps_stop_actions_available'
run_exact \
    'console.stop_receipt_identity_is_session_scoped' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::operator_console_http::tests::foreign_session_observes_latched_stop_without_receiving_its_response_id'
run_exact \
    'motion.transport_fault_latches_stop_uncertainty' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::live_motion_owner::tests::physical_tick_failure_latches_exact_fault_and_retains_stop_uncertainty'
run_exact \
    'map.atomic_save_and_exact_reload' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::nano_map_persistence::tests::save_has_explicit_empty_and_stale_errors_then_round_trips_exact_latest'
run_exact \
    'map.atomic_replacement_leaves_no_temporary_residue' \
    'kiko-slam' \
    'nano-agent' \
    'dense::occupancy_persistence::tests::atomic_save_replaces_and_loads_without_temporary_residue'
run_exact \
    'map.warm_start_requires_exact_replay' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::nano_map_persistence::tests::production_selection_loads_only_the_exact_session_and_never_claims_localization'
run_exact \
    'map.finalized_journal_identity_matches_retained_occupancy' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::nano_map_persistence::tests::finalized_journal_epoch_and_revision_must_exactly_match_retained_occupancy'
run_exact \
    'map.selection_publish_rejects_replaced_root_path' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::nano_map_persistence::tests::descriptor_relative_selection_publish_cannot_follow_a_replaced_root_path'
run_exact \
    'storage.dataset_limits_and_terminal_reserve_parse_once' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::nano_agent_launch::tests::navigation_dataset_limits_are_bounded_and_terminal_reserve_is_checked'
run_exact \
    'storage.ingress_capacity_cannot_exceed_record_limit' \
    'kiko-slam' \
    'nano-agent' \
    'live_runtime::tests::storage_record_limit_is_an_upper_bound_on_the_parsed_ingress_capacity'
run_exact \
    'storage.root_replacement_and_file_boundary_fail_closed' \
    'kiko-slam' \
    'nano-agent' \
    'dataset::storage_quota::tests::file_boundary_and_root_replacement_fail_closed'
run_exact \
    'storage.exact_dataset_boundary_publishes_only_after_journal' \
    'kiko-slam' \
    'nano-agent' \
    'dataset::tests::quota_exact_file_boundary_publishes_manifest_after_journal_finalization'
run_exact \
    'storage.quota_exhaustion_aborts_without_manifest' \
    'kiko-slam' \
    'nano-agent' \
    'dataset::tests::quota_file_exhaustion_poison_aborts_without_manifest'
run_exact \
    'shutdown.lifecycle_zero_precedes_controller_disarm' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::live_motion_owner::tests::terminal_shutdown_orders_lifecycle_zeros_before_confirmed_controller_disarm'
run_exact \
    'shutdown.uncertain_stop_is_fault_not_false_cancel' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::operator_console_runtime::tests::shutdown_with_uncertain_controller_stop_rejects_authority_and_latches_fault'
run_exact \
    'restart.console_capability_is_private_and_fresh' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::operator_console_http::tests::capability_persistence_is_private_atomic_and_never_reuses_old_value'
run_exact \
    'restart.terminal_http_completion_observed_by_owner' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::operator_console_http::tests::save_map_completion_is_http_observed_only_by_its_owning_session'
run_exact \
    'restart.control_socket_shutdown_joins_owner' \
    'kiko-slam' \
    'nano-agent' \
    'navigation::control_socket::tests::explicit_task_shutdown_joins_and_returns_cleanup_evidence'

printf '\n%s\n' \
    'acceptance_result=pass' \
    'claims_not_established=installation,systemd_pid1_execution,cold_power_boot,device_presence,usb_exclusivity,physical_watchdog,physical_emergency_stop,motor_stop_distance,head_torque,camera_stream,slam_accuracy,mpc_tracking,performance'
