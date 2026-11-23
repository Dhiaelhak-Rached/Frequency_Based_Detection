# PX4 SITL + Gazebo Acoustic Guidance Validation Report

> **Scope:** Software-in-the-loop simulation using **PX4 SITL v1.15.4 + Gazebo Garden (`gz_x500`) + MAVSDK-Python 3.17.4**, running the actual `guidance.py`, `px4_controller.py`, and UMA-16 front-only `FrequencyDetector` + `ArrayDOA` pipeline.
> Simulated results validate closed-loop control logic and PX4 Offboard behavior; they **do not** replace controlled physical hardware acoustic/propeller-noise flight tests.

**Overall Result:** 13/17 scenarios PASSED

## Summary Table

| Scenario | Status | Summary |
|---|---|---|
| `B1_full_state_sequence` | **FAIL** | Stages=['TARGET_LOST_HOLD', 'HOVER_SEARCH', 'ALIGN_YAW', 'ALIGN_VERTICAL']; yaw_rot_ok=True, climb_ok=True, realign_ok=False, hold_ok=True, horiz_progress=0.39m, shift_at=None, dropout_at=None |
| `B2_descend_sign_check` | **PASS** | Alt 7.85m -> 5.18m (target 5.5m), max_down_cmd=+0.3m/s, rem_diff=+0.32m |
| `E1_close_climb` | **PASS** | Alt 4.86->6.75m (tgt 7.0m), rem=+0.25m, el=3.5°, progress=2.21m |
| `E2_large_climb_travel_cap` | **FAIL** | Cmd climb stopped at +2.90m (limit=3.0m), end_alt=7.85m vs tgt=9.5m, rem_diff=+1.65m, end_cmd_down=-0.0652194834720756 |
| `E3_long_range_deadband` | **FAIL** | At first FOLLOW: rem_diff=Nonem; end_diff=+1.56m, re_climbed=False, progress=0.02m |
| `E4_moving_target_3d` | **PASS** | Tracked moving target: alt 4.86->6.35m (tgt 6.88m), yaw=32.6° |
| `E5_noisy_delayed_propeller` | **PASS** | Noisy+propeller(suppressed): states=['TARGET_LOST_HOLD', 'HOVER_SEARCH', 'ALIGN_YAW', 'ALIGN_VERTICAL', 'FOLLOW'], rem_diff=+0.49m, mean_az_err=0.29°, mean_el_err=1.0° |
| `S1_loss_during_climb` | **PASS** | entered_climb=True, held_zero_on_loss=True, states=['TARGET_LOST_HOLD', 'HOVER_SEARCH', 'ALIGN_YAW', 'ALIGN_VERTICAL', 'TARGET_LOST_HOLD', 'HOVER_SEARCH', 'ALIGN_YAW', 'ALIGN_VERTICAL', 'FOLLOW'] |
| `S2_loss_during_follow` | **FAIL** | states=['TARGET_LOST_HOLD', 'HOVER_SEARCH', 'ALIGN_YAW', 'ALIGN_VERTICAL', 'TARGET_LOST_HOLD'], final=TARGET_LOST_HOLD (acoustic_observation_invalid), cmd_fwd=0.0 |
| `S3_stale_and_repeated_obs` | **PASS** | states=['TARGET_LOST_HOLD', 'HOVER_SEARCH', 'ALIGN_YAW', 'ALIGN_VERTICAL', 'FOLLOW', 'TARGET_LOST_HOLD'], final=TARGET_LOST_HOLD (acoustic_observation_timeout) |
| `S4_unreliable_elevation` | **PASS** | states=['TARGET_LOST_HOLD', 'HOVER_SEARCH', 'ALIGN_YAW', 'ALIGN_VERTICAL', 'FOLLOW', 'TARGET_LOST_HOLD'], saw_invalid=True, final=TARGET_LOST_HOLD (acoustic_observation_invalid) |
| `S5_telemetry_stale_abort` | **PASS** | final=ABORTED (flight_telemetry_stale) |
| `S6_altitude_limits` | **PASS** | final=ABORTED (altitude_envelope_violation), end_alt=6.02m (limit=6.0m), cmd_down=0.0 |
| `S7_geofence_violation` | **PASS** | final=ABORTED (geofence_violation), progress=2.51m |
| `S8_manual_override` | **PASS** | final=ABORTED (offboard_lost_or_manual_override) |
| `S9_offboard_connection_loss` | **PASS** | final=ABORTED (offboard_lost_or_manual_override) |
| `T0_subprocess_px4_controller` | **PASS** | Real px4_controller.py subprocess exit=0; yaw -1.2°->29.2°, alt 5.15m->8.28m, dist 9.58m->5.00m, states_seen=True, offboard_released=True |

## Detailed Metrics & Observations

### `B1_full_state_sequence` — FAIL
- **Summary:** Stages=['TARGET_LOST_HOLD', 'HOVER_SEARCH', 'ALIGN_YAW', 'ALIGN_VERTICAL']; yaw_rot_ok=True, climb_ok=True, realign_ok=False, hold_ok=True, horiz_progress=0.39m, shift_at=None, dropout_at=None
```json
{
  "duration_s": 64.98,
  "samples": 1278,
  "state_sequence": [
    "TARGET_LOST_HOLD",
    "HOVER_SEARCH",
    "ALIGN_YAW",
    "ALIGN_VERTICAL"
  ],
  "final_state": "ALIGN_VERTICAL",
  "final_reason": "aligning_vertical",
  "start_drone_alt_m": 5.02,
  "end_drone_alt_m": 5.84,
  "end_target_alt_m": 6.5,
  "remaining_alt_diff_m": 0.66,
  "end_yaw_deg": 35.0,
  "end_meas_az_deg": 1.5,
  "end_meas_el_deg": 4.0,
  "start_horiz_dist_m": 9.68,
  "end_horiz_dist_m": 9.29,
  "horiz_progress_m": 0.39,
  "max_forward_cmd_m_s": 0.0,
  "min_down_cmd_m_s": -0.18,
  "max_down_cmd_m_s": 0.0,
  "mean_az_err_deg": 0.27,
  "mean_el_err_deg": 0.07,
  "shift_at_s": null,
  "dropout_at_s": null
}
```
- **Plot:** [B1_full_state_sequence.png](file:///home/dhia/Desktop/projects/freq_project/frequency_id/sim/reports/plots/B1_full_state_sequence.png)

### `B2_descend_sign_check` — PASS
- **Summary:** Alt 7.85m -> 5.18m (target 5.5m), max_down_cmd=+0.3m/s, rem_diff=+0.32m
```json
{
  "duration_s": 29.98,
  "samples": 593,
  "state_sequence": [
    "TARGET_LOST_HOLD",
    "HOVER_SEARCH",
    "ALIGN_YAW",
    "ALIGN_VERTICAL",
    "FOLLOW",
    "ALIGN_VERTICAL"
  ],
  "final_state": "ALIGN_VERTICAL",
  "final_reason": "aligning_vertical",
  "start_drone_alt_m": 7.85,
  "end_drone_alt_m": 5.18,
  "end_target_alt_m": 5.5,
  "remaining_alt_diff_m": 0.32,
  "end_yaw_deg": -0.6,
  "end_meas_az_deg": -3.2,
  "end_meas_el_deg": 6.1,
  "start_horiz_dist_m": 7.0,
  "end_horiz_dist_m": 3.24,
  "horiz_progress_m": 3.77,
  "max_forward_cmd_m_s": 0.5,
  "min_down_cmd_m_s": -0.18,
  "max_down_cmd_m_s": 0.3,
  "mean_az_err_deg": 0.13,
  "mean_el_err_deg": 0.37
}
```
- **Plot:** [B2_descend_sign_check.png](file:///home/dhia/Desktop/projects/freq_project/frequency_id/sim/reports/plots/B2_descend_sign_check.png)

### `E1_close_climb` — PASS
- **Summary:** Alt 4.86->6.75m (tgt 7.0m), rem=+0.25m, el=3.5°, progress=2.21m
```json
{
  "duration_s": 27.99,
  "samples": 554,
  "state_sequence": [
    "TARGET_LOST_HOLD",
    "HOVER_SEARCH",
    "ALIGN_YAW",
    "ALIGN_VERTICAL",
    "FOLLOW",
    "ALIGN_VERTICAL"
  ],
  "final_state": "ALIGN_VERTICAL",
  "final_reason": "aligning_vertical",
  "start_drone_alt_m": 4.86,
  "end_drone_alt_m": 6.75,
  "end_target_alt_m": 7.0,
  "remaining_alt_diff_m": 0.25,
  "end_yaw_deg": 18.3,
  "end_meas_az_deg": -4.7,
  "end_meas_el_deg": 3.5,
  "start_horiz_dist_m": 6.49,
  "end_horiz_dist_m": 4.28,
  "horiz_progress_m": 2.21,
  "max_forward_cmd_m_s": 0.5,
  "min_down_cmd_m_s": -0.3,
  "max_down_cmd_m_s": 0.0,
  "mean_az_err_deg": 0.29,
  "mean_el_err_deg": 0.33
}
```
- **Plot:** [E1_close_climb.png](file:///home/dhia/Desktop/projects/freq_project/frequency_id/sim/reports/plots/E1_close_climb.png)

### `E2_large_climb_travel_cap` — FAIL
- **Summary:** Cmd climb stopped at +2.90m (limit=3.0m), end_alt=7.85m vs tgt=9.5m, rem_diff=+1.65m, end_cmd_down=-0.0652194834720756
```json
{
  "duration_s": 29.97,
  "samples": 593,
  "state_sequence": [
    "TARGET_LOST_HOLD",
    "HOVER_SEARCH",
    "ALIGN_YAW",
    "ALIGN_VERTICAL"
  ],
  "final_state": "ALIGN_VERTICAL",
  "final_reason": "aligning_vertical",
  "start_drone_alt_m": 4.95,
  "end_drone_alt_m": 7.85,
  "end_target_alt_m": 9.5,
  "remaining_alt_diff_m": 1.65,
  "end_yaw_deg": -0.0,
  "end_meas_az_deg": 1.2,
  "end_meas_el_deg": 11.9,
  "start_horiz_dist_m": 8.01,
  "end_horiz_dist_m": 7.76,
  "horiz_progress_m": 0.25,
  "max_forward_cmd_m_s": 0.0,
  "min_down_cmd_m_s": -0.3,
  "max_down_cmd_m_s": 0.0,
  "mean_az_err_deg": 0.09,
  "mean_el_err_deg": 0.21,
  "commanded_climb_before_cap_m": 2.9
}
```
- **Plot:** [E2_large_climb_travel_cap.png](file:///home/dhia/Desktop/projects/freq_project/frequency_id/sim/reports/plots/E2_large_climb_travel_cap.png)

### `E3_long_range_deadband` — FAIL
- **Summary:** At first FOLLOW: rem_diff=Nonem; end_diff=+1.56m, re_climbed=False, progress=0.02m
```json
{
  "duration_s": 29.99,
  "samples": 593,
  "state_sequence": [
    "TARGET_LOST_HOLD",
    "HOVER_SEARCH",
    "ALIGN_YAW",
    "ALIGN_VERTICAL"
  ],
  "final_state": "ALIGN_VERTICAL",
  "final_reason": "aligning_vertical",
  "start_drone_alt_m": 4.91,
  "end_drone_alt_m": 4.39,
  "end_target_alt_m": 5.95,
  "remaining_alt_diff_m": 1.56,
  "end_yaw_deg": -2.2,
  "end_meas_az_deg": -3.0,
  "end_meas_el_deg": 7.7,
  "start_horiz_dist_m": 11.97,
  "end_horiz_dist_m": 11.95,
  "horiz_progress_m": 0.02,
  "max_forward_cmd_m_s": 0.0,
  "min_down_cmd_m_s": -0.16,
  "max_down_cmd_m_s": 0.0,
  "mean_az_err_deg": 0.09,
  "mean_el_err_deg": 0.05,
  "first_follow_alt_diff_m": null,
  "re_climbed_as_range_closed": false
}
```
- **Plot:** [E3_long_range_deadband.png](file:///home/dhia/Desktop/projects/freq_project/frequency_id/sim/reports/plots/E3_long_range_deadband.png)

### `E4_moving_target_3d` — PASS
- **Summary:** Tracked moving target: alt 4.86->6.35m (tgt 6.88m), yaw=32.6°
```json
{
  "duration_s": 32.0,
  "samples": 633,
  "state_sequence": [
    "TARGET_LOST_HOLD",
    "HOVER_SEARCH",
    "ALIGN_YAW",
    "ALIGN_VERTICAL",
    "FOLLOW"
  ],
  "final_state": "FOLLOW",
  "final_reason": "following_direction_without_range",
  "start_drone_alt_m": 4.86,
  "end_drone_alt_m": 6.35,
  "end_target_alt_m": 6.88,
  "remaining_alt_diff_m": 0.53,
  "end_yaw_deg": 32.6,
  "end_meas_az_deg": 5.6,
  "end_meas_el_deg": 4.4,
  "start_horiz_dist_m": 7.77,
  "end_horiz_dist_m": 7.48,
  "horiz_progress_m": 0.29,
  "max_forward_cmd_m_s": 0.5,
  "min_down_cmd_m_s": -0.17,
  "max_down_cmd_m_s": 0.0,
  "mean_az_err_deg": 0.4,
  "mean_el_err_deg": 0.12
}
```
- **Plot:** [E4_moving_target_3d.png](file:///home/dhia/Desktop/projects/freq_project/frequency_id/sim/reports/plots/E4_moving_target_3d.png)

### `E5_noisy_delayed_propeller` — PASS
- **Summary:** Noisy+propeller(suppressed): states=['TARGET_LOST_HOLD', 'HOVER_SEARCH', 'ALIGN_YAW', 'ALIGN_VERTICAL', 'FOLLOW'], rem_diff=+0.49m, mean_az_err=0.29°, mean_el_err=1.0°
```json
{
  "duration_s": 33.98,
  "samples": 672,
  "state_sequence": [
    "TARGET_LOST_HOLD",
    "HOVER_SEARCH",
    "ALIGN_YAW",
    "ALIGN_VERTICAL",
    "FOLLOW"
  ],
  "final_state": "FOLLOW",
  "final_reason": "following_direction_without_range",
  "start_drone_alt_m": 5.11,
  "end_drone_alt_m": 6.31,
  "end_target_alt_m": 6.8,
  "remaining_alt_diff_m": 0.49,
  "end_yaw_deg": 22.1,
  "end_meas_az_deg": -4.6,
  "end_meas_el_deg": 3.9,
  "start_horiz_dist_m": 7.91,
  "end_horiz_dist_m": 5.38,
  "horiz_progress_m": 2.53,
  "max_forward_cmd_m_s": 0.5,
  "min_down_cmd_m_s": -0.19,
  "max_down_cmd_m_s": 0.0,
  "mean_az_err_deg": 0.29,
  "mean_el_err_deg": 1.0
}
```
- **Plot:** [E5_noisy_delayed_propeller.png](file:///home/dhia/Desktop/projects/freq_project/frequency_id/sim/reports/plots/E5_noisy_delayed_propeller.png)

### `S1_loss_during_climb` — PASS
- **Summary:** entered_climb=True, held_zero_on_loss=True, states=['TARGET_LOST_HOLD', 'HOVER_SEARCH', 'ALIGN_YAW', 'ALIGN_VERTICAL', 'TARGET_LOST_HOLD', 'HOVER_SEARCH', 'ALIGN_YAW', 'ALIGN_VERTICAL', 'FOLLOW']
```json
{
  "duration_s": 12.96,
  "samples": 257,
  "state_sequence": [
    "TARGET_LOST_HOLD",
    "HOVER_SEARCH",
    "ALIGN_YAW",
    "ALIGN_VERTICAL",
    "TARGET_LOST_HOLD",
    "HOVER_SEARCH",
    "ALIGN_YAW",
    "ALIGN_VERTICAL",
    "FOLLOW"
  ],
  "final_state": "FOLLOW",
  "final_reason": "alignment_stable",
  "start_drone_alt_m": 5.11,
  "end_drone_alt_m": 7.27,
  "end_target_alt_m": 7.5,
  "remaining_alt_diff_m": 0.23,
  "end_yaw_deg": -0.2,
  "end_meas_az_deg": -0.2,
  "end_meas_el_deg": 2.1,
  "start_horiz_dist_m": 7.95,
  "end_horiz_dist_m": 7.78,
  "horiz_progress_m": 0.17,
  "max_forward_cmd_m_s": 0.0,
  "min_down_cmd_m_s": -0.3,
  "max_down_cmd_m_s": 0.0,
  "mean_az_err_deg": 0.04,
  "mean_el_err_deg": 0.55
}
```
- **Plot:** [S1_loss_during_climb.png](file:///home/dhia/Desktop/projects/freq_project/frequency_id/sim/reports/plots/S1_loss_during_climb.png)

### `S2_loss_during_follow` — FAIL
- **Summary:** states=['TARGET_LOST_HOLD', 'HOVER_SEARCH', 'ALIGN_YAW', 'ALIGN_VERTICAL', 'TARGET_LOST_HOLD'], final=TARGET_LOST_HOLD (acoustic_observation_invalid), cmd_fwd=0.0
```json
{
  "duration_s": 13.98,
  "samples": 277,
  "state_sequence": [
    "TARGET_LOST_HOLD",
    "HOVER_SEARCH",
    "ALIGN_YAW",
    "ALIGN_VERTICAL",
    "TARGET_LOST_HOLD"
  ],
  "final_state": "TARGET_LOST_HOLD",
  "final_reason": "acoustic_observation_invalid",
  "start_drone_alt_m": 5.31,
  "end_drone_alt_m": 7.21,
  "end_target_alt_m": 5.0,
  "remaining_alt_diff_m": -2.21,
  "end_yaw_deg": 0.4,
  "end_meas_az_deg": null,
  "end_meas_el_deg": null,
  "start_horiz_dist_m": 8.99,
  "end_horiz_dist_m": 9.07,
  "horiz_progress_m": -0.08,
  "max_forward_cmd_m_s": 0.0,
  "min_down_cmd_m_s": 0.0,
  "max_down_cmd_m_s": 0.15,
  "mean_az_err_deg": 0.09,
  "mean_el_err_deg": 0.18
}
```
- **Plot:** [S2_loss_during_follow.png](file:///home/dhia/Desktop/projects/freq_project/frequency_id/sim/reports/plots/S2_loss_during_follow.png)

### `S3_stale_and_repeated_obs` — PASS
- **Summary:** states=['TARGET_LOST_HOLD', 'HOVER_SEARCH', 'ALIGN_YAW', 'ALIGN_VERTICAL', 'FOLLOW', 'TARGET_LOST_HOLD'], final=TARGET_LOST_HOLD (acoustic_observation_timeout)
```json
{
  "duration_s": 11.96,
  "samples": 237,
  "state_sequence": [
    "TARGET_LOST_HOLD",
    "HOVER_SEARCH",
    "ALIGN_YAW",
    "ALIGN_VERTICAL",
    "FOLLOW",
    "TARGET_LOST_HOLD"
  ],
  "final_state": "TARGET_LOST_HOLD",
  "final_reason": "acoustic_observation_timeout",
  "start_drone_alt_m": 5.17,
  "end_drone_alt_m": 6.25,
  "end_target_alt_m": 5.0,
  "remaining_alt_diff_m": -1.25,
  "end_yaw_deg": 0.3,
  "end_meas_az_deg": -0.1,
  "end_meas_el_deg": -6.8,
  "start_horiz_dist_m": 9.02,
  "end_horiz_dist_m": 6.86,
  "horiz_progress_m": 2.15,
  "max_forward_cmd_m_s": 0.5,
  "min_down_cmd_m_s": 0.0,
  "max_down_cmd_m_s": 0.0,
  "mean_az_err_deg": 0.24,
  "mean_el_err_deg": 1.51
}
```
- **Plot:** [S3_stale_and_repeated_obs.png](file:///home/dhia/Desktop/projects/freq_project/frequency_id/sim/reports/plots/S3_stale_and_repeated_obs.png)

### `S4_unreliable_elevation` — PASS
- **Summary:** states=['TARGET_LOST_HOLD', 'HOVER_SEARCH', 'ALIGN_YAW', 'ALIGN_VERTICAL', 'FOLLOW', 'TARGET_LOST_HOLD'], saw_invalid=True, final=TARGET_LOST_HOLD (acoustic_observation_invalid)
```json
{
  "duration_s": 10.95,
  "samples": 217,
  "state_sequence": [
    "TARGET_LOST_HOLD",
    "HOVER_SEARCH",
    "ALIGN_YAW",
    "ALIGN_VERTICAL",
    "FOLLOW",
    "TARGET_LOST_HOLD"
  ],
  "final_state": "TARGET_LOST_HOLD",
  "final_reason": "acoustic_observation_invalid",
  "start_drone_alt_m": 5.1,
  "end_drone_alt_m": 5.15,
  "end_target_alt_m": 5.0,
  "remaining_alt_diff_m": -0.15,
  "end_yaw_deg": 0.5,
  "end_meas_az_deg": -29.3,
  "end_meas_el_deg": -0.1,
  "start_horiz_dist_m": 9.01,
  "end_horiz_dist_m": 7.01,
  "horiz_progress_m": 2.0,
  "max_forward_cmd_m_s": 0.5,
  "min_down_cmd_m_s": 0.0,
  "max_down_cmd_m_s": 0.0,
  "mean_az_err_deg": 10.54,
  "mean_el_err_deg": 13.13
}
```
- **Plot:** [S4_unreliable_elevation.png](file:///home/dhia/Desktop/projects/freq_project/frequency_id/sim/reports/plots/S4_unreliable_elevation.png)

### `S5_telemetry_stale_abort` — PASS
- **Summary:** final=ABORTED (flight_telemetry_stale)
```json
{
  "duration_s": 7.55,
  "samples": 150,
  "state_sequence": [
    "TARGET_LOST_HOLD",
    "HOVER_SEARCH",
    "ALIGN_YAW",
    "ALIGN_VERTICAL",
    "FOLLOW",
    "ABORTED"
  ],
  "final_state": "ABORTED",
  "final_reason": "flight_telemetry_stale",
  "start_drone_alt_m": 5.03,
  "end_drone_alt_m": 5.43,
  "end_target_alt_m": 5.0,
  "remaining_alt_diff_m": -0.43,
  "end_yaw_deg": -0.1,
  "end_meas_az_deg": 1.9,
  "end_meas_el_deg": -2.5,
  "start_horiz_dist_m": 8.78,
  "end_horiz_dist_m": 7.52,
  "horiz_progress_m": 1.25,
  "max_forward_cmd_m_s": 0.5,
  "min_down_cmd_m_s": 0.0,
  "max_down_cmd_m_s": 0.0,
  "mean_az_err_deg": 0.08,
  "mean_el_err_deg": 0.47
}
```
- **Plot:** [S5_telemetry_stale_abort.png](file:///home/dhia/Desktop/projects/freq_project/frequency_id/sim/reports/plots/S5_telemetry_stale_abort.png)

### `S6_altitude_limits` — PASS
- **Summary:** final=ABORTED (altitude_envelope_violation), end_alt=6.02m (limit=6.0m), cmd_down=0.0
```json
{
  "duration_s": 4.1,
  "samples": 82,
  "state_sequence": [
    "TARGET_LOST_HOLD",
    "HOVER_SEARCH",
    "ALIGN_YAW",
    "ALIGN_VERTICAL",
    "ABORTED"
  ],
  "final_state": "ABORTED",
  "final_reason": "altitude_envelope_violation",
  "start_drone_alt_m": 5.09,
  "end_drone_alt_m": 6.02,
  "end_target_alt_m": 8.5,
  "remaining_alt_diff_m": 2.48,
  "end_yaw_deg": -0.1,
  "end_meas_az_deg": 0.5,
  "end_meas_el_deg": 17.7,
  "start_horiz_dist_m": 8.08,
  "end_horiz_dist_m": 8.13,
  "horiz_progress_m": -0.05,
  "max_forward_cmd_m_s": 0.0,
  "min_down_cmd_m_s": -0.3,
  "max_down_cmd_m_s": 0.0,
  "mean_az_err_deg": 0.07,
  "mean_el_err_deg": 0.49
}
```
- **Plot:** [S6_altitude_limits.png](file:///home/dhia/Desktop/projects/freq_project/frequency_id/sim/reports/plots/S6_altitude_limits.png)

### `S7_geofence_violation` — PASS
- **Summary:** final=ABORTED (geofence_violation), progress=2.51m
```json
{
  "duration_s": 9.53,
  "samples": 189,
  "state_sequence": [
    "TARGET_LOST_HOLD",
    "HOVER_SEARCH",
    "ALIGN_YAW",
    "ALIGN_VERTICAL",
    "FOLLOW",
    "ABORTED"
  ],
  "final_state": "ABORTED",
  "final_reason": "geofence_violation",
  "start_drone_alt_m": 5.14,
  "end_drone_alt_m": 6.14,
  "end_target_alt_m": 5.0,
  "remaining_alt_diff_m": -1.14,
  "end_yaw_deg": -0.0,
  "end_meas_az_deg": 0.0,
  "end_meas_el_deg": -6.5,
  "start_horiz_dist_m": 11.95,
  "end_horiz_dist_m": 9.44,
  "horiz_progress_m": 2.51,
  "max_forward_cmd_m_s": 0.5,
  "min_down_cmd_m_s": 0.0,
  "max_down_cmd_m_s": 0.0,
  "mean_az_err_deg": 0.05,
  "mean_el_err_deg": 0.31
}
```
- **Plot:** [S7_geofence_violation.png](file:///home/dhia/Desktop/projects/freq_project/frequency_id/sim/reports/plots/S7_geofence_violation.png)

### `S8_manual_override` — PASS
- **Summary:** final=ABORTED (offboard_lost_or_manual_override)
```json
{
  "duration_s": 8.71,
  "samples": 173,
  "state_sequence": [
    "TARGET_LOST_HOLD",
    "HOVER_SEARCH",
    "ALIGN_YAW",
    "ALIGN_VERTICAL",
    "FOLLOW",
    "ALIGN_VERTICAL",
    "ABORTED"
  ],
  "final_state": "ABORTED",
  "final_reason": "offboard_lost_or_manual_override",
  "start_drone_alt_m": 5.15,
  "end_drone_alt_m": 6.12,
  "end_target_alt_m": 5.0,
  "remaining_alt_diff_m": -1.12,
  "end_yaw_deg": -1.1,
  "end_meas_az_deg": 0.5,
  "end_meas_el_deg": -11.6,
  "start_horiz_dist_m": 8.93,
  "end_horiz_dist_m": 6.88,
  "horiz_progress_m": 2.06,
  "max_forward_cmd_m_s": 0.5,
  "min_down_cmd_m_s": 0.0,
  "max_down_cmd_m_s": 0.21,
  "mean_az_err_deg": 0.06,
  "mean_el_err_deg": 0.55
}
```
- **Plot:** [S8_manual_override.png](file:///home/dhia/Desktop/projects/freq_project/frequency_id/sim/reports/plots/S8_manual_override.png)

### `S9_offboard_connection_loss` — PASS
- **Summary:** final=ABORTED (offboard_lost_or_manual_override)
```json
{
  "duration_s": 8.4,
  "samples": 167,
  "state_sequence": [
    "TARGET_LOST_HOLD",
    "HOVER_SEARCH",
    "ALIGN_YAW",
    "ALIGN_VERTICAL",
    "FOLLOW",
    "ABORTED"
  ],
  "final_state": "ABORTED",
  "final_reason": "offboard_lost_or_manual_override",
  "start_drone_alt_m": 5.11,
  "end_drone_alt_m": 5.32,
  "end_target_alt_m": 5.0,
  "remaining_alt_diff_m": -0.32,
  "end_yaw_deg": -0.8,
  "end_meas_az_deg": -1.0,
  "end_meas_el_deg": -5.6,
  "start_horiz_dist_m": 8.9,
  "end_horiz_dist_m": 6.76,
  "horiz_progress_m": 2.13,
  "max_forward_cmd_m_s": 0.5,
  "min_down_cmd_m_s": 0.0,
  "max_down_cmd_m_s": 0.0,
  "mean_az_err_deg": 0.07,
  "mean_el_err_deg": 0.37
}
```
- **Plot:** [S9_offboard_connection_loss.png](file:///home/dhia/Desktop/projects/freq_project/frequency_id/sim/reports/plots/S9_offboard_connection_loss.png)

### `T0_subprocess_px4_controller` — PASS
- **Summary:** Real px4_controller.py subprocess exit=0; yaw -1.2°->29.2°, alt 5.15m->8.28m, dist 9.58m->5.00m, states_seen=True, offboard_released=True
```json
{
  "exit_code": 0,
  "start_yaw_deg": -1.2,
  "end_yaw_deg": 29.2,
  "start_alt_m": 5.15,
  "end_alt_m": 8.28,
  "horiz_progress_m": 4.58,
  "offboard_released": true,
  "stderr_tail": [
    "[GUIDANCE] FOLLOW following_direction_without_range forward=0.50 down=0.00 yaw_rate=0.00",
    "[GUIDANCE] FOLLOW following_direction_without_range forward=0.50 down=0.00 yaw_rate=0.00",
    "[GUIDANCE] ALIGN_VERTICAL aligning_vertical forward=0.00 down=0.22 yaw_rate=0.00",
    "[GUIDANCE] ALIGN_VERTICAL aligning_vertical forward=0.00 down=0.21 yaw_rate=0.00",
    "[GUIDANCE] ALIGN_VERTICAL aligning_vertical forward=0.00 down=0.16 yaw_rate=0.00",
    "[GUIDANCE] ALIGN_VERTICAL aligning_vertical forward=0.00 down=0.15 yaw_rate=0.00",
    "[GUIDANCE] ALIGN_VERTICAL aligning_vertical forward=0.00 down=0.14 yaw_rate=0.00",
    "[GUIDANCE] ALIGN_VERTICAL aligning_vertical forward=0.00 down=0.14 yaw_rate=0.00",
    "[GUIDANCE] ALIGN_VERTICAL aligning_vertical forward=0.00 down=0.14 yaw_rate=0.00",
    "[GUIDANCE] ALIGN_VERTICAL aligning_vertical forward=0.00 down=0.14 yaw_rate=0.00",
    "[GUIDANCE] ALIGN_VERTICAL aligning_vertical forward=0.00 down=0.15 yaw_rate=0.00",
    "[GUIDANCE] ABORTED acoustic_stream_closed forward=0.00 down=0.00 yaw_rate=0.00"
  ]
}
```
