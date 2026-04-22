INSERT INTO statistical_patterns (id, statistical_pattern, likely_causes_and_what_to_inspect, metric_name, type_of_motor)
VALUES

-- ============================================================
-- ACTUATOR METRICS
-- ============================================================

-- actuator: clamping_time
(1,  'Trend Acceleration',          'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                   'clamping_time',           'pneumatic_actuator'),
(2,  'Step Jump',                   'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                   'clamping_time',           'pneumatic_actuator'),
(3,  'Random Spikes',               'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                          'clamping_time',           'pneumatic_actuator'),
(4,  'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                      'clamping_time',           'pneumatic_actuator'),
(5,  'Variance Growth',             'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                     'clamping_time',           'pneumatic_actuator'),
--(6,  'Baseline Shift',              'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                      'clamping_time',           'pneumatic_actuator'),
--(7,  'Slow Drift',                  'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                               'clamping_time',           'pneumatic_actuator'),

-- actuator: pre_data_handshake_wait
(6,  'Trend Acceleration',          'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                   'pre_data_handshake_wait', 'pneumatic_actuator'),
(7,  'Step Jump',                   'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                   'pre_data_handshake_wait', 'pneumatic_actuator'),
(8, 'Random Spikes',               'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                          'pre_data_handshake_wait', 'pneumatic_actuator'),
(9, 'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                      'pre_data_handshake_wait', 'pneumatic_actuator'),
(10, 'Variance Growth',             'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                     'pre_data_handshake_wait', 'pneumatic_actuator'),
--(13, 'Baseline Shift',              'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                      'pre_data_handshake_wait', 'pneumatic_actuator'),
--(14, 'Slow Drift',                  'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                               'pre_data_handshake_wait', 'pneumatic_actuator'),

-- ============================================================
-- SERVO METRICS
-- ============================================================

-- servo: z_axis_positioning_time
(11, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                'z_axis_positioning_time', 'servo'),
(12, 'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                          'z_axis_positioning_time', 'servo'),
(13, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                              'z_axis_positioning_time', 'servo'),
--(18, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                        'z_axis_positioning_time', 'servo'),
(14, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',           'z_axis_positioning_time', 'servo'),
--(20, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                            'z_axis_positioning_time', 'servo'),
--(21, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'z_axis_positioning_time', 'servo'),

-- servo: gantry_positioning_time
(15, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                'gantry_positioning_time','servo'),
(16, 'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                          'gantry_positioning_time','servo'),
(17, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                              'gantry_positioning_time','servo'),
--(25, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                        'gantry_positioning_time','servo'),
(18, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',           'gantry_positioning_time','servo'),
--(27, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                            'gantry_positioning_time','servo'),
--(28, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'gantry_positioning_time','servo'),

-- servo: z_axis_homing_time
(19, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                'z_axis_homing_time',     'servo'),
(20, 'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                          'z_axis_homing_time',     'servo'),
(21, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                              'z_axis_homing_time',     'servo'),
--(32, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                        'z_axis_homing_time',     'servo'),
(22, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',           'z_axis_homing_time',     'servo'),
--(34, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                            'z_axis_homing_time',     'servo'),
--(35, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'z_axis_homing_time',     'servo');

-- ============================================================
-- GALVO METRICS
-- ============================================================

-- galvo: marking_galvo_positioning_time

(23, 'Step Jump','Sudden degradation due to cable fault, thermal threshold in driver, or mirror contamination. Inspect connectors, driver condition, and mirror surface.','marking_galvo_positioning_time', 'galvo'),

(24, 'Random Spikes','Handshake dropouts or EMI disrupting position confirmation. Inspect shielding, grounding, and cable routing near high-current lines.','marking_galvo_positioning_time', 'galvo'),

(25, 'Increasing Outlier Frequency','Confirmation failures becoming dominant; likely degraded position detector or timeout mismatch. Check detector alignment and controller timeout settings.','marking_galvo_positioning_time', 'galvo'),

(26, 'Variance Growth','Inconsistent settle due to position detector signal degradation or driver thermal instability. Check detector cable, shielding, and driver gain/cooling.','marking_galvo_positioning_time', 'galvo');

-- (27, 'Baseline Shift', 'Persistent degradation not recovering between shifts; likely wear, detector misalignment, or driver component drift. Inspect bearing, detector mount, and driver health.', 'marking_galvo_positioning_time', 'galvo');

-- (28, 'Slow Drift','Driver thermal buildup or mirror bearing drag increasing settle time. Inspect galvo 2 driver temperature, cooling, and mirror resistance.','marking_galvo_positioning_time', 'galvo');