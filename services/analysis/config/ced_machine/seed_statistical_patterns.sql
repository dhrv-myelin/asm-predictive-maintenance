-- seed_statistical_patterns.sql
 
INSERT INTO statistical_patterns (id, statistical_pattern, likely_causes_and_what_to_inspect, metric_name, type_of_motor)
VALUES
 
-- ============================================================
-- ACTUATOR METRICS
-- Patterns: Trend Acceleration, Step Jump, Random Spikes,
--           Increasing Outlier Frequency, Variance Growth,
--           Baseline Shift, Slow Drift
-- ============================================================
 
-- actuator: entry_stopper_lowering_time
(1,  'Trend Acceleration',          'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                    'entry_stopper_lowering_time',  'pneumatic_actuator'),
(2,  'Step Jump',                   'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                    'entry_stopper_lowering_time',  'pneumatic_actuator'),
(3,  'Random Spikes',               'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                           'entry_stopper_lowering_time',  'pneumatic_actuator'),
(4,  'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                       'entry_stopper_lowering_time',  'pneumatic_actuator'),
(5,  'Variance Growth',             'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                      'entry_stopper_lowering_time',  'pneumatic_actuator'),
--(6,  'Baseline Shift',              'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                       'entry_stopper_lowering_time',  'pneumatic_actuator'),
--(7,  'Slow Drift',                  'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                'entry_stopper_lowering_time',  'pneumatic_actuator'),

-- actuator: entry_stopper_raising_time
(6,  'Trend Acceleration',          'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                    'entry_stopper_raising_time',   'pneumatic_actuator'),
(7,  'Step Jump',                   'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                    'entry_stopper_raising_time',   'pneumatic_actuator'),
(8, 'Random Spikes',               'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                           'entry_stopper_raising_time',   'pneumatic_actuator'),
(9, 'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                       'entry_stopper_raising_time',   'pneumatic_actuator'),
(10, 'Variance Growth',             'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                      'entry_stopper_raising_time',   'pneumatic_actuator'),
--(13, 'Baseline Shift',              'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                       'entry_stopper_raising_time',   'pneumatic_actuator'),
--(14, 'Slow Drift',                  'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                'entry_stopper_raising_time',   'pneumatic_actuator'),

-- actuator: pallet_clamping_time
(11, 'Trend Acceleration',          'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                    'pallet_clamping_time',         'pneumatic_actuator'),
(12, 'Step Jump',                   'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                    'pallet_clamping_time',         'pneumatic_actuator'),
(13, 'Random Spikes',               'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                           'pallet_clamping_time',         'pneumatic_actuator'),
(14, 'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                       'pallet_clamping_time',         'pneumatic_actuator'),
(15, 'Variance Growth',             'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                      'pallet_clamping_time',         'pneumatic_actuator'),
--(20, 'Baseline Shift',              'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                       'pallet_clamping_time',         'pneumatic_actuator'),
--(21, 'Slow Drift',                  'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                'pallet_clamping_time',         'pneumatic_actuator'),

-- actuator: pallet_lifting_time
(16, 'Trend Acceleration',          'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                    'pallet_lifting_time',          'pneumatic_actuator'),
(17, 'Step Jump',                   'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                    'pallet_lifting_time',          'pneumatic_actuator'),
(18, 'Random Spikes',               'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                           'pallet_lifting_time',          'pneumatic_actuator'),
(19, 'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                       'pallet_lifting_time',          'pneumatic_actuator'),
(20, 'Variance Growth',             'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                      'pallet_lifting_time',          'pneumatic_actuator'),
--(27, 'Baseline Shift',              'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                       'pallet_lifting_time',          'pneumatic_actuator'),
--(28, 'Slow Drift',                  'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                'pallet_lifting_time',          'pneumatic_actuator'),

-- actuator: pallet_unclamping_time
(21, 'Trend Acceleration',          'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                    'pallet_unclamping_time',       'pneumatic_actuator'),
(22, 'Step Jump',                   'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                    'pallet_unclamping_time',       'pneumatic_actuator'),
(23, 'Random Spikes',               'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                           'pallet_unclamping_time',       'pneumatic_actuator'),
(24, 'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                       'pallet_unclamping_time',       'pneumatic_actuator'),
(25, 'Variance Growth',             'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                      'pallet_unclamping_time',       'pneumatic_actuator'),
--(34, 'Baseline Shift',              'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                       'pallet_unclamping_time',       'pneumatic_actuator'),
--(35, 'Slow Drift',                  'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                'pallet_unclamping_time',       'pneumatic_actuator'),

-- actuator: pallet_lowering_time
(26, 'Trend Acceleration',          'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                    'pallet_lowering_time',         'pneumatic_actuator'),
(27, 'Step Jump',                   'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                    'pallet_lowering_time',         'pneumatic_actuator'),
(28, 'Random Spikes',               'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                           'pallet_lowering_time',         'pneumatic_actuator'),
(29, 'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                       'pallet_lowering_time',         'pneumatic_actuator'),
(30, 'Variance Growth',             'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                      'pallet_lowering_time',         'pneumatic_actuator'),
--(41, 'Baseline Shift',              'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                       'pallet_lowering_time',         'pneumatic_actuator'),
--(42, 'Slow Drift',                  'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                'pallet_lowering_time',         'pneumatic_actuator'),

-- actuator: exit_stopper_lowering_time
(31, 'Trend Acceleration',          'Friction approaching motor torque limit. Inspect conveyor rollers, pallet underside wear, belt tension, and motor temperature.',                                                       'exit_stopper_lowering_time',   'pneumatic_actuator'),
(32, 'Step Jump',                   'Local mechanical obstruction or seized roller creating higher drag. Manually rotate rollers, check debris or pallet interference along travel path.',                                  'exit_stopper_lowering_time',   'pneumatic_actuator'),
(33, 'Random Spikes',               'Transient pallet interference, vibration, or electrical noise affecting sensor or drive. Correlate timestamps with nearby machine events and inspect sensors and cabling.',            'exit_stopper_lowering_time',   'pneumatic_actuator'),
--(46, 'Periodic Oscillation',        'Rotating mechanical defect (eccentric pulley, worn roller, belt irregularity). Inspect rollers and pulleys for periodic drag or misalignment.',                                       'exit_stopper_lowering_time',   'pneumatic_actuator'),
(34, 'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                       'exit_stopper_lowering_time',   'pneumatic_actuator'),
(35, 'Variance Growth',             'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                      'exit_stopper_lowering_time',   'pneumatic_actuator'),
--(49, 'Baseline Shift',              'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                       'exit_stopper_lowering_time',   'pneumatic_actuator'),
--(50, 'Slow Drift',                  'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                'exit_stopper_lowering_time',   'pneumatic_actuator'),

-- actuator: exit_stopper_raising_time
(36, 'Trend Acceleration',          'Friction approaching motor torque limit. Inspect conveyor rollers, pallet underside wear, belt tension, and motor temperature.',                                                       'exit_stopper_raising_time',    'pneumatic_actuator'),
(37, 'Step Jump',                   'Local mechanical obstruction or seized roller creating higher drag. Manually rotate rollers, check debris or pallet interference along travel path.',                                  'exit_stopper_raising_time',    'pneumatic_actuator'),
(38, 'Random Spikes',               'Transient pallet interference, vibration, or electrical noise affecting sensor or drive. Correlate timestamps with nearby machine events and inspect sensors and cabling.',            'exit_stopper_raising_time',    'pneumatic_actuator'),
--(54, 'Periodic Oscillation',        'Rotating mechanical defect (eccentric pulley, worn roller, belt irregularity). Inspect rollers and pulleys for periodic drag or misalignment.',                                       'exit_stopper_raising_time',    'pneumatic_actuator'),
(39, 'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                       'exit_stopper_raising_time',    'pneumatic_actuator'),
(40, 'Variance Growth',             'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                      'exit_stopper_raising_time',    'pneumatic_actuator'),
--(57, 'Baseline Shift',              'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                       'exit_stopper_raising_time',    'pneumatic_actuator'),
--(58, 'Slow Drift',                  'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                'exit_stopper_raising_time',    'pneumatic_actuator'),

-- ============================================================
-- STEPPER METRICS
-- Patterns: Trend Acceleration, Step Jump, Random Spikes,
--           Periodic Oscillation, Increasing Outlier Frequency,
--           Variance Growth, Baseline Shift, Slow Drift
-- ============================================================
 
-- stepper: pallet_moveout_time
(41, 'Trend Acceleration',          'Friction approaching motor torque limit. Inspect conveyor rollers, pallet underside wear, belt tension, and motor temperature.',                                                       'pallet_moveout_time',          'stepper'),
(42, 'Step Jump',                   'Local mechanical obstruction or seized roller creating higher drag. Manually rotate rollers, check debris or pallet interference along travel path.',                                  'pallet_moveout_time',          'stepper'),
(43, 'Random Spikes',               'Transient pallet interference, vibration, or electrical noise affecting sensor or drive. Correlate timestamps with nearby machine events and inspect sensors and cabling.',            'pallet_moveout_time',          'stepper'),
(44, 'Increasing Outlier Frequency','Lost steps from mechanical overload or driver thermal throttling. Inspect coupling tightness, verify driver current limits, check for resonance at operating speeds.',                 'pallet_moveout_time',          'stepper'),
(45, 'Variance Growth',             'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                      'pallet_moveout_time',          'stepper'),
--(64, 'Baseline Shift',              'Persistent load increase or driver current limit change. Verify driver settings, inspect mechanical load, check belt tension and coupling.',                                           'pallet_moveout_time',          'stepper'),
--(65, 'Slow Drift',                  'Gradual belt elongation, bearing wear, or thermal effects on driver. Inspect belt tension, lubricate bearings, monitor driver temperature.',                                           'pallet_moveout_time',          'stepper'),
 
-- stepper: pallet_movein_time
(46, 'Trend Acceleration',          'Friction approaching motor torque limit. Inspect conveyor rollers, pallet underside wear, belt tension, and motor temperature.',                                                       'pallet_movein_time',           'stepper'),
(47, 'Step Jump',                   'Local mechanical obstruction or seized roller creating higher drag. Manually rotate rollers, check debris or pallet interference along travel path.',                                  'pallet_movein_time',           'stepper'),
(48, 'Random Spikes',               'Transient pallet interference, vibration, or electrical noise affecting sensor or drive. Correlate timestamps with nearby machine events and inspect sensors and cabling.',            'pallet_movein_time',           'stepper'),
--(69, 'Periodic Oscillation',        'Rotating mechanical defect (eccentric pulley, worn roller, belt irregularity). Inspect rollers and pulleys for periodic drag or misalignment.',                                       'pallet_movein_time',           'stepper'),
(49, 'Increasing Outlier Frequency','Lost steps from mechanical overload or driver thermal throttling. Inspect coupling tightness, verify driver current limits, check for resonance at operating speeds.',                 'pallet_movein_time',           'stepper'),
(50, 'Variance Growth',             'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                      'pallet_movein_time',           'stepper'),
--(72, 'Baseline Shift',              'Persistent load increase or driver current limit change. Verify driver settings, inspect mechanical load, check belt tension and coupling.',                                           'pallet_movein_time',           'stepper'),
--(73, 'Slow Drift',                  'Gradual belt elongation, bearing wear, or thermal effects on driver. Inspect belt tension, lubricate bearings, monitor driver temperature.',                                           'pallet_movein_time',           'stepper'),
 
-- ============================================================
-- SERVO METRICS
-- Patterns: Trend Acceleration, Step Jump, Random Spikes,
--           Periodic Oscillation, Increasing Outlier Frequency,
--           Baseline Shift, Slow Drift
-- ============================================================
 
-- servo: cavity_1_dispensing_time
(51, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                 'cavity_1_dispensing_time',     'servo'),
(52, 'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                           'cavity_1_dispensing_time',     'servo'),
(53, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                               'cavity_1_dispensing_time',     'servo'),
--(77, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                         'cavity_1_dispensing_time',     'servo'),
(54, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',            'cavity_1_dispensing_time',     'servo'),
--(79, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                             'cavity_1_dispensing_time',     'servo'),
--(80, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'cavity_1_dispensing_time',     'servo'),
 
-- servo: cavity_2_dispensing_time
(55, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                 'cavity_2_dispensing_time',     'servo'),
(56, 'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                           'cavity_2_dispensing_time',     'servo'),
(57, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                               'cavity_2_dispensing_time',     'servo'),
--(84, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                         'cavity_2_dispensing_time',     'servo'),
(58, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',            'cavity_2_dispensing_time',     'servo'),
--(86, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                             'cavity_2_dispensing_time',     'servo'),
--(87, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'cavity_2_dispensing_time',     'servo'),
 
-- servo: cavity_3_dispensing_time
(59, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                 'cavity_3_dispensing_time',     'servo'),
(60, 'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                           'cavity_3_dispensing_time',     'servo'),
(61, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                               'cavity_3_dispensing_time',     'servo'),
--(91, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                         'cavity_3_dispensing_time',     'servo'),
(62, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',            'cavity_3_dispensing_time',     'servo'),
--(93, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                             'cavity_3_dispensing_time',     'servo'),
--(94, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'cavity_3_dispensing_time',     'servo'),
 
-- servo: cavity_4_dispensing_time
(63,  'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                'cavity_4_dispensing_time',     'servo'),
(64,  'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                          'cavity_4_dispensing_time',     'servo'),
(65,  'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                              'cavity_4_dispensing_time',     'servo'),
--(98,  'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                        'cavity_4_dispensing_time',     'servo'),
(66,  'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',           'cavity_4_dispensing_time',     'servo'),
--(100, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                            'cavity_4_dispensing_time',     'servo'),
--(101, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'cavity_4_dispensing_time',     'servo'),
 
-- servo: cavity_5_dispensing_time
(67, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                'cavity_5_dispensing_time',     'servo'),
(68, 'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                          'cavity_5_dispensing_time',     'servo'),
(69, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                              'cavity_5_dispensing_time',     'servo'),
--(105, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                        'cavity_5_dispensing_time',     'servo'),
(70, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',           'cavity_5_dispensing_time',     'servo'),
--(107, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                            'cavity_5_dispensing_time',     'servo'),
--(108, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'cavity_5_dispensing_time',     'servo'),
 
-- servo: cavity_6_dispensing_time
(71, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                'cavity_6_dispensing_time',     'servo'),
(72, 'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                          'cavity_6_dispensing_time',     'servo'),
(73, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                              'cavity_6_dispensing_time',     'servo'),
--(112, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                        'cavity_6_dispensing_time',     'servo'),
(74, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',           'cavity_6_dispensing_time',     'servo'),
--(114, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                            'cavity_6_dispensing_time',     'servo'),
--(115, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'cavity_6_dispensing_time',     'servo'),
 
-- servo: inspection_time
(75, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                'inspection_time',              'servo'),
(76, 'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                          'inspection_time',              'servo'),
(77, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                              'inspection_time',              'servo'),
--(119, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                        'inspection_time',              'servo'),
(78, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',           'inspection_time',              'servo'),
--(121, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                            'inspection_time',              'servo'),
--(122, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'inspection_time',              'servo'),

-- servo: gantry_safety_positioning_time
(79, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                'gantry_safety_positioning_time', 'servo'),
(80, 'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                          'gantry_safety_positioning_time', 'servo'),
(81, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                              'gantry_safety_positioning_time', 'servo'),
--(126, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                        'gantry_safety_positioning_time', 'servo'),
(82, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',           'gantry_safety_positioning_time', 'servo'),
--(128, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                            'gantry_safety_positioning_time', 'servo'),
--(129, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'gantry_safety_positioning_time', 'servo'),

-- servo: maint_move_to_safe_time
(83, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                'maint_move_to_safe_time',      'servo'),
(84, 'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                          'maint_move_to_safe_time',      'servo'),
(85, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                              'maint_move_to_safe_time',      'servo'),
--(133, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                        'maint_move_to_safe_time',      'servo'),
(86, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',           'maint_move_to_safe_time',      'servo'),
--(135, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                            'maint_move_to_safe_time',      'servo'),
--(136, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'maint_move_to_safe_time',      'servo'),


-- servo: maint_between_glue_purge_delay
(87, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                'maint_move_to_safe_time',      'servo'),
(88, 'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                          'maint_move_to_safe_time',      'servo'),
(89, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                              'maint_move_to_safe_time',      'servo'),
--(140, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                        'maint_move_to_safe_time',      'servo'),
(90, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',           'maint_move_to_safe_time',      'servo'),
--(142, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                            'maint_move_to_safe_time',      'servo'),
--(143, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'maint_move_to_safe_time',      'servo'),

-- servo: maint_between_nozzle_clean_delay
(91, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                'maint_move_to_safe_time',      'servo'),
(92, 'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                          'maint_move_to_safe_time',      'servo'),
(93, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                              'maint_move_to_safe_time',      'servo'),
--(147, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                        'maint_move_to_safe_time',      'servo'),
(94, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',           'maint_move_to_safe_time',      'servo'),
--(149, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                            'maint_move_to_safe_time',      'servo'),
--(150, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'maint_move_to_safe_time',      'servo'),

-- servo: maint_glue_purging_time
(95, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                'maint_move_to_safe_time',      'servo'),
(96, 'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                          'maint_move_to_safe_time',      'servo'),
(97, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                              'maint_move_to_safe_time',      'servo'),
--(154, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                        'maint_move_to_safe_time',      'servo'),
(98, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',           'maint_move_to_safe_time',      'servo'),
--(156, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                            'maint_move_to_safe_time',      'servo'),
--(157, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'maint_move_to_safe_time',      'servo'),

-- servo: maint_nozzle_cleaning_time
(99, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                'maint_move_to_safe_time',      'servo'),
(100, 'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                          'maint_move_to_safe_time',      'servo'),
(101, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                              'maint_move_to_safe_time',      'servo'),
--(161, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                        'maint_move_to_safe_time',      'servo'),
(102, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',           'maint_move_to_safe_time',      'servo'),
--(163, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                            'maint_move_to_safe_time',      'servo'),
--(164, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'maint_move_to_safe_time',      'servo'),

--servo: maint_post_glue_purge_delay
(103, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                'maint_move_to_safe_time',      'servo'),
(104, 'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                          'maint_move_to_safe_time',      'servo'),
(105, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                              'maint_move_to_safe_time',      'servo'),
--(168, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                        'maint_move_to_safe_time',      'servo'),
(106, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',           'maint_move_to_safe_time',      'servo'),
--(170, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                            'maint_move_to_safe_time',      'servo'),
--(171, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'maint_move_to_safe_time',      'servo'),

--servo: maint_post_nozzle_clean_delay
(107, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                'maint_move_to_safe_time',      'servo'),
(108, 'Step Jump',                   'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                          'maint_move_to_safe_time',      'servo'),
(109, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                              'maint_move_to_safe_time',      'servo'),
--(175, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                        'maint_move_to_safe_time',      'servo'),
(110, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',           'maint_move_to_safe_time',      'servo');
--(177, 'Baseline Shift',              'Persistent load change or servo parameter update. Verify PID tuning, check mechanical preload, inspect ball screw and guide lubrication.',                                            'maint_move_to_safe_time',      'servo'),
--(178, 'Slow Drift',                  'Progressive bearing wear, thermal expansion, or lubrication degradation. Monitor motor temperature, lubricate guides and screw, track torque trend.',                                 'maint_move_to_safe_time',      'servo');

