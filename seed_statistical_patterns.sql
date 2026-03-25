-- seed_statistical_patterns.sql
 
INSERT INTO statistical_patterns (id, statistical_pattern, likely_causes_and_what_to_inspect, metric_name, type_of_motor)
VALUES
 
-- ============================================================
-- ACTUATOR METRICS
-- Patterns: Trend Acceleration, Step Jump, Random Spikes,
--           Increasing Outlier Frequency
-- ============================================================
 
-- actuator: entry_stopper_lowering_time
(1,  'Trend Acceleration',          'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                    'entry_stopper_lowering_time',  'pneumatic_actuator'),
(2,  'Step Jump',                   'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                    'entry_stopper_lowering_time',  'pneumatic_actuator'),
(3,  'Random Spikes',               'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                           'entry_stopper_lowering_time',  'pneumatic_actuator'),
(4,  'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                       'entry_stopper_lowering_time',  'pneumatic_actuator'),
 
-- actuator: entry_stopper_raising_time
(5,  'Trend Acceleration',          'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                    'entry_stopper_raising_time',   'pneumatic_actuator'),
(6,  'Step Jump',                   'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                    'entry_stopper_raising_time',   'pneumatic_actuator'),
(7,  'Random Spikes',               'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                           'entry_stopper_raising_time',   'pneumatic_actuator'),
(8,  'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                       'entry_stopper_raising_time',   'pneumatic_actuator'),
 
-- actuator: pallet_clamping_time
(9,  'Trend Acceleration',          'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                    'pallet_clamping_time',         'pneumatic_actuator'),
(10, 'Step Jump',                   'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                    'pallet_clamping_time',         'pneumatic_actuator'),
(11, 'Random Spikes',               'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                           'pallet_clamping_time',         'pneumatic_actuator'),
(12, 'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                       'pallet_clamping_time',         'pneumatic_actuator'),
 
-- actuator: pallet_lifting_time
(13, 'Trend Acceleration',          'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                    'pallet_lifting_time',          'pneumatic_actuator'),
(14, 'Step Jump',                   'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                    'pallet_lifting_time',          'pneumatic_actuator'),
(15, 'Random Spikes',               'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                           'pallet_lifting_time',          'pneumatic_actuator'),
(16, 'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                       'pallet_lifting_time',          'pneumatic_actuator'),
 
-- actuator: pallet_unclamping_time
(17, 'Trend Acceleration',          'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                    'pallet_unclamping_time',       'pneumatic_actuator'),
(18, 'Step Jump',                   'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                    'pallet_unclamping_time',       'pneumatic_actuator'),
(19, 'Random Spikes',               'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                           'pallet_unclamping_time',       'pneumatic_actuator'),
(20, 'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                       'pallet_unclamping_time',       'pneumatic_actuator'),
 
-- actuator: pallet_lowering_time
(21, 'Trend Acceleration',          'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                    'pallet_lowering_time',         'pneumatic_actuator'),
(22, 'Step Jump',                   'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                    'pallet_lowering_time',         'pneumatic_actuator'),
(23, 'Random Spikes',               'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                           'pallet_lowering_time',         'pneumatic_actuator'),
(24, 'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                       'pallet_lowering_time',         'pneumatic_actuator'),
 
-- actuator: exit_stopper_lowering_time
(25, 'Trend Acceleration',          'Friction approaching motor torque limit. Inspect conveyor rollers, pallet underside wear, belt tension, and motor temperature.',                                                       'exit_stopper_lowering_time',   'pneumatic_actuator'),
(26, 'Step Jump',                   'Local mechanical obstruction or seized roller creating higher drag. Manually rotate rollers, check debris or pallet interference along travel path.',                                  'exit_stopper_lowering_time',   'pneumatic_actuator'),
(27, 'Random Spikes',               'Transient pallet interference, vibration, or electrical noise affecting sensor or drive. Correlate timestamps with nearby machine events and inspect sensors and cabling.',            'exit_stopper_lowering_time',   'pneumatic_actuator'),
(28, 'Periodic Oscillation',        'Rotating mechanical defect (eccentric pulley, worn roller, belt irregularity). Inspect rollers and pulleys for periodic drag or misalignment.',                                       'exit_stopper_lowering_time',   'pneumatic_actuator'),
(29, 'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                       'exit_stopper_lowering_time',   'pneumatic_actuator'),
 
-- ============================================================
-- STEPPER METRICS
-- Patterns: Trend Acceleration, Step Jump, Random Spikes,
--           Periodic Oscillation, Increasing Outlier Frequency
-- ============================================================
 
-- stepper: pallet_moveout_time
(30, 'Trend Acceleration',          'Friction approaching motor torque limit. Inspect conveyor rollers, pallet underside wear, belt tension, and motor temperature.',                                                       'pallet_moveout_time',          'stepper'),
(31, 'Step Jump',                   'Local mechanical obstruction or seized roller creating higher drag. Manually rotate rollers, check debris or pallet interference along travel path.',                                  'pallet_moveout_time',          'stepper'),
(32, 'Random Spikes',               'Transient pallet interference, vibration, or electrical noise affecting sensor or drive. Correlate timestamps with nearby machine events and inspect sensors and cabling.',            'pallet_moveout_time',          'stepper'),
(33, 'Increasing Outlier Frequency','Lost steps from mechanical overload or driver thermal throttling. Inspect coupling tightness, verify driver current limits, check for resonance at operating speeds.',                 'pallet_moveout_time',          'stepper'),
 
-- stepper: pallet_movein_time
(34, 'Trend Acceleration',          'Friction approaching motor torque limit. Inspect conveyor rollers, pallet underside wear, belt tension, and motor temperature.',                                                       'pallet_movein_time',           'stepper'),
(35, 'Step Jump',                   'Local mechanical obstruction or seized roller creating higher drag. Manually rotate rollers, check debris or pallet interference along travel path.',                                  'pallet_movein_time',           'stepper'),
(36, 'Random Spikes',               'Transient pallet interference, vibration, or electrical noise affecting sensor or drive. Correlate timestamps with nearby machine events and inspect sensors and cabling.',            'pallet_movein_time',           'stepper'),
(37, 'Periodic Oscillation',        'Rotating mechanical defect (eccentric pulley, worn roller, belt irregularity). Inspect rollers and pulleys for periodic drag or misalignment.',                                       'pallet_movein_time',           'stepper'),
(38, 'Increasing Outlier Frequency','Lost steps from mechanical overload or driver thermal throttling. Inspect coupling tightness, verify driver current limits, check for resonance at operating speeds.',                 'pallet_movein_time',           'stepper'),
 
-- ============================================================
-- SERVO METRICS
-- Patterns: Trend Acceleration, Step Jumps, Random Spikes,
--           Periodic Oscillation, Increasing Outlier Frequency
-- ============================================================
 
-- servo: cavity_1_dispensing_time
(39, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                 'cavity_1_dispensing_time',     'servo'),
(40, 'Step Jumps',                  'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                           'cavity_1_dispensing_time',     'servo'),
(41, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                               'cavity_1_dispensing_time',     'servo'),
(42, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                         'cavity_1_dispensing_time',     'servo'),
(43, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',            'cavity_1_dispensing_time',     'servo'),
 
-- servo: cavity_2_dispensing_time
(44, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                 'cavity_2_dispensing_time',     'servo'),
(45, 'Step Jumps',                  'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                           'cavity_2_dispensing_time',     'servo'),
(46, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                               'cavity_2_dispensing_time',     'servo'),
(47, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                         'cavity_2_dispensing_time',     'servo'),
(48, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',            'cavity_2_dispensing_time',     'servo'),
 
-- servo: cavity_3_dispensing_time
(49, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                 'cavity_3_dispensing_time',     'servo'),
(50, 'Step Jumps',                  'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                           'cavity_3_dispensing_time',     'servo'),
(51, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                               'cavity_3_dispensing_time',     'servo'),
(52, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                         'cavity_3_dispensing_time',     'servo'),
(53, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',            'cavity_3_dispensing_time',     'servo'),
 
-- servo: cavity_4_dispensing_time
(54, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                 'cavity_4_dispensing_time',     'servo'),
(55, 'Step Jumps',                  'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                           'cavity_4_dispensing_time',     'servo'),
(56, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                               'cavity_4_dispensing_time',     'servo'),
(57, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                         'cavity_4_dispensing_time',     'servo'),
(58, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',            'cavity_4_dispensing_time',     'servo'),
 
-- servo: cavity_5_dispensing_time
(59, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                 'cavity_5_dispensing_time',     'servo'),
(60, 'Step Jumps',                  'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                           'cavity_5_dispensing_time',     'servo'),
(61, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                               'cavity_5_dispensing_time',     'servo'),
(62, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                         'cavity_5_dispensing_time',     'servo'),
(63, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',            'cavity_5_dispensing_time',     'servo'),
 
-- servo: cavity_6_dispensing_time
(64, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                 'cavity_6_dispensing_time',     'servo'),
(65, 'Step Jumps',                  'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                           'cavity_6_dispensing_time',     'servo'),
(66, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                               'cavity_6_dispensing_time',     'servo'),
(67, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                         'cavity_6_dispensing_time',     'servo'),
(68, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',            'cavity_6_dispensing_time',     'servo'),
 
-- servo: inspection_time
(69, 'Trend Acceleration',          'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                 'inspection_time',              'servo'),
(70, 'Step Jumps',                  'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                           'inspection_time',              'servo'),
(71, 'Random Spikes',               'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                               'inspection_time',              'servo'),
(72, 'Periodic Oscillation',        'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                         'inspection_time',              'servo'),
(73, 'Increasing Outlier Frequency','Encoder noise or mechanical backlash corrupting position feedback. Inspect encoder disk and cable flex points, review drive fault history, verify PID tuning under load.',            'inspection_time',              'servo'),
 
 
(74, 'Trend Acceleration',          'Friction approaching motor torque limit. Inspect conveyor rollers, pallet underside wear, belt tension, and motor temperature.',                                                       'exit_stopper_raising_time',   'pneumatic_actuator'),
(75, 'Step Jump',                   'Local mechanical obstruction or seized roller creating higher drag. Manually rotate rollers, check debris or pallet interference along travel path.',                                  'exit_stopper_raising_time',   'pneumatic_actuator'),
(76, 'Random Spikes',               'Transient pallet interference, vibration, or electrical noise affecting sensor or drive. Correlate timestamps with nearby machine events and inspect sensors and cabling.',            'exit_stopper_raising_time',   'pneumatic_actuator'),
(77, 'Periodic Oscillation',        'Rotating mechanical defect (eccentric pulley, worn roller, belt irregularity). Inspect rollers and pulleys for periodic drag or misalignment.',                                       'exit_stopper_raising_time',   'pneumatic_actuator'),
(78, 'Increasing Outlier Frequency','Seal degradation or pressure drops causing inconsistent stroke. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                       'exit_stopper_raising_time',   'pneumatic_actuator');