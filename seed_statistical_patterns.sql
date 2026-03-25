-- seed_statistical_patterns.sql

INSERT INTO statistical_patterns (id, statistical_pattern, likely_causes_and_what_to_inspect, metric_name, type_of_motor)
VALUES

-- ============================================================
-- ACTUATOR METRICS
-- Patterns: Trend Acceleration, Step Jump, Recovery Cycles,
--           Random Spikes, Baseline Shift, Variance Growth, Slow Drift
-- ============================================================

-- actuator: entry_stopper_lowering_time
(1,  'Trend Acceleration', 'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                        'entry_stopper_lowering_time',      'pneumatic_actuator'),
(2,  'Step Jump',          'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                         'entry_stopper_lowering_time',      'pneumatic_actuator'),
(3,  'Recovery Cycles',    'Thermal cycling reducing grease film. Check duty cycle vs rating, measure barrel temperature, verify grease grade.',                                                                        'entry_stopper_lowering_time',      'pneumatic_actuator'),
(4,  'Random Spikes',      'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                                 'entry_stopper_lowering_time',      'pneumatic_actuator'),
(5,  'Baseline Shift',     'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                            'entry_stopper_lowering_time',      'pneumatic_actuator'),
(6,  'Variance Growth',    'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                            'entry_stopper_lowering_time',      'pneumatic_actuator'),
(7,  'Slow Drift',         'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                     'entry_stopper_lowering_time',      'pneumatic_actuator'),

-- actuator: entry_stopper_raising_time
(8,  'Trend Acceleration', 'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                        'entry_stopper_raising_time',       'pneumatic_actuator'),
(9,  'Step Jump',          'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                         'entry_stopper_raising_time',       'pneumatic_actuator'),
(10, 'Recovery Cycles',    'Thermal cycling reducing grease film. Check duty cycle vs rating, measure barrel temperature, verify grease grade.',                                                                        'entry_stopper_raising_time',       'pneumatic_actuator'),
(11, 'Random Spikes',      'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                                 'entry_stopper_raising_time',       'pneumatic_actuator'),
(12, 'Baseline Shift',     'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                            'entry_stopper_raising_time',       'pneumatic_actuator'),
(13, 'Variance Growth',    'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                            'entry_stopper_raising_time',       'pneumatic_actuator'),
(14, 'Slow Drift',         'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                     'entry_stopper_raising_time',       'pneumatic_actuator'),

-- actuator: pallet_clamping_time
(15, 'Trend Acceleration', 'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                        'pallet_clamping_time',             'pneumatic_actuator'),
(16, 'Step Jump',          'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                         'pallet_clamping_time',             'pneumatic_actuator'),
(17, 'Recovery Cycles',    'Thermal cycling reducing grease film. Check duty cycle vs rating, measure barrel temperature, verify grease grade.',                                                                        'pallet_clamping_time',             'pneumatic_actuator'),
(18, 'Random Spikes',      'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                                 'pallet_clamping_time',             'pneumatic_actuator'),
(19, 'Baseline Shift',     'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                            'pallet_clamping_time',             'pneumatic_actuator'),
(20, 'Variance Growth',    'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                            'pallet_clamping_time',             'pneumatic_actuator'),
(21, 'Slow Drift',         'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                     'pallet_clamping_time',             'pneumatic_actuator'),

-- actuator: pallet_lifting_time
(22, 'Trend Acceleration', 'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                        'pallet_lifting_time',              'pneumatic_actuator'),
(23, 'Step Jump',          'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                         'pallet_lifting_time',              'pneumatic_actuator'),
(24, 'Recovery Cycles',    'Thermal cycling reducing grease film. Check duty cycle vs rating, measure barrel temperature, verify grease grade.',                                                                        'pallet_lifting_time',              'pneumatic_actuator'),
(25, 'Random Spikes',      'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                                 'pallet_lifting_time',              'pneumatic_actuator'),
(26, 'Baseline Shift',     'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                            'pallet_lifting_time',              'pneumatic_actuator'),
(27, 'Variance Growth',    'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                            'pallet_lifting_time',              'pneumatic_actuator'),
(28, 'Slow Drift',         'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                     'pallet_lifting_time',              'pneumatic_actuator'),

-- actuator: pallet_unclamping_time
(29, 'Trend Acceleration', 'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                        'pallet_unclamping_time',           'pneumatic_actuator'),
(30, 'Step Jump',          'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                         'pallet_unclamping_time',           'pneumatic_actuator'),
(31, 'Recovery Cycles',    'Thermal cycling reducing grease film. Check duty cycle vs rating, measure barrel temperature, verify grease grade.',                                                                        'pallet_unclamping_time',           'pneumatic_actuator'),
(32, 'Random Spikes',      'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                                 'pallet_unclamping_time',           'pneumatic_actuator'),
(33, 'Baseline Shift',     'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                            'pallet_unclamping_time',           'pneumatic_actuator'),
(34, 'Variance Growth',    'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                            'pallet_unclamping_time',           'pneumatic_actuator'),
(35, 'Slow Drift',         'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                     'pallet_unclamping_time',           'pneumatic_actuator'),

-- actuator: pallet_lowering_time
(36, 'Trend Acceleration', 'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                        'pallet_lowering_time',             'pneumatic_actuator'),
(37, 'Step Jump',          'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                         'pallet_lowering_time',             'pneumatic_actuator'),
(38, 'Recovery Cycles',    'Thermal cycling reducing grease film. Check duty cycle vs rating, measure barrel temperature, verify grease grade.',                                                                        'pallet_lowering_time',             'pneumatic_actuator'),
(39, 'Random Spikes',      'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                                 'pallet_lowering_time',             'pneumatic_actuator'),
(40, 'Baseline Shift',     'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                            'pallet_lowering_time',             'pneumatic_actuator'),
(41, 'Variance Growth',    'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                            'pallet_lowering_time',             'pneumatic_actuator'),
(42, 'Slow Drift',         'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                     'pallet_lowering_time',             'pneumatic_actuator'),

-- actuator: pallet_moveout_time
(43, 'Trend Acceleration', 'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                        'pallet_moveout_time',              'stepper'),
(44, 'Step Jump',          'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                         'pallet_moveout_time',              'stepper'),
(45, 'Recovery Cycles',    'Thermal cycling reducing grease film. Check duty cycle vs rating, measure barrel temperature, verify grease grade.',                                                                        'pallet_moveout_time',              'stepper'),
(46, 'Random Spikes',      'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                                 'pallet_moveout_time',              'stepper'),
(47, 'Baseline Shift',     'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                            'pallet_moveout_time',              'stepper'),
(48, 'Variance Growth',    'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                            'pallet_moveout_time',              'stepper'),
(49, 'Slow Drift',         'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                     'pallet_moveout_time',              'stepper'),

-- ============================================================
-- STEPPER METRICS
-- Patterns: Trend Acceleration, Step Jump, Recovery Cycles,
--           Random Spikes, Periodic Oscillation, Variance Growth, Slow Drift
-- ============================================================

-- stepper: pallet_movein_time
(50, 'Trend Acceleration',   'Friction approaching motor torque limit. Inspect conveyor rollers, pallet underside wear, belt tension, and motor temperature.',                                                          'pallet_movein_time',               'stepper'),
(51, 'Step Jump',            'Local mechanical obstruction or seized roller creating higher drag. Manually rotate rollers, check debris or pallet interference along travel path.',                                     'pallet_movein_time',               'stepper'),
(52, 'Recovery Cycles',      'Thermal effects in motor or drivetrain increasing drag during sustained operation. Check motor case temperature, duty cycle, and lubrication condition.',                                 'pallet_movein_time',               'stepper'),
(53, 'Random Spikes',        'Transient pallet interference, vibration, or electrical noise affecting sensor or drive. Correlate timestamps with nearby machine events and inspect sensors and cabling.',               'pallet_movein_time',               'stepper'),
(54, 'Periodic Oscillation', 'Rotating mechanical defect (eccentric pulley, worn roller, belt irregularity). Inspect rollers and pulleys for periodic drag or misalignment.',                                          'pallet_movein_time',               'stepper'),
(55, 'Variance Growth',      'Inconsistent friction due to contamination, roller wear, or pallet condition variation. Inspect rails/rollers for debris and verify pallet bottom surfaces.',                             'pallet_movein_time',               'stepper'),
(56, 'Slow Drift',           'Progressive wear or lubrication loss increasing rolling resistance. Check roller bearings, belt condition, and lubrication schedule.',                                                    'pallet_movein_time',               'stepper'),

-- stepper: exit_stopper_lowering_time
(57, 'Trend Acceleration',   'Friction approaching motor torque limit. Inspect conveyor rollers, pallet underside wear, belt tension, and motor temperature.',                                                          'exit_stopper_lowering_time',       'pneumatic_actuator'),
(58, 'Step Jump',            'Local mechanical obstruction or seized roller creating higher drag. Manually rotate rollers, check debris or pallet interference along travel path.',                                     'exit_stopper_lowering_time',       'pneumatic_actuator'),
(59, 'Recovery Cycles',      'Thermal effects in motor or drivetrain increasing drag during sustained operation. Check motor case temperature, duty cycle, and lubrication condition.',                                 'exit_stopper_lowering_time',       'pneumatic_actuator'),
(60, 'Random Spikes',        'Transient pallet interference, vibration, or electrical noise affecting sensor or drive. Correlate timestamps with nearby machine events and inspect sensors and cabling.',               'exit_stopper_lowering_time',       'pneumatic_actuator'),
(61, 'Periodic Oscillation', 'Rotating mechanical defect (eccentric pulley, worn roller, belt irregularity). Inspect rollers and pulleys for periodic drag or misalignment.',                                          'exit_stopper_lowering_time',       'pneumatic_actuator'),
(62, 'Variance Growth',      'Inconsistent friction due to contamination, roller wear, or pallet condition variation. Inspect rails/rollers for debris and verify pallet bottom surfaces.',                             'exit_stopper_lowering_time',       'pneumatic_actuator'),
(63, 'Slow Drift',           'Progressive wear or lubrication loss increasing rolling resistance. Check roller bearings, belt condition, and lubrication schedule.',                                                    'exit_stopper_lowering_time',       'pneumatic_actuator'),

-- ============================================================
-- SERVO METRICS
-- Patterns: Trend Acceleration, Step Jumps, Recovery Cycles,
--           Random Spikes, Periodic Oscillation, Slow Drift
-- ============================================================

-- servo: cavity_1_dispensing_time
(64, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                    'cavity_1_dispensing_time',         'servo'),
(65, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                               'cavity_1_dispensing_time',         'servo'),
(66, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                                                        'cavity_1_dispensing_time',         'servo'),
(67, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                                   'cavity_1_dispensing_time',         'servo'),
(68, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                             'cavity_1_dispensing_time',         'servo'),
(69, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                                                                'cavity_1_dispensing_time',         'servo'),

-- servo: cavity_2_dispensing_time
(70, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                    'cavity_2_dispensing_time',         'servo'),
(71, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                               'cavity_2_dispensing_time',         'servo'),
(72, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                                                        'cavity_2_dispensing_time',         'servo'),
(73, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                                   'cavity_2_dispensing_time',         'servo'),
(74, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                             'cavity_2_dispensing_time',         'servo'),
(75, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                                                                'cavity_2_dispensing_time',         'servo'),

-- servo: cavity_3_dispensing_time
(76, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                    'cavity_3_dispensing_time',         'servo'),
(77, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                               'cavity_3_dispensing_time',         'servo'),
(78, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                                                        'cavity_3_dispensing_time',         'servo'),
(79, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                                   'cavity_3_dispensing_time',         'servo'),
(80, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                             'cavity_3_dispensing_time',         'servo'),
(81, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                                                                'cavity_3_dispensing_time',         'servo'),

-- servo: cavity_4_dispensing_time
(82, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                    'cavity_4_dispensing_time',         'servo'),
(83, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                               'cavity_4_dispensing_time',         'servo'),
(84, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                                                        'cavity_4_dispensing_time',         'servo'),
(85, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                                   'cavity_4_dispensing_time',         'servo'),
(86, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                             'cavity_4_dispensing_time',         'servo'),
(87, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                                                                'cavity_4_dispensing_time',         'servo'),

-- servo: cavity_5_dispensing_time
(88, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                    'cavity_5_dispensing_time',         'servo'),
(89, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                               'cavity_5_dispensing_time',         'servo'),
(90, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                                                        'cavity_5_dispensing_time',         'servo'),
(91, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                                   'cavity_5_dispensing_time',         'servo'),
(92, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                             'cavity_5_dispensing_time',         'servo'),
(93, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                                                                'cavity_5_dispensing_time',         'servo'),

-- servo: cavity_6_dispensing_time
(94,  'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                   'cavity_6_dispensing_time',         'servo'),
(95,  'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                              'cavity_6_dispensing_time',         'servo'),
(96,  'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                                                       'cavity_6_dispensing_time',         'servo'),
(97,  'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                                  'cavity_6_dispensing_time',         'servo'),
(98,  'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                            'cavity_6_dispensing_time',         'servo'),
(99,  'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                                                               'cavity_6_dispensing_time',         'servo'),

-- servo: pre_cavity_2_dispensing_delay
(100, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                   'pre_cavity_2_dispensing_delay',    'servo'),
(101, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                              'pre_cavity_2_dispensing_delay',    'servo'),
(102, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                                                       'pre_cavity_2_dispensing_delay',    'servo'),
(103, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                                  'pre_cavity_2_dispensing_delay',    'servo'),
(104, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                            'pre_cavity_2_dispensing_delay',    'servo'),
(105, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                                                               'pre_cavity_2_dispensing_delay',    'servo'),

-- servo: pre_cavity_3_dispensing_delay
(106, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                   'pre_cavity_3_dispensing_delay',    'servo'),
(107, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                              'pre_cavity_3_dispensing_delay',    'servo'),
(108, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                                                       'pre_cavity_3_dispensing_delay',    'servo'),
(109, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                                  'pre_cavity_3_dispensing_delay',    'servo'),
(110, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                            'pre_cavity_3_dispensing_delay',    'servo'),
(111, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                                                               'pre_cavity_3_dispensing_delay',    'servo'),

-- servo: pre_cavity_4_dispensing_delay
(112, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                   'pre_cavity_4_dispensing_delay',    'servo'),
(113, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                              'pre_cavity_4_dispensing_delay',    'servo'),
(114, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                                                       'pre_cavity_4_dispensing_delay',    'servo'),
(115, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                                  'pre_cavity_4_dispensing_delay',    'servo'),
(116, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                            'pre_cavity_4_dispensing_delay',    'servo'),
(117, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                                                               'pre_cavity_4_dispensing_delay',    'servo'),

-- servo: pre_cavity_5_dispensing_delay
(118, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                   'pre_cavity_5_dispensing_delay',    'servo'),
(119, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                              'pre_cavity_5_dispensing_delay',    'servo'),
(120, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                                                       'pre_cavity_5_dispensing_delay',    'servo'),
(121, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                                  'pre_cavity_5_dispensing_delay',    'servo'),
(122, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                            'pre_cavity_5_dispensing_delay',    'servo'),
(123, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                                                               'pre_cavity_5_dispensing_delay',    'servo'),

-- servo: pre_cavity_6_dispensing_delay
(124, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                   'pre_cavity_6_dispensing_delay',    'servo'),
(125, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                              'pre_cavity_6_dispensing_delay',    'servo'),
(126, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                                                       'pre_cavity_6_dispensing_delay',    'servo'),
(127, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                                  'pre_cavity_6_dispensing_delay',    'servo'),
(128, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                            'pre_cavity_6_dispensing_delay',    'servo'),
(129, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                                                               'pre_cavity_6_dispensing_delay',    'servo'),

-- servo: inspection_time
(130, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                   'inspection_time',                  'servo'),
(131, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                              'inspection_time',                  'servo'),
(132, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                                                       'inspection_time',                  'servo'),
(133, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                                  'inspection_time',                  'servo'),
(134, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                            'inspection_time',                  'servo'),
(135, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                                                               'inspection_time',                  'servo');