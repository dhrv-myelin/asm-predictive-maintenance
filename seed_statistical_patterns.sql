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
(15, 'Trend Acceleration', 'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                        'entry_stopper_raising_time',       'pneumatic_actuator'),
(16, 'Step Jump',          'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                         'entry_stopper_raising_time',       'pneumatic_actuator'),
(17, 'Recovery Cycles',    'Thermal cycling reducing grease film. Check duty cycle vs rating, measure barrel temperature, verify grease grade.',                                                                        'entry_stopper_raising_time',       'pneumatic_actuator'),
(18, 'Random Spikes',      'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                                 'entry_stopper_raising_time',       'pneumatic_actuator'),
(19, 'Baseline Shift',     'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                            'entry_stopper_raising_time',       'pneumatic_actuator'),
(20, 'Variance Growth',    'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                            'entry_stopper_raising_time',       'pneumatic_actuator'),
(21, 'Slow Drift',         'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                     'entry_stopper_raising_time',       'pneumatic_actuator'),

-- actuator: pallet_clamping_time
(22, 'Trend Acceleration', 'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                        'pallet_clamping_time',             'pneumatic_actuator'),
(23, 'Step Jump',          'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                         'pallet_clamping_time',             'pneumatic_actuator'),
(24, 'Recovery Cycles',    'Thermal cycling reducing grease film. Check duty cycle vs rating, measure barrel temperature, verify grease grade.',                                                                        'pallet_clamping_time',             'pneumatic_actuator'),
(25, 'Random Spikes',      'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                                 'pallet_clamping_time',             'pneumatic_actuator'),
(26, 'Baseline Shift',     'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                            'pallet_clamping_time',             'pneumatic_actuator'),
(27, 'Variance Growth',    'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                            'pallet_clamping_time',             'pneumatic_actuator'),
(28, 'Slow Drift',         'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                     'pallet_clamping_time',             'pneumatic_actuator'),

-- actuator: pallet_lifting_time
(29, 'Trend Acceleration', 'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                        'pallet_lifting_time',              'pneumatic_actuator'),
(30, 'Step Jump',          'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                         'pallet_lifting_time',              'pneumatic_actuator'),
(31, 'Recovery Cycles',    'Thermal cycling reducing grease film. Check duty cycle vs rating, measure barrel temperature, verify grease grade.',                                                                        'pallet_lifting_time',              'pneumatic_actuator'),
(32, 'Random Spikes',      'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                                 'pallet_lifting_time',              'pneumatic_actuator'),
(33, 'Baseline Shift',     'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                            'pallet_lifting_time',              'pneumatic_actuator'),
(34, 'Variance Growth',    'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                            'pallet_lifting_time',              'pneumatic_actuator'),
(35, 'Slow Drift',         'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                     'pallet_lifting_time',              'pneumatic_actuator'),

-- actuator: pallet_unclamping_time
(36, 'Trend Acceleration', 'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                        'pallet_unclamping_time',           'pneumatic_actuator'),
(37, 'Step Jump',          'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                         'pallet_unclamping_time',           'pneumatic_actuator'),
(38, 'Recovery Cycles',    'Thermal cycling reducing grease film. Check duty cycle vs rating, measure barrel temperature, verify grease grade.',                                                                        'pallet_unclamping_time',           'pneumatic_actuator'),
(39, 'Random Spikes',      'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                                 'pallet_unclamping_time',           'pneumatic_actuator'),
(40, 'Baseline Shift',     'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                            'pallet_unclamping_time',           'pneumatic_actuator'),
(41, 'Variance Growth',    'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                            'pallet_unclamping_time',           'pneumatic_actuator'),
(42, 'Slow Drift',         'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                     'pallet_unclamping_time',           'pneumatic_actuator'),

-- actuator: pallet_lowering_time
(43, 'Trend Acceleration', 'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                        'pallet_lowering_time',             'pneumatic_actuator'),
(44, 'Step Jump',          'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                         'pallet_lowering_time',             'pneumatic_actuator'),
(45, 'Recovery Cycles',    'Thermal cycling reducing grease film. Check duty cycle vs rating, measure barrel temperature, verify grease grade.',                                                                        'pallet_lowering_time',             'pneumatic_actuator'),
(46, 'Random Spikes',      'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                                 'pallet_lowering_time',             'pneumatic_actuator'),
(47, 'Baseline Shift',     'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                            'pallet_lowering_time',             'pneumatic_actuator'),
(48, 'Variance Growth',    'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                            'pallet_lowering_time',             'pneumatic_actuator'),
(49, 'Slow Drift',         'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                     'pallet_lowering_time',             'pneumatic_actuator'),

-- actuator: downstream_waiting_time
(50, 'Trend Acceleration', 'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                        'downstream_waiting_time',          'pneumatic_actuator'),
(51, 'Step Jump',          'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                         'downstream_waiting_time',          'pneumatic_actuator'),
(52, 'Recovery Cycles',    'Thermal cycling reducing grease film. Check duty cycle vs rating, measure barrel temperature, verify grease grade.',                                                                        'downstream_waiting_time',          'pneumatic_actuator'),
(53, 'Random Spikes',      'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                                 'downstream_waiting_time',          'pneumatic_actuator'),
(54, 'Baseline Shift',     'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                            'downstream_waiting_time',          'pneumatic_actuator'),
(55, 'Variance Growth',    'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                            'downstream_waiting_time',          'pneumatic_actuator'),
(56, 'Slow Drift',         'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                     'downstream_waiting_time',          'pneumatic_actuator'),

-- actuator: pallet_moveout_time
(57, 'Trend Acceleration', 'Friction consuming pressure margin. Inspect bore and seals for scoring/hardening, verify supply pressure, check lubrication unit.',                                                        'pallet_moveout_time',              'stepper'),
(58, 'Step Jump',          'Stiction onset or flow restriction. Check exhaust port contamination, piston rod seal lip rollover, needle valve partial closure.',                                                         'pallet_moveout_time',              'stepper'),
(59, 'Recovery Cycles',    'Thermal cycling reducing grease film. Check duty cycle vs rating, measure barrel temperature, verify grease grade.',                                                                        'pallet_moveout_time',              'stepper'),
(60, 'Random Spikes',      'Solenoid valve contamination or sticking spool. Inspect air quality (particles/moisture) and valve seals.',                                                                                 'pallet_moveout_time',              'stepper'),
(61, 'Baseline Shift',     'Supply pressure change or new restriction. Check air lines, supply pressure at actuator port, upstream valves.',                                                                            'pallet_moveout_time',              'stepper'),
(62, 'Variance Growth',    'Pressure fluctuations, load variation, or mounting misalignment. Log supply pressure, inspect mounts, verify load consistency.',                                                            'pallet_moveout_time',              'stepper'),
(63, 'Slow Drift',         'Seal wear and lubrication thinning. Review seal life rating and maintenance interval.',                                                                                                     'pallet_moveout_time',              'stepper'),

-- ============================================================
-- STEPPER METRICS
-- Patterns: Trend Acceleration, Step Jump, Recovery Cycles,
--           Random Spikes, Periodic Oscillation, Variance Growth, Slow Drift
-- ============================================================

-- stepper: pallet_movein_time
(64, 'Trend Acceleration',   'Friction approaching motor torque limit. Inspect conveyor rollers, pallet underside wear, belt tension, and motor temperature.',                                                          'pallet_movein_time',               'stepper'),
(65, 'Step Jump',            'Local mechanical obstruction or seized roller creating higher drag. Manually rotate rollers, check debris or pallet interference along travel path.',                                     'pallet_movein_time',               'stepper'),
(66, 'Recovery Cycles',      'Thermal effects in motor or drivetrain increasing drag during sustained operation. Check motor case temperature, duty cycle, and lubrication condition.',                                 'pallet_movein_time',               'stepper'),
(67, 'Random Spikes',        'Transient pallet interference, vibration, or electrical noise affecting sensor or drive. Correlate timestamps with nearby machine events and inspect sensors and cabling.',               'pallet_movein_time',               'stepper'),
(68, 'Periodic Oscillation', 'Rotating mechanical defect (eccentric pulley, worn roller, belt irregularity). Inspect rollers and pulleys for periodic drag or misalignment.',                                          'pallet_movein_time',               'stepper'),
(69, 'Variance Growth',      'Inconsistent friction due to contamination, roller wear, or pallet condition variation. Inspect rails/rollers for debris and verify pallet bottom surfaces.',                             'pallet_movein_time',               'stepper'),
(70, 'Slow Drift',           'Progressive wear or lubrication loss increasing rolling resistance. Check roller bearings, belt condition, and lubrication schedule.',                                                    'pallet_movein_time',               'stepper'),

-- stepper: exit_stopper_lowering_time
(71, 'Trend Acceleration',   'Friction approaching motor torque limit. Inspect conveyor rollers, pallet underside wear, belt tension, and motor temperature.',                                                          'exit_stopper_lowering_time',       'stepper'),
(72, 'Step Jump',            'Local mechanical obstruction or seized roller creating higher drag. Manually rotate rollers, check debris or pallet interference along travel path.',                                     'exit_stopper_lowering_time',       'stepper'),
(73, 'Recovery Cycles',      'Thermal effects in motor or drivetrain increasing drag during sustained operation. Check motor case temperature, duty cycle, and lubrication condition.',                                 'exit_stopper_lowering_time',       'stepper'),
(74, 'Random Spikes',        'Transient pallet interference, vibration, or electrical noise affecting sensor or drive. Correlate timestamps with nearby machine events and inspect sensors and cabling.',               'exit_stopper_lowering_time',       'stepper'),
(75, 'Periodic Oscillation', 'Rotating mechanical defect (eccentric pulley, worn roller, belt irregularity). Inspect rollers and pulleys for periodic drag or misalignment.',                                          'exit_stopper_lowering_time',       'stepper'),
(76, 'Variance Growth',      'Inconsistent friction due to contamination, roller wear, or pallet condition variation. Inspect rails/rollers for debris and verify pallet bottom surfaces.',                             'exit_stopper_lowering_time',       'stepper'),
(77, 'Slow Drift',           'Progressive wear or lubrication loss increasing rolling resistance. Check roller bearings, belt condition, and lubrication schedule.',                                                    'exit_stopper_lowering_time',       'stepper'),

-- ============================================================
-- SERVO METRICS
-- Patterns: Trend Acceleration, Step Jumps, Recovery Cycles,
--           Random Spikes, Periodic Oscillation, Slow Drift
-- ============================================================

-- servo: dispensing_time
(78, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                    'dispensing_time',                  'servo'),
(79, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                               'dispensing_time',                  'servo'),
(80, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                                                        'dispensing_time',                  'servo'),
(81, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                                   'dispensing_time',                  'servo'),
(82, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                             'dispensing_time',                  'servo'),
(83, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                                                                'dispensing_time',                  'servo'),

-- servo: inspection_time
(84, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                                                                    'inspection_time',                  'servo'),
(85, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',                                               'inspection_time',                  'servo'),
(86, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                                                        'inspection_time',                  'servo'),
(87, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                                                                   'inspection_time',                  'servo'),
(88, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                                                             'inspection_time',                  'servo'),
(89, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                                                                'inspection_time',                  'servo');