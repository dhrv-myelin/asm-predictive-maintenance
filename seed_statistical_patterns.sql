-- seed_statistical_patterns.sql

INSERT INTO statistical_patterns (id, statistical_pattern, likely_causes_and_what_to_inspect, metric_name, type_of_motor)
VALUES
-- actuator: entry_stopper_lowering_time
(1,  'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                               'entry_stopper_lowering_time',         'actuator'),
(2,  'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',          'entry_stopper_lowering_time',         'actuator'),
(3,  'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                    'entry_stopper_lowering_time',         'actuator'),
(4,  'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                              'entry_stopper_lowering_time',         'actuator'),
(5,  'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                        'entry_stopper_lowering_time',         'actuator'),
(6,  'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                           'entry_stopper_lowering_time',         'actuator'),

-- actuator: movein_to_entry_stopper_up_delay
(7,  'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                               'movein_to_entry_stopper_up_delay',    'actuator'),
(8,  'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',          'movein_to_entry_stopper_up_delay',    'actuator'),
(9,  'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                    'movein_to_entry_stopper_up_delay',    'actuator'),
(10, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                              'movein_to_entry_stopper_up_delay',    'actuator'),
(11, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                        'movein_to_entry_stopper_up_delay',    'actuator'),
(12, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                           'movein_to_entry_stopper_up_delay',    'actuator'),

-- actuator: entry_stopper_raising_time
(13, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                               'entry_stopper_raising_time',          'actuator'),
(14, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',          'entry_stopper_raising_time',          'actuator'),
(15, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                    'entry_stopper_raising_time',          'actuator'),
(16, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                              'entry_stopper_raising_time',          'actuator'),
(17, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                        'entry_stopper_raising_time',          'actuator'),
(18, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                           'entry_stopper_raising_time',          'actuator'),

-- actuator: pallet_clamping_time
(19, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                               'pallet_clamping_time',                'actuator'),
(20, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',          'pallet_clamping_time',                'actuator'),
(21, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                    'pallet_clamping_time',                'actuator'),
(22, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                              'pallet_clamping_time',                'actuator'),
(23, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                        'pallet_clamping_time',                'actuator'),
(24, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                           'pallet_clamping_time',                'actuator'),

-- actuator: pallet_lifting_time
(25, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                               'pallet_lifting_time',                 'actuator'),
(26, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',          'pallet_lifting_time',                 'actuator'),
(27, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                    'pallet_lifting_time',                 'actuator'),
(28, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                              'pallet_lifting_time',                 'actuator'),
(29, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                        'pallet_lifting_time',                 'actuator'),
(30, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                           'pallet_lifting_time',                 'actuator'),

-- actuator: pallet_unclamping_time
(31, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                               'pallet_unclamping_time',              'actuator'),
(32, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',          'pallet_unclamping_time',              'actuator'),
(33, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                    'pallet_unclamping_time',              'actuator'),
(34, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                              'pallet_unclamping_time',              'actuator'),
(35, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                        'pallet_unclamping_time',              'actuator'),
(36, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                           'pallet_unclamping_time',              'actuator'),

-- actuator: pallet_lowering_time
(37, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                               'pallet_lowering_time',                'actuator'),
(38, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',          'pallet_lowering_time',                'actuator'),
(39, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                    'pallet_lowering_time',                'actuator'),
(40, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                              'pallet_lowering_time',                'actuator'),
(41, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                        'pallet_lowering_time',                'actuator'),
(42, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                           'pallet_lowering_time',                'actuator'),

-- actuator: downstream_waiting_time
(43, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                               'downstream_waiting_time',             'actuator'),
(44, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',          'downstream_waiting_time',             'actuator'),
(45, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                    'downstream_waiting_time',             'actuator'),
(46, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                              'downstream_waiting_time',             'actuator'),
(47, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                        'downstream_waiting_time',             'actuator'),
(48, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                           'downstream_waiting_time',             'actuator'),

-- actuator: pallet_moveout_time
(49, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                               'pallet_moveout_time',                 'actuator'),
(50, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',          'pallet_moveout_time',                 'actuator'),
(51, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                    'pallet_moveout_time',                 'actuator'),
(52, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                              'pallet_moveout_time',                 'actuator'),
(53, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                        'pallet_moveout_time',                 'actuator'),
(54, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                           'pallet_moveout_time',                 'actuator'),

-- stepper: pallet_movein_time
(55, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                               'pallet_movein_time',                  'stepper'),
(56, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',          'pallet_movein_time',                  'stepper'),
(57, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                    'pallet_movein_time',                  'stepper'),
(58, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                              'pallet_movein_time',                  'stepper'),
(59, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                        'pallet_movein_time',                  'stepper'),
(60, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                           'pallet_movein_time',                  'stepper'),

-- stepper: exit_stopper_lowering_time
(61, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                               'exit_stopper_lowering_time',          'stepper'),
(62, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',          'exit_stopper_lowering_time',          'stepper'),
(63, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                    'exit_stopper_lowering_time',          'stepper'),
(64, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                              'exit_stopper_lowering_time',          'stepper'),
(65, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                        'exit_stopper_lowering_time',          'stepper'),
(66, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                           'exit_stopper_lowering_time',          'stepper'),

-- servo: dispensing_time
(67, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                               'dispensing_time',                     'servo'),
(68, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',          'dispensing_time',                     'servo'),
(69, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                    'dispensing_time',                     'servo'),
(70, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                              'dispensing_time',                     'servo'),
(71, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                        'dispensing_time',                     'servo'),
(72, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                           'dispensing_time',                     'servo'),

-- servo: inspection_time
(73, 'Trend Acceleration',   'Rapid friction growth; servo torque margin collapsing. Check RMS torque trend and motor temperature.',                                               'inspection_time',                     'servo'),
(74, 'Step Jumps',           'Sudden friction change due to ball screw stiction, bearing damage, or gearbox defect. Check following error and position-linked behavior.',          'inspection_time',                     'servo'),
(75, 'Recovery Cycles',      'Thermal expansion or grease viscosity effects temporarily raising friction. Inspect screw temperature and lubrication condition.',                    'inspection_time',                     'servo'),
(76, 'Random Spikes',        'Encoder noise, EMI, or intermittent load disturbance causing corrective motion. Inspect encoder cables and grounding.',                              'inspection_time',                     'servo'),
(77, 'Periodic Oscillation', 'Repeating mechanical resistance due to cyclic defect or resonance. Inspect screw nut, guides, or couplings.',                                        'inspection_time',                     'servo'),
(78, 'Slow Drift',           'Gradual mechanical wear increasing friction. Check backlash growth and RMS torque trend.',                                                           'inspection_time',                     'servo');
