# AZMO Safe Gesture Vocabulary

These are high-level whole-chassis performance requests. They are never servo instructions.

| Gesture | Intended behavior |
|---|---|
| `neutral` | Return to a stable neutral stance |
| `listen` | Freeze locomotion and orient toward the speaker |
| `survey` | Rise slightly and scan as though assessing a battlefield |
| `loom` | Lower, lean forward, and take one slow bounded step |
| `recoil` | Shift backward sharply but safely |
| `stomp` | One controlled front-leg lift and plant |
| `boast` | Raise body and widen stance |
| `enthrone` | Assume a broad, symmetrical imperial posture |
| `contempt` | Turn slightly away and hold |
| `rage` | Lower stance with bounded weight shifts |
| `circle` | Slow lateral orbit around a tracked subject |
| `dismiss` | Rotate away and begin a slow departure |
| `victory` | Rise to configured safe maximum body height |
| `approach` | Move toward a tracked subject at safe speed |
| `retreat` | Move away from a tracked subject |
| `none` | Request no motion |

The future real-time motion controller—not the language model—owns inverse kinematics, foot
placement, collision checks, joint limits, acceleration limits, balance, watchdogs, and emergency
stop behavior.
