# Warehouse-HMARL-Demo
HMARL experimentations with self learning agents, low and high level decisionmakers

<img width="579" height="194" alt="Screenshot 2025-10-21 at 12 49 45 AM" src="https://github.com/user-attachments/assets/ef22cdb5-9b1a-4fdb-8e84-7d19859c0cf3" />

Quick Start:
#### 1) Train worker (5–10 minutes on CPU)
python train_worker.py

#### 2) Train manager (5–15 minutes on CPU)
python train_manager.py

#### 3) Visual demo
python run_demo.py

#### Notes

The worker learns how to reach a goal given [worker_xy, goal_xy].
The manager learns what assignment (worker_id -> task_id) to issue.
Manager acts every ASSIGN_INTERVAL steps; workers act every step.
