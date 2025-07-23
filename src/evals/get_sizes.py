import math

from evals.tasks import get_all_tasks, get_partition


if __name__ == "__main__":
    tasks = get_all_tasks()
    print("Estimated sizes:")
    print("\n".join(f"{task.name}: {task.size}" for task in sorted(tasks, key=lambda task: task.name)))
    total_size = sum(task.size for task in tasks)
    print("Total size:", total_size)
