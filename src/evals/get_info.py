from evals.tasks import get_all_tasks


def get_time_str(minutes: float) -> str:
    return f"{int(minutes/60)}h{int(minutes) % 60}m{int(minutes*60) % 60}s"


if __name__ == "__main__":
    total_time = 9*60  # Total time for a 70B to run all tasks.

    tasks = get_all_tasks()
    total_size = sum(task.size for task in tasks)
    est = 0
    print("Estimated sizes:")
    for task in sorted(tasks, key=lambda task: task.size, reverse=True):
        estimated_time = total_time*task.size/total_size
        est += estimated_time
        print(f"{task.name}: {task.size}rows ({get_time_str(estimated_time)})")
    print("Total size:", total_size, "Total time:", get_time_str(total_time))
    print(get_time_str(est))
    print()

    dimensions = sorted({task.dimension for task in tasks})
    for dim in dimensions:
        print("Dimension:", dim, "tasks:", [task.name for task in tasks if task.dimension == dim])
