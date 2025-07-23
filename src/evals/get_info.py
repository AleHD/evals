from evals.tasks import get_all_tasks


if __name__ == "__main__":
    tasks = get_all_tasks()
    print("Estimated sizes:")
    print("\n".join(f"{task.name}: {task.size}" for task in sorted(tasks, key=lambda task: task.name)))
    total_size = sum(task.size for task in tasks)
    print("Total size:", total_size)
    print()

    dimensions = sorted({task.dimension for task in tasks})
    for dim in dimensions:
        print("Dimension:", dim, "tasks:", [task.name for task in tasks if task.dimension == dim])
