"""Per-task success must use evaluated episode counts, including all failures."""
from eval.eval_libero_v4_trajectory import summarize_task_success


def test_task_success_counts_unequal_groups_and_retains_zero_success_tasks():
    episodes = [
        {"task_name": "stove", "success": True},
        {"task_name": "bowl", "success": False},
        {"task_name": "stove", "success": False},
        {"task_name": "drawer", "success": True},
        {"task_name": "stove", "success": True},
    ]
    result = summarize_task_success(episodes)
    assert list(result) == ["bowl", "drawer", "stove"]
    assert result["bowl"] == dict(success_count=0, n_episodes=1, success_rate=0.)
    assert result["drawer"] == dict(success_count=1, n_episodes=1, success_rate=1.)
    assert result["stove"] == dict(success_count=2, n_episodes=3, success_rate=2/3)
    assert sum(task["success_count"] for task in result.values()) == 3
    assert sum(task["n_episodes"] for task in result.values()) == len(episodes)
    assert summarize_task_success([]) == {}
