from piper_xvla.train_xvla_piper import split_episode_ids


def test_split_episode_ids_is_deterministic_disjoint_and_reserves_validation_tail():
    train, val = split_episode_ids(list(range(101)), val_count=11)

    assert len(train) == 90
    assert val == list(range(90, 101))
    assert set(train).isdisjoint(val)
