import pytest

from scripts.cross_validate_gait_features import subject_folds


def test_subject_folds_cover_each_identity_once_without_overlap():
    groups = subject_folds(list(range(17)))
    assert sorted(subject for group in groups for subject in group) == list(range(17))
    assert [group.tolist() for group in groups] == [
        group.tolist() for group in subject_folds(list(reversed(range(17))))
    ]
    assert all(len(group) >= 2 for group in groups)


def test_subject_folds_reject_cohort_too_small_for_verification():
    with pytest.raises(ValueError, match="six subjects"):
        subject_folds([1, 2, 3, 4, 5])
