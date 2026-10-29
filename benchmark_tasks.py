MOLECULENET_TASKS = (
    "bace",
    "bbbp",
    "hiv",
    "clintox",
    "tox21",
    "sider",
    "toxcast",
    "muv",
)

PYTDC_TASKS = (
    "ames",
    "herg",
    "hia_hou",
    "pgp_broccatelli",
    "bioavailability_ma",
    "DILI",
)

PAPER_TASKS = MOLECULENET_TASKS + PYTDC_TASKS

def is_smoke_task(task: str) -> bool:

    return task == "smoke_run" or task.endswith("_smoke")
