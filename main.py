from typing import Callable, Dict
from scripts.script import Script
import scripts.main_experiment as main_experiment
# import scripts.the_effect_of_noisy_labels as the_effect_of_noisy_labels
from sys import argv


SCRIPTS: Dict[str, Callable[[], Script]] = {
    "main_experiment": main_experiment.get_script,
    # "the_effect_of_noisy_labels": the_effect_of_noisy_labels.get_script,
}

def main():
    if len(argv) != 3:
        print("Usage: 'main.py {} [func-name]'".format("{" + " | ".join(list(SCRIPTS)) + "}"))
        return

    else:
        script_name = argv[1]
        func_name = argv[2]
        script = SCRIPTS[script_name]()
        script.run(func_name)

if __name__ == "__main__":
    main()