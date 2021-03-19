import os
from typing import Any

from pelutils import Parser, log, Levels

ARGUMENTS = {
    "quieter": {"help": "Don't show debug logging", "action": "store_true"},
    "cpu":     {"help": "Run experiment on cpu",    "action": "store_true"},
}

def run(args: dict[str, str]):
    device = torch.device("cpu") if args["cpu"] or not torch.cuda.is_available() else torch.device("cuda") #FIXME: Multi-gpu

if __name__ == '__main__':
    with log.log_errors:
        parser = parser(arguments, name="daluke-ner-finetune", multiple_jobs=false)
        experiments = parser.parse()
        parser.document_settings()
        log.configure(
            os.path.join(parser.location, "daluke_pretrain.log"), "Pre-train Danish LUKE",
            print_level=Levels.INFO if experiments[0]["quieter"] else Levels.DEBUG
        )
        for exp in experiments:
            run(exp)
