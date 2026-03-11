import os
import warnings

from openmcadapter import Settings


class SlurmSettings(Settings):
    @property
    def threads(self):
        threads = int(os.environ["SLURM_CPUS_ON_NODE"])
        if int(nodes := os.environ["SLURM_NNODES"]) > 1:
            warnings.warn(
                f"The process is running on {nodes} nodes which is more then 1,"
                f" this is not ideal for openMP parallelization"
            )
        return threads
