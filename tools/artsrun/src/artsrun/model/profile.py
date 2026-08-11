"""Machine/node settings: how many nodes, how wide each one runs, how the
ranks are launched."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class Launcher(StrEnum):
    LOCAL = "local"
    SSH = "ssh"
    SLURM = "slurm"


class SshSettings(BaseModel):
    """The nodes this campaign may use, and their names.

    `budget` states how many nodes are available; `hosts` must name exactly
    that many, in rank order.  Stating the count separately is what catches a
    roster that quietly lost a line.
    """

    budget: int = Field(ge=1)
    hosts: list[str] = Field(default_factory=list)


class SlurmSettings(BaseModel):
    """Submission parameters.

    Each cell is its own exclusive job, submitted up front — scheduling the
    queue is Slurm's whole purpose, so no budget of ours meters it, and the
    campaign survives the submitting login node because every job records
    its own outcome.
    """

    partition: str | None = None
    # Build work may queue elsewhere than the cells: a debug partition takes
    # a small compile job sooner than the batch partition takes a node.
    build_partition: str | None = None
    # A compile is not a measurement — it asks for a handful of cpus it can
    # get anywhere in the queue, never an exclusive node.
    build_cpus: int = Field(default=8, ge=1)
    account: str | None = None
    qos: str | None = None
    extra_sbatch: list[str] = Field(default_factory=list)
    poll_interval_s: float = Field(default=10.0, gt=0)


class Profile(BaseModel):
    name: str
    launcher: Launcher
    nodes: list[int] = Field(min_length=1)
    workers: int = Field(ge=1)
    progress: int = Field(ge=0)

    pin: bool = True
    provider: str | None = None
    route_table_size: int = 16
    core_dump: bool = False
    net_interface: str | None = None
    fabric_domain: str | None = None
    regpool_slab_mb: int | None = None

    # Parallel connections each node opens.  A local run has the runtime find
    # its own block of node_count x port_count; a remote one must be told which
    # ports to use, and then the list has to name exactly this many.
    port_count: int = Field(default=1, ge=1)
    ports: list[int] = Field(default_factory=list)

    ssh: SshSettings | None = None
    slurm: SlurmSettings | None = None

    cell_timeout_s: int = Field(default=300, ge=1)
    repeats: int = Field(default=1, ge=1)

    @property
    def threads_per_node(self) -> int:
        """The core block one rank occupies.

        Whether the machine actually has that many is left to the run: ARTS
        does not oversubscribe, and a geometry that asks for more threads than
        there are cores fails at startup on its own.
        """
        return self.workers + self.progress

    @property
    def max_nodes(self) -> int:
        return max(self.nodes)

    @property
    def hosts(self) -> list[str]:
        return self.ssh.hosts if self.ssh else []

    @model_validator(mode="after")
    def _check(self) -> "Profile":
        if self.launcher is Launcher.LOCAL:
            if self.ports:
                raise ValueError(
                    "ports must not be set for launcher=local: the ranks share a "
                    "machine, so the spawning rank finds a free block itself"
                )
        else:
            if not self.ports:
                raise ValueError(f"ports is required for launcher={self.launcher}")
            if len(self.ports) != self.port_count:
                raise ValueError(
                    f"ports names {len(self.ports)} port(s) but port_count is "
                    f"{self.port_count}; the list must name exactly that many"
                )
        if self.launcher is Launcher.SSH:
            if self.ssh is None:
                raise ValueError("launcher=ssh requires an ssh section")
            if len(self.ssh.hosts) != self.ssh.budget:
                raise ValueError(
                    f"ssh.hosts names {len(self.ssh.hosts)} host(s) but "
                    f"ssh.budget is {self.ssh.budget}; name one host per node"
                )
            if self.ssh.budget < self.max_nodes:
                raise ValueError(
                    f"ssh.budget={self.ssh.budget} is below the widest node "
                    f"count ({self.max_nodes}); that cell has nowhere to run"
                )
        if self.launcher is Launcher.SLURM and self.slurm is None:
            raise ValueError("launcher=slurm requires a slurm section")
        if self.provider == "verbs":
            # The runtime requires RDM endpoints; the verbs core provider
            # offers only connection-oriented MSG endpoints, so RDM exists
            # solely as the layered stack — and fi_getinfo treats a lone core
            # name as excluding utility layering, so bare "verbs" can never
            # match.  The layered name is the unit, stated explicitly so its
            # presence is a user decision rather than a silent rewrite.
            raise ValueError(
                "provider=verbs can never match: name the layered stack "
                "explicitly — provider=verbs;ofi_rxm"
            )
        if any(n < 1 for n in self.nodes):
            raise ValueError("node counts must be >= 1")
        return self
