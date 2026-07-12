"""Capability adapters (wave 2+).

The seam that translates the one capability contract into concrete tool
surfaces: the deepagents LangChain ``StructuredTool`` adapter (M5) and
the MCP tool adapter (M8). Both iterate
:meth:`~inqtrix.capabilities.registry.CapabilityRegistry.manifest` /
``definitions`` — no adapter re-declares a tool list. Empty until those
milestones land; kept as a package so the import path is stable.
"""
