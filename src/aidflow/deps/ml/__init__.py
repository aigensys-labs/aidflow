"""
The Dependency Abstraction Layer (Dependency Firewall).

This module serves as the sole authorized gateway for all external dependencies 
and Python standard library imports across the entire AIDflow ecosystem. 
Following a strict architectural mandate, no module within the AIDflow package 
is permitted to import third-party libraries or native Python modules directly; 
instead, they must consume these dependencies via the `.deps` abstraction layer.

Key Architectural Objectives:
1. Total Dependency Control: Centralizes all external imports into a single, 
   auditable namespace, eliminating 'Import Spaghetti' and preventing hidden 
   dependencies from leaking into the core logic.
   
2. API Stability & Refactorability: Acts as a buffer against breaking changes in 
   external libraries. By wrapping dependencies (e.g., `itables` or `pandas`), 
   AIDflow can evolve its internal implementation or switch providers without 
   requiring a single change to the higher-level orchestrators.

3. Global Configuration of Externals: Allows for the unified configuration of 
   external library behaviors—such as setting global options for `itables` or 
   standardizing OS-level path handling—at a single entry point.

4. Enhanced Portability: Simplifies environment migration and dependency 
   auditing, as the entire external footprint of the project is explicitly 
   declared within this directory.
"""