"""Tools JARVIS's agents can use — always behind permissions, limits and an audit log.

The agents do not get direct access to anything. They ask for a tool in a fixed
text format; JARVIS checks the policy (allow / confirm / deny), asks you when
needed, runs the tool with limits, logs the call, and gives the agent the result
marked as untrusted data.
"""
