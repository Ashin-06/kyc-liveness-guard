# Security policy

## Reporting a vulnerability

Please report suspected vulnerabilities privately, through GitHub's private vulnerability
reporting on this repository (Security → Report a vulnerability) or by contacting the
maintainer, [@Ashin-06](https://github.com/Ashin-06), directly. Do not open a public issue.

Include what you found, how to reproduce it and what an attacker could do with it. You should
get an acknowledgement within 5 working days. Please give us 90 days to fix an issue before
disclosing it, and don't test against anyone else's deployment.

## Supported versions

| Version | Supported |
|---|---|
| 1.0.x | yes |
| < 1.0 | no (prototype releases) |

## Scope

In scope: the liveness service and its HTTP API, the challenge tokens and replay protection,
the voice, pulse, lip-sync and fusion logic, the capture page, the container image and the CI
configuration in this repository.

Out of scope: the Streamlit dashboard (`app.py`, a local analysis tool), your reverse proxy or
TLS setup, device attestation on the customer's phone, and findings that need an
already-compromised host or a leaked API key with the matching role.

A presentation attack that gets through (a deepfake, a replay, a photo) is a detection limit,
not necessarily a vulnerability; the README lists the known ones. Reports of new attacks that
pass are still welcome through the same channel.

This project has **not** had an independent security review or penetration test.
