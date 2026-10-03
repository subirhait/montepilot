# montepilotR

Development R interface to the MontePilot Python engine. Version 0.0.5.9001
exposes backend, floating-point precision, estimate/coverage MCSE stopping, and
checkpoint controls for the built-in demonstration.

This package is intentionally marked as a prototype. Before CRAN submission it
needs a real maintainer email, generated documentation, tests on Windows,
macOS, and Linux, graceful CRAN test skips, and a stable PyPI release of the
Python core.

For local development:

```r
remotes::install_local("r/montepilotR")
library(montepilotR)
montepilot_install(".")
montepilot_doctor()
result <- montepilot_demo(backend = "auto")
```
