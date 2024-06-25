# UV-config-from-MCCS-YAML

Script to generate SKA-Low station configuration from official Gitlab repository.

* Retrieves the lasest `ska-low-deployment` git repository (where station YAML files are located)
* Generates aa_uv's internally-used UV Configuration for a station, from YAML files
* Copies these over to `aa_uv/src/aa_uv/config`