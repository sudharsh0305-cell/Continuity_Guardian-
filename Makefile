.PHONY: build run scan report clean help

help:
	@echo "Continuity Guardian CLI & Container Tooling Commands:"
	@echo "  make build   - Build the multi-stage Docker container image"
	@echo "  make run     - Spawn the Streamlit UI dashboard container on port 8501"
	@echo "  make scan    - Run batch video tracking headlessly inside the container"
	@echo "  make report  - Generate executive summaries from container log files"
	@echo "  make clean   - Remove locally cached log metrics and anomaly clips"

build:
	docker compose build

run:
	docker compose up

scan:
	docker compose run --rm guardian scan

report:
	docker compose run --rm guardian report

clean:
	rm -rf logs/*.json
	rm -rf anomaly_clips/*.mp4
	rm -rf anomaly_snapshots/*.png
	rm -f master_continuity_log.json reid_registry.json continuity_summary_report.md
