.PHONY: build test lint clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

build: ## Build the dirigent binary
	cd go && go build -o ../bin/dirigent ./cmd/dirigent

test: ## Run tests
	cd go && go test ./...

lint: ## Run go vet
	cd go && go vet ./...

check: lint test ## Run lint + tests

clean: ## Remove build artifacts
	rm -rf bin/ .dirigent-session.json .dirigent-worktrees/
