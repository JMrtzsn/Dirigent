// Package patterns provides composable agent interaction patterns.
package patterns

import (
	"context"
	"sync"

	"github.com/johannesmauritzson/dirigent/core"
	"github.com/johannesmauritzson/dirigent/runtime"
)

// Sender identifiers for tracing which pattern initiated a delegation.
const (
	SenderDelegate   = "pattern:delegate"
	SenderFanout     = "pattern:fanout"
	SenderReviewLoop = "pattern:review_loop"
)

// Delegate sends a task to a single agent. Thin wrapper over runtime.Delegate.
func Delegate(ctx context.Context, rt *runtime.Runtime, opts runtime.DelegateOpts) (core.DelegateResponse, error) {
	if opts.Sender == "" {
		opts.Sender = SenderDelegate
	}
	return rt.Delegate(ctx, opts)
}

// Fanout sends the same task to multiple agents concurrently and collects all results.
func Fanout(ctx context.Context, rt *runtime.Runtime, agents []string, task string, taskCtx map[string]any) ([]core.DelegateResponse, error) {
	if len(agents) == 0 {
		return nil, &core.RegistryError{Message: "must provide at least one agent"}
	}

	results := make([]core.DelegateResponse, len(agents))
	errs := make([]error, len(agents))
	var wg sync.WaitGroup

	for i, name := range agents {
		wg.Add(1)
		go func(idx int, agentName string) {
			defer wg.Done()
			resp, err := rt.Delegate(ctx, runtime.DelegateOpts{
				To:      agentName,
				Task:    task,
				Context: taskCtx,
				Sender:  SenderFanout,
			})
			results[idx] = resp
			errs[idx] = err
		}(i, name)
	}
	wg.Wait()

	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	return results, nil
}

// ReviewResult holds the outcome of a review loop.
type ReviewResult struct {
	Accepted    bool
	FinalResult any
	Rounds      int
	History     []HistoryEntry
}

// HistoryEntry records a single produce or review step.
type HistoryEntry struct {
	Round   int
	Phase   string // "produce" or "review"
	Success bool
	Result  any
}

// ReviewOpts configures a review loop.
type ReviewOpts struct {
	Producer  string
	Reviewer  string
	Task      string
	Context   map[string]any
	MaxRounds int
}

// ReviewLoop runs a produce → review → revise cycle up to MaxRounds.
func ReviewLoop(ctx context.Context, rt *runtime.Runtime, opts ReviewOpts) (ReviewResult, error) {
	history := make([]HistoryEntry, 0)
	currentCtx := make(map[string]any)
	for k, v := range opts.Context {
		currentCtx[k] = v
	}

	var lastResult any

	for round := 1; round <= opts.MaxRounds; round++ {
		// Producer generates
		production, err := rt.Delegate(ctx, runtime.DelegateOpts{
			To:      opts.Producer,
			Task:    opts.Task,
			Context: currentCtx,
			Sender:  SenderReviewLoop,
		})
		if err != nil {
			return ReviewResult{}, err
		}

		history = append(history, HistoryEntry{
			Round:   round,
			Phase:   "produce",
			Success: production.Success,
			Result:  production.Result,
		})

		if !production.Success {
			return ReviewResult{
				Accepted:    false,
				FinalResult: production.Result,
				Rounds:      round,
				History:     history,
			}, nil
		}

		lastResult = production.Result

		// Reviewer evaluates
		reviewCtx := map[string]any{
			"submission":    production.Result,
			"original_task": opts.Task,
			"round":         round,
		}
		for k, v := range currentCtx {
			reviewCtx[k] = v
		}

		review, err := rt.Delegate(ctx, runtime.DelegateOpts{
			To:      opts.Reviewer,
			Task:    "Review this submission for: " + opts.Task,
			Context: reviewCtx,
			Sender:  SenderReviewLoop,
		})
		if err != nil {
			return ReviewResult{}, err
		}

		history = append(history, HistoryEntry{
			Round:   round,
			Phase:   "review",
			Success: review.Success,
			Result:  review.Result,
		})

		// Check acceptance
		if resultMap, ok := review.Result.(map[string]any); ok {
			if accepted, ok := resultMap["accepted"].(bool); ok && accepted {
				return ReviewResult{
					Accepted:    true,
					FinalResult: lastResult,
					Rounds:      round,
					History:     history,
				}, nil
			}
			if feedback, ok := resultMap["feedback"]; ok {
				currentCtx["feedback"] = feedback
			}
		}

		currentCtx["previous_attempt"] = production.Result
	}

	return ReviewResult{
		Accepted:    false,
		FinalResult: lastResult,
		Rounds:      opts.MaxRounds,
		History:     history,
	}, nil
}
