// Package patterns provides composable agent interaction patterns.
package patterns

import (
	"context"
	"sync"

	"github.com/johannesmauritzson/dirigent/core"
	"github.com/johannesmauritzson/dirigent/runtime"
)

// Delegate sends a task to a single agent. Thin wrapper over runtime.Delegate.
func Delegate(ctx context.Context, rt *runtime.Runtime, opts runtime.DelegateOpts) (core.DelegateResponse, error) {
	if opts.Sender == "" {
		opts.Sender = "__pattern__"
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
				Sender:  "__fanout__",
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
	History     []map[string]any
}

// ReviewLoop runs a produce → review → revise cycle up to maxRounds.
func ReviewLoop(ctx context.Context, rt *runtime.Runtime, producer, reviewer, task string, taskCtx map[string]any, maxRounds int) (ReviewResult, error) {
	history := make([]map[string]any, 0)
	currentCtx := make(map[string]any)
	for k, v := range taskCtx {
		currentCtx[k] = v
	}

	var lastResult any

	for round := 1; round <= maxRounds; round++ {
		// Producer generates
		production, err := rt.Delegate(ctx, runtime.DelegateOpts{
			To:      producer,
			Task:    task,
			Context: currentCtx,
			Sender:  "__review_loop__",
		})
		if err != nil {
			return ReviewResult{}, err
		}

		history = append(history, map[string]any{
			"round":   round,
			"phase":   "produce",
			"success": production.Success,
			"result":  production.Result,
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
			"original_task": task,
			"round":         round,
		}
		for k, v := range currentCtx {
			reviewCtx[k] = v
		}

		review, err := rt.Delegate(ctx, runtime.DelegateOpts{
			To:      reviewer,
			Task:    "Review this submission for: " + task,
			Context: reviewCtx,
			Sender:  "__review_loop__",
		})
		if err != nil {
			return ReviewResult{}, err
		}

		history = append(history, map[string]any{
			"round":   round,
			"phase":   "review",
			"success": review.Success,
			"result":  review.Result,
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
		Rounds:      maxRounds,
		History:     history,
	}, nil
}
