package patterns_test

import (
	"context"
	"testing"

	"github.com/johannesmauritzson/dirigent/core"
	"github.com/johannesmauritzson/dirigent/patterns"
	"github.com/johannesmauritzson/dirigent/runtime"
)

// fakeAgent is a test helper implementing core.Agent.
type fakeAgent struct {
	name      string
	caps      []string
	handleFn  func(ctx context.Context, req core.DelegateRequest) (core.DelegateResponse, error)
	callCount int
}

func (f *fakeAgent) Info() core.AgentInfo {
	caps := make([]core.Capability, len(f.caps))
	for i, c := range f.caps {
		caps[i] = core.Capability{Name: c}
	}
	return core.AgentInfo{Name: f.name, Capabilities: caps}
}

func (f *fakeAgent) Handle(ctx context.Context, req core.DelegateRequest) (core.DelegateResponse, error) {
	f.callCount++
	if f.handleFn != nil {
		return f.handleFn(ctx, req)
	}
	return core.NewDelegateResponse(f.name, req.Sender, req.ID, true, "default_result", ""), nil
}

func newTestRuntime(agents ...core.Agent) *runtime.Runtime {
	rt := runtime.New(runtime.DefaultConfig())
	for _, ag := range agents {
		_ = rt.Register(ag)
	}
	return rt
}

func TestDelegateByCapability(t *testing.T) {
	planner := &fakeAgent{
		name: "fakeplanner",
		caps: []string{"plan"},
		handleFn: func(_ context.Context, req core.DelegateRequest) (core.DelegateResponse, error) {
			return core.NewDelegateResponse("fakeplanner", req.Sender, req.ID, true, map[string]any{"steps": []string{"step1", "step2"}}, ""), nil
		},
	}
	rt := newTestRuntime(planner)

	resp, err := patterns.Delegate(context.Background(), rt, runtime.DelegateOpts{
		Capability: "plan",
		Task:       "build auth",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !resp.Success {
		t.Fatal("expected success")
	}
	result, ok := resp.Result.(map[string]any)
	if !ok {
		t.Fatalf("expected map result, got %T", resp.Result)
	}
	steps, ok := result["steps"].([]string)
	if !ok {
		t.Fatalf("expected []string steps, got %T", result["steps"])
	}
	if len(steps) != 2 {
		t.Fatalf("expected 2 steps, got %d", len(steps))
	}
}

func TestDelegateByName(t *testing.T) {
	dev := &fakeAgent{
		name: "fakedeveloper",
		caps: []string{"implement"},
		handleFn: func(_ context.Context, req core.DelegateRequest) (core.DelegateResponse, error) {
			return core.NewDelegateResponse("fakedeveloper", req.Sender, req.ID, true, "code_output", ""), nil
		},
	}
	rt := newTestRuntime(dev)

	resp, err := patterns.Delegate(context.Background(), rt, runtime.DelegateOpts{
		To:   "fakedeveloper",
		Task: "write code",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !resp.Success {
		t.Fatal("expected success")
	}
	if resp.Result != "code_output" {
		t.Fatalf("expected code_output, got %v", resp.Result)
	}
}

func TestFanout(t *testing.T) {
	dev1 := &fakeAgent{
		name: "fakedeveloper",
		caps: []string{"implement"},
		handleFn: func(_ context.Context, req core.DelegateRequest) (core.DelegateResponse, error) {
			return core.NewDelegateResponse("fakedeveloper", req.Sender, req.ID, true, "code_output", ""), nil
		},
	}
	dev2 := &fakeAgent{
		name: "dev2",
		caps: []string{"implement"},
		handleFn: func(_ context.Context, req core.DelegateRequest) (core.DelegateResponse, error) {
			return core.NewDelegateResponse("dev2", req.Sender, req.ID, true, "code_output_2", ""), nil
		},
	}
	rt := newTestRuntime(dev1, dev2)

	results, err := patterns.Fanout(context.Background(), rt, []string{"fakedeveloper", "dev2"}, "implement feature", nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	for _, r := range results {
		if !r.Success {
			t.Fatal("expected all success")
		}
	}
	outputs := map[any]bool{}
	for _, r := range results {
		outputs[r.Result] = true
	}
	if !outputs["code_output"] || !outputs["code_output_2"] {
		t.Fatalf("expected both outputs, got %v", outputs)
	}
}

func TestReviewLoopAcceptsOnSecondRound(t *testing.T) {
	dev := &fakeAgent{
		name: "fakedeveloper",
		caps: []string{"implement"},
		handleFn: func(_ context.Context, req core.DelegateRequest) (core.DelegateResponse, error) {
			return core.NewDelegateResponse("fakedeveloper", req.Sender, req.ID, true, "code_output", ""), nil
		},
	}
	reviewer := &fakeAgent{
		name: "fakereviewer",
		caps: []string{"review"},
	}
	reviewer.handleFn = func(_ context.Context, req core.DelegateRequest) (core.DelegateResponse, error) {
		accepted := reviewer.callCount >= 2
		return core.NewDelegateResponse("fakereviewer", req.Sender, req.ID, true,
			map[string]any{"accepted": accepted, "feedback": map[string]any{"hint": "try harder"}}, ""), nil
	}

	rt := newTestRuntime(dev, reviewer)

	result, err := patterns.ReviewLoop(context.Background(), rt, patterns.ReviewOpts{
		Producer:  "fakedeveloper",
		Reviewer:  "fakereviewer",
		Task:      "implement auth",
		MaxRounds: 5,
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.Accepted {
		t.Fatal("expected accepted")
	}
	if result.Rounds != 2 {
		t.Fatalf("expected 2 rounds, got %d", result.Rounds)
	}
	if len(result.History) != 4 {
		t.Fatalf("expected 4 history entries (2 produce + 2 review), got %d", len(result.History))
	}
}

func TestTracesRecorded(t *testing.T) {
	planner := &fakeAgent{
		name: "fakeplanner",
		caps: []string{"plan"},
		handleFn: func(_ context.Context, req core.DelegateRequest) (core.DelegateResponse, error) {
			return core.NewDelegateResponse("fakeplanner", req.Sender, req.ID, true, nil, ""), nil
		},
	}
	rt := newTestRuntime(planner)

	_, err := patterns.Delegate(context.Background(), rt, runtime.DelegateOpts{
		Capability: "plan",
		Task:       "test",
	})
	if err != nil {
		t.Fatal(err)
	}

	traces := rt.Traces()
	if len(traces) != 1 {
		t.Fatalf("expected 1 trace, got %d", len(traces))
	}
	if !traces[0].Success {
		t.Fatal("expected success in trace")
	}
}

func TestUnknownCapabilityErrors(t *testing.T) {
	rt := newTestRuntime()

	_, err := patterns.Delegate(context.Background(), rt, runtime.DelegateOpts{
		Capability: "nonexistent",
		Task:       "x",
	})
	if err == nil {
		t.Fatal("expected error for unknown capability")
	}
	if _, ok := err.(*core.RegistryError); !ok {
		t.Fatalf("expected RegistryError, got %T", err)
	}
}
