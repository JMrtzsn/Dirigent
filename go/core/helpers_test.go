package core_test

import (
	"context"

	"github.com/johannesmauritzson/dirigent/core"
)

// fakeAgent is a test helper implementing the Agent interface.
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
