// Package runtime provides the execution engine that connects agents via the registry.
package runtime

import (
	"context"
	"fmt"
	"time"

	"github.com/johannesmauritzson/dirigent/core"
)

// Config holds runtime configuration.
type Config struct {
	Timeout            time.Duration
	MaxDelegationDepth int
}

// DefaultConfig returns sensible defaults.
func DefaultConfig() Config {
	return Config{
		Timeout:            5 * time.Minute,
		MaxDelegationDepth: 10,
	}
}

// Trace records metadata about a delegation.
type Trace struct {
	RequestID  string
	Sender     string
	Receiver   string
	Capability string
	Task       string
	ResponseID string
	Success    bool
	Error      string
}

// DelegationDepthError is returned when delegation exceeds the configured limit.
type DelegationDepthError struct {
	Depth int
	Limit int
}

func (e *DelegationDepthError) Error() string {
	return fmt.Sprintf("Delegation depth %d exceeds limit %d", e.Depth, e.Limit)
}

// Runtime orchestrates agent execution via capability-based routing.
type Runtime struct {
	registry *core.Registry
	config   Config
	traces   []Trace
	depth    int
}

// New creates a runtime with the given config.
func New(cfg Config) *Runtime {
	return &Runtime{
		registry: core.NewRegistry(),
		config:   cfg,
	}
}

// Registry returns the underlying registry.
func (rt *Runtime) Registry() *core.Registry {
	return rt.registry
}

// Traces returns a copy of recorded traces.
func (rt *Runtime) Traces() []Trace {
	cp := make([]Trace, len(rt.traces))
	copy(cp, rt.traces)
	return cp
}

// Register registers one or more agents.
func (rt *Runtime) Register(agents ...core.Agent) error {
	for _, ag := range agents {
		if err := rt.registry.Register(ag); err != nil {
			return err
		}
	}
	return nil
}

// DelegateOpts specifies a delegation target.
type DelegateOpts struct {
	Capability  string
	To          string
	Task        string
	Context     map[string]any
	Constraints map[string]any
	Sender      string
}

// Delegate dispatches work to an agent by capability or name.
func (rt *Runtime) Delegate(ctx context.Context, opts DelegateOpts) (core.DelegateResponse, error) {
	if opts.Capability == "" && opts.To == "" {
		return core.DelegateResponse{}, fmt.Errorf("must provide either Capability or To")
	}
	if opts.Sender == "" {
		opts.Sender = "__runtime__"
	}

	if rt.depth >= rt.config.MaxDelegationDepth {
		return core.DelegateResponse{}, &DelegationDepthError{Depth: rt.depth, Limit: rt.config.MaxDelegationDepth}
	}

	var agent core.Agent
	var err error
	if opts.To != "" {
		agent, err = rt.registry.ResolveName(opts.To)
	} else {
		agent, err = rt.registry.ResolveCapability(opts.Capability)
	}
	if err != nil {
		return core.DelegateResponse{}, err
	}

	capOrTo := opts.Capability
	if capOrTo == "" {
		capOrTo = opts.To
	}

	req := core.NewDelegateRequest(opts.Sender, agent.Info().Name, capOrTo, opts.Task, opts.Context, opts.Constraints)

	trace := Trace{
		RequestID:  req.ID,
		Sender:     req.Sender,
		Receiver:   req.Receiver,
		Capability: req.Capability,
		Task:       req.Task,
	}

	rt.depth++
	defer func() { rt.depth-- }()

	timeoutCtx, cancel := context.WithTimeout(ctx, rt.config.Timeout)
	defer cancel()

	type result struct {
		resp core.DelegateResponse
		err  error
	}
	ch := make(chan result, 1)
	go func() {
		resp, err := agent.Handle(timeoutCtx, req)
		ch <- result{resp, err}
	}()

	select {
	case <-timeoutCtx.Done():
		resp := core.NewDelegateResponse(agent.Info().Name, opts.Sender, req.ID, false, nil, fmt.Sprintf("Timed out after %s", rt.config.Timeout))
		trace.ResponseID = resp.ID
		trace.Success = false
		trace.Error = resp.Error
		rt.traces = append(rt.traces, trace)
		return resp, nil
	case r := <-ch:
		if r.err != nil {
			resp := core.NewDelegateResponse(agent.Info().Name, opts.Sender, req.ID, false, nil, r.err.Error())
			trace.ResponseID = resp.ID
			trace.Success = false
			trace.Error = resp.Error
			rt.traces = append(rt.traces, trace)
			return resp, nil
		}
		trace.ResponseID = r.resp.ID
		trace.Success = r.resp.Success
		trace.Error = r.resp.Error
		rt.traces = append(rt.traces, trace)
		return r.resp, nil
	}
}
